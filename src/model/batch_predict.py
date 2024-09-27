"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Example command to call script:
```
python batch_predict.py \
  --requests_filepath test_completion_requests.jsonl \
  --save_filepath test_completion_requests_results.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 80000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20
```

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
    - file should be a jsonl file, where each line is a json object with API parameters and an optional metadata field
    - e.g., {"model": "text-embedding-3-small", "input": "embed me", "metadata": {"row_id": 1}}
    - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
    - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
    - the code to generate the example file is appended to the bottom of this script
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-3-small", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-3-small`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 0, 1, 2, ...)
    - Run main()
"""

# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from src.utils import *
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata

@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0

@dataclass
class APIRequest:
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict = field(default_factory=dict)
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_tmp_filepath: str,
        status_tracker: StatusTracker,
        temperature: float = 0,  # Add temperature as a parameter with a default value
        #max_tokens: int = 2000
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None

        # Add or update the temperature parameter in the request JSON
        #self.request_json['temperature'] = temperature
        # Add or update the max token parameter in the request JSON
        #self.request_json['max_tokens'] = max_tokens
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_tmp_filepath)
                #save_file(data)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_tmp_filepath)
            #save_file(data)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_tmp_filepath}")

class Predictor:
    def __init__(self, requests_filepath, request_url, save_tmp_filepath, api_key, max_requests_per_minute, max_tokens_per_minute, token_encoding_name, max_attempts, logging_level):
        self.requests_filepath = requests_filepath
        self.request_url = request_url
        self.save_tmp_filepath = save_tmp_filepath
        self.api_key = api_key
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.token_encoding_name = token_encoding_name
        self.max_attempts = max_attempts
        self.logging_level = logging_level
        logging.basicConfig(level=logging_level)  # Initialize logging based on the provided logging level


    async def process_api_requests_from_file(self, request_url):
        """Processes API requests in parallel, throttling to stay under rate limits."""

        # constants
        seconds_to_pause_after_rate_limit_error = 15
        seconds_to_sleep_each_loop = 0.001

        logging.debug("Logging initialized at level {}".format(self.logging_level))

        # infer API endpoint and construct request header
        api_endpoint = api_endpoint_from_url(request_url)
        request_header = {"Authorization": f"Bearer {self.api_key}"}
        # use api-key header for Azure deployments
        if "/deployments" in request_url:
            request_header = {"api-key": f"{self.api_key}"}

        # initialize trackers
        queue_of_requests_to_retry = asyncio.Queue()
        task_id_generator = task_id_generator_function()
        self.status_tracker = StatusTracker()
        next_request = None  # variable to hold the next request to call

        # initialize available capacity counts
        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()

        # initialize flags
        file_not_finished = True  # after file is empty, we'll skip reading it
        logging.debug(f"Initialization complete.")

        # initialize file reading
        with open(self.requests_filepath) as file:
            requests = file.__iter__()

            async with aiohttp.ClientSession() as session:
                while True:
                # get next request (if one is not already waiting for capacity)
                    if next_request is None:
                        if not queue_of_requests_to_retry.empty():
                            next_request = queue_of_requests_to_retry.get_nowait()
                            logging.debug(
                                f"Retrying request {next_request.task_id}: {next_request}"
                            )
                        elif file_not_finished:
                            try:
                                # get new request
                                request_json = json.loads(next(requests))

                                # request_json['max_tokens'] = 2000

                                # truncate request_json message if it exceeds max_token as input
                                request_json = truncate_if_exceeds(request_json, self.token_encoding_name)

                                next_request = APIRequest(
                                    task_id=next(task_id_generator),
                                    request_json=request_json,
                                    token_consumption=num_tokens_consumed_from_request(
                                        request_json, api_endpoint, self.token_encoding_name),
                                    attempts_left=self.max_attempts,
                                    metadata=request_json.pop("metadata", None),
                                )
                                self.status_tracker.num_tasks_started += 1
                                self.status_tracker.num_tasks_in_progress += 1
                                logging.debug(
                                    f"Reading request {next_request.task_id}: {next_request}"
                                )
                            except StopIteration:
                                # if file runs out, set flag to stop reading it
                                logging.debug("Read file exhausted")
                                file_not_finished = False

                    # update available capacity
                    current_time = time.time()
                    seconds_since_update = current_time - last_update_time
                    available_request_capacity = min(
                        available_request_capacity
                        + self.max_requests_per_minute * seconds_since_update / 60.0,
                        self.max_requests_per_minute,
                    )
                    available_token_capacity = min(
                        available_token_capacity
                        + self.max_tokens_per_minute * seconds_since_update / 60.0,
                        self.max_tokens_per_minute,
                    )
                    last_update_time = current_time

                    # if enough capacity available, call API
                    if next_request:
                        next_request_tokens = next_request.token_consumption
                        if (
                            available_request_capacity >= 1
                            and available_token_capacity >= next_request_tokens
                        ):
                            # update counters
                            available_request_capacity -= 1
                            available_token_capacity -= next_request_tokens
                            next_request.attempts_left -= 1

                            # call API
                            asyncio.create_task(
                                next_request.call_api(
                                    session=session,
                                    request_url=request_url,
                                    request_header=request_header,
                                    retry_queue=queue_of_requests_to_retry,
                                    save_tmp_filepath=self.save_tmp_filepath,
                                    status_tracker=self.status_tracker,
                                )
                            )
                            next_request = None  # reset next_request to empty

                    # if all tasks are finished, break
                    if self.status_tracker.num_tasks_in_progress == 0:
                        break

                    # main loop sleeps briefly so concurrent tasks can run
                    await asyncio.sleep(seconds_to_sleep_each_loop)

                    # if a rate limit error was hit recently, pause to cool down
                    seconds_since_rate_limit_error = (
                        time.time() - self.status_tracker.time_of_last_rate_limit_error
                    )
                    if (
                        seconds_since_rate_limit_error
                        < seconds_to_pause_after_rate_limit_error
                    ):
                        remaining_seconds_to_pause = (
                            seconds_to_pause_after_rate_limit_error
                            - seconds_since_rate_limit_error
                        )
                        await asyncio.sleep(remaining_seconds_to_pause)
                        # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                        logging.warn(
                            f"Pausing to cool down until {time.ctime(self.status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                        )

        if len(self.save_tmp_filepath):
            save_file(self.save_tmp_filepath)

        else:
            logging.info("Error faced when getting results from{}".format(self.save_tmp_filepath))
