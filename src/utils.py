import re  # for matching endpoint from request URL
import json
import tiktoken
import time
import logging
from google.cloud import storage, bigquery
from google.oauth2 import service_account
from config.params import *

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")



def truncate_if_exceeds(request_json: dict,
    token_encoding_name: str):
    """Count the number of tokens in the request_json message"""
    encoding = tiktoken.get_encoding(token_encoding_name)
    max_tokens = 2000

    # Apply the tokenization function to your text
    tokens = encoding.encode(request_json["messages"][1]["content"])

    if len(encoding.encode(request_json["messages"][1]["content"])) > max_tokens:
        truncated_tokens = tokens[:2000]

        truncated_content = encoding.decode(truncated_tokens)
        request_json["messages"][1]["content"] = truncated_content
    return request_json


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_token", 20)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def save_file(filepath:str):
    """Uploads data to the specified bucket or LOCALLY"""

    if LOCATION=="LOCAL":
        append_to_jsonl(filepath, SAVE_FILEPATH)
        logging.info("Parallel processing complete. Results saved to {}".format(SAVE_FILEPATH))
        return None

    elif LOCATION == "GCP":

        storage_client = storage.Client()
        filename = SAVE_FILEPATH.split("/")[-1]
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"models/{filename}")

        blob.upload_from_filename(filepath)  # Uploads the file
        logging.info(f"Data uploaded to {filename} in bucket {GCS_BUCKET}.")
    else:
        raise ValueError(f"Unknown LOCATION value: {LOCATION}")

def restructure_original_file(filepath:str):
    """Uploads data to the specified bucket or LOCALLY"""


    content = []  # Initialize an empty list to hold the JSON objects

    with open("tmp.jsonl", 'w') as file:
        for line in file:
            json_object = json.loads(line)
            content.append(json_object)

    rows_to_insert = [ {'conversation_id': row[2]['conversation_id'],
                        'conversation': row[2]['conversation'],
                        'prediction': row[1]
    }
         for row in content]
    jsonl_string = '\n'.join(json.dumps(row) for row in rows_to_insert)

    storage_client = storage.Client()
    filename = SAVE_FILEPATH.split("/")[-1]
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"models/restructured/{filename}")

    blob.upload_from_string(jsonl_string, content_type='application/json')

    logging.info(f"Data uploaded to {filename} in bucket {GCS_BUCKET}.")
