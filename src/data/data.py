from google.cloud import bigquery
from colorama import Fore, Style
from google.oauth2 import service_account
import json
import pandas as pd
from config.params import *


class DataFactory:
    def get_bq_data(self):

        QUERY = f"""
                SELECT * FROM `{GCP_PROJECT}.{BQ_DATASET_SOURCE}.{BQ_TABLE_SOURCE}`
                WHERE DATE(created_at) BETWEEN '{ST_DATE}' AND DATE_SUB(CURRENT_DATE(), INTERVAL 5 DAY)
                LIMIT {LIMIT_ROWS}
                """

        client = bigquery.Client()

        # perform a query.
        query_job = client.query(QUERY)  # API request

        # get result
        result = query_job.result()

        # store results to a dataframe
        df = result.to_dataframe()
        return df


    def create_requests_file(self, df, filename):
        jobs = []
        for conversation_id, conversation in zip(df['conversation_id'], df['conversation']):
            # construct the user_message for each conversation
            user_message = f'''
            The company is named XXXX and delivers XXXXX.

            Below is a chat discussion delimited with {delimiter}.

            The customer's messages are identified as "user".

            The agent's messages are identified as "admin".

            The conversation's messages are ordered from the oldest to the most recent one. To identify the topics, the first message is likely to be the most important one.

            Please, identify the main topics mentioned in this comment from the list of topics below.

            Return a list of the relevant topics for the conversation.

            Also, perform some sentiment analysis on the user's messages from 0 to 1. 0 means the user is very frustrated and pissed off. 1 means the user is satisfied by the resolution, polite and nice.

            Output is a JSON list with the following elements:
            relevant_topics: [<topic1>, <topic2>, ...],
            sentiment_analysis: score

            If topics are not relevant to the conversation, return an empty list ([]).

            Include only topics from the provided below list.

            List of topics:
            {", ".join(topics_list_str)}

            conversation:
            {delimiter}
            {conversation}
            {delimiter}
            '''
            # assign model name to the model variable
            model = MODEL

            jobs.append({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "response_format": { "type": "json_object" },
                "temperature": 0,
                # "max_tokens": 2000,
                "metadata": {"conversation_id": conversation_id,
                            "conversation": conversation.tolist()}

            })

        # write jobs to a .jsonl file
        with open(filename, "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")

        print("Request file created!")
