from src.data.data import DataFactory
from src.model.batch_predict import Predictor
import asyncio
import os
from config.params import *
from dotenv import load_dotenv, find_dotenv


async def main():

    # delete tmp file if any
    if os.path.exists(SAVE_TMP_FILEPATH):
        # delete the file
        os.remove(SAVE_TMP_FILEPATH)
        print(f"The file {SAVE_TMP_FILEPATH} has been deleted.")

    # open-api-key is mounted as a volume in the cloud function
    secret_location = '/secret/openai-api-key'
    with open(secret_location, 'r') as secret_file:
        OPENAI_API_KEY= secret_file.read().strip()

    predictor = Predictor(
        requests_filepath=REQUESTS_FILEPATH,
        save_tmp_filepath=SAVE_TMP_FILEPATH,
        request_url=REQUEST_URL,
        api_key=OPENAI_API_KEY,
        max_requests_per_minute=MAX_REQUESTS_PER_MINUTE,
        max_tokens_per_minute=MAX_TOKENS_PER_MINUTE,
        token_encoding_name=TOKEN_ENCODING_NAME,
        max_attempts=MAX_ATTEMPTS,
        logging_level=LOGGING_LEVEL
    )

    # fetch data and structure the conversation file to apply predictions on it
    df = DataFactory().get_bq_data()

    DataFactory().create_requests_file(df,REQUESTS_FILEPATH)

    # predict and save data locally or to GCP
    await predictor.process_api_requests_from_file(request_url=REQUEST_URL)

    # fetch data from the cloud bucket and push it to bigquery
    print("Data loaded to BigQuery")

def entry_point(request, context):
    # This function will be the entry point for the Google Cloud Function

    # Now, run the main async function
    asyncio.run(main())

    return 'The function completed successfully.'
