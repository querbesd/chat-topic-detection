import tempfile
import os
import time
date = time.strftime("%Y%m%d")


LOCATION = os.environ.get("LOCATION")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join("exports", "data")
LOCAL_REGISTRY_PATH =  os.path.join("exports", "prediction_outputs")
TMP_REGISTRY_PATH = tempfile.gettempdir()

GCP_PROJECT = os.environ.get("GCP_PROJECT")
LIMIT_ROWS = int(os.environ.get("LIMIT_ROWS"))
MODEL = os.environ.get("MODEL")
ST_DATE = os.environ.get("ST_DATE")

GCS_BUCKET =  os.environ.get("GCP_BUCKET")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_DATASET_SOURCE = os.environ.get("BQ_DATASET_SOURCE")
BQ_TABLE = os.environ.get("BQ_TABLE")
BQ_TABLE_SOURCE = os.environ.get("BQ_TABLE_SOURCE")
BQ_REGION = os.environ.get("BQ_REGION")
GBQ_KEYPATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_DC")

REQUESTS_FILEPATH = os.getenv('REQUESTS_FILEPATH')

SAVE_FILEPATH = os.path.join(LOCAL_REGISTRY_PATH, f"{date}.jsonl")
SAVE_TMP_FILEPATH = os.path.join(TMP_REGISTRY_PATH, f"{date}.jsonl")
API_KEY = os.getenv('OPENAI_API_KEY')
REQUEST_URL = os.getenv('REQUEST_URL')
MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE'))
MAX_TOKENS_PER_MINUTE = int(os.getenv('MAX_TOKENS_PER_MINUTE'))
TOKEN_ENCODING_NAME = os.getenv('TOKEN_ENCODING_NAME')
MAX_ATTEMPTS = int(os.getenv('MAX_ATTEMPTS'))
LOGGING_LEVEL = int(os.getenv('LOGGING_LEVEL'))



topics_list_str = ['Topic_1',
                    'Topic_2',
                    'Topic_3'] # List all topics

delimiter = '####'

system_message = "You're a helpful assistant. Your task is to analyse chat discussions between a customer and the agent of an ecommerce website and give the output in JSON."
