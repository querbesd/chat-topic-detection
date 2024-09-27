# Chat Topic Detection

This project is designed to automatically detect one or several predefined topics in chat conversations with our customer support. The project also assesses the customer satisfaction at the end of the conversation. The goal is to 1. enrich the 360° customer view at the source of some triggering automations 2. collect more insights on the reasons why the customers are contact us.


## Table of Contents

- [Pre-requisites](#pre-requisites)
- [Code Description](#code-description)
- [Folder Structure](#folder-structure)
- [Installation Steps](#installation-steps)

## Pre-requisites

### OpenAI API key

This project relies on OpenAI LLM. To access OpenAI models, you need to create an API on https://platform.openai.com/

The project uses a [fine-tuned](https://platform.openai.com/docs/guides/fine-tuning) model stored in OpenAI. This is highly recommended if you want to improve the accuracy of the predictions and if your topic list present nuances. However, the project also works with a base model like gpt-3.5-turbo-1106. OpenAI models and their price constantly evolve. Check the model list to see if there is a better model fitting your use case and budget.

### Cloud solution

This project has been built on GCP cloud solution. But it can be adapted to other cloud solutions. A deployment to a [GCP cloud function](https://cloud.google.com/functions?hl=en) can be considered for a regular execution of the script.

### Conversations input data

The conversation input data should be stored in a table of your data warehouse. This requires you previously extracted your conversations to your datawarehouse. A preprocessing is also needed to match the below specific schema. The conversation array contains messages ranked in the ascending order.

```
[
  {
    "name": "conversation_id",
    "type": "STRING",
  },
  {
    "name": "client_id",
    "type": "STRING",
  },
  {
    "name": "created_at",
    "type": "TIMESTAMP",
    "description": "Timestamp of the first chat message sent by the customer"
  },
  {
    "name": "conversation",
    "mode": "REPEATED",
    "type": "RECORD"
    "fields": [
      {
        "name": "body",
        "type": "STRING",
        "description": "message of the customer or of the agent"
      },
      {
        "name": "extracted_type",
        "type": "STRING",
        "description": "'user' or 'admin'",
      }
    ]
  }
]
```


## Code description

- Fetch conversation data from the input table and create the input requests file to be ingested by the LLM. The file is stored in a temporary file.
- Performs batch predictions asynchroneously. `batch_predict.py` parallelizes requests to the OpenAI API while throttling to stay under rate limits. The script for the batch predicting was inspired from the openai-cookbook (see code [here](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py))
- Save the output file locally or in a GCS bucket.

## Folder Structure

```bash
.
├── config/
│   ├── env.yaml
│   ├── params.py
│   └── .envrc
├── src/
│   ├── data/
│   │   └── data.py
│   ├── model/
│   │   └── batch_predict.py
│   ├── utils/
│   │   └── utils.py
│   ├── main.py
└── requirements.txt

```

## Installation steps


1. Clone the repository

```
git clone https://github.com/querbesd/chat-topic-detection.git
cd intercom-chat-topic-detection
```

2. Create a virtual environment

```
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
# or
venv\Scripts\activate  # For Windows
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Set up your env variables and script params:
  - define your env variables in env.yaml
  - update your topics list in params.py
  - customize the prompt in data.py

5. Run the script

```
python src/main.py
```
