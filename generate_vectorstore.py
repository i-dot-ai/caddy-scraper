import os
from datetime import datetime

import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection


from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import OpenSearchVectorSearch

from dotenv import load_dotenv

load_dotenv()
opensearch_https = os.environ.get("OPENSEARCH_URL")
# embedding_model_name = os.environ.get("HF_EMBEDDING_MODEL")
bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID")

region = "eu-west-3"
service = "aoss"
session = boto3.Session()
credentials = session.get_credentials()
auth_creds = AWS4Auth(
    region=region,
    service=service,
    refreshable_credentials=credentials,
)

embedding_model = BedrockEmbeddings(
    model_id=bedrock_model_id, region_name=region)


def generate_rolling_vectorstore_name():
    """Generates a name name for the index based on todays day

    Name is formatted as index_name_YYYYMMDD

    Returns:
        rolling_index_name (string): the name of the index with the date appended

    """

    name_suffix = os.environ.get("OPENSEARCH_INDEX_SUFFIX")

    today = datetime.now().strftime("%Y%m%d")
    rolling_index_name = f"{name_suffix}_{today}"

    # output the new name
    print(f"New index name: {rolling_index_name}")
    return rolling_index_name


index_name = generate_rolling_vectorstore_name()

vectorstore = OpenSearchVectorSearch(
    index_name=index_name,
    opensearch_url=opensearch_https,
    http_auth=auth_creds,
    embedding_function=embedding_model,
    connection_class=RequestsHttpConnection,
)
