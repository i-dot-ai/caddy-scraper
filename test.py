import boto3
import logging
from botocore.exceptions import ClientError
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize AWS Bedrock client
region = 'eu-west-3'
model_id = "cohere.embed-english-v3"
session = boto3.Session()
client = session.client('bedrock', region_name=region)

# Print the model ID to verify it's being loaded correctly
print(f"Bedrock Model ID: {model_id}")

try:
    response = client.list_foundation_models()
    print('a list of embedding models ')
    print(response)
    print('list of embedding models over')
    
    # Test invoking the Bedrock model
    response = client.add_embeddings(
        Body="Test text for embedding"
    )
    print(response)
except ClientError as e:
    logging.error(f"ClientError: {e}")
    print(f"ClientError: {e}")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    print(f"Unexpected error: {e}")
