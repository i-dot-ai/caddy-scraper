import os
import json
from datetime import datetime
import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from dotenv import load_dotenv
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
opensearch_https = os.environ.get("OPENSEARCH_HTTPS")
bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID")
bucket_name = os.environ.get("S3_BUCKET_NAME")
collection_id = os.environ.get("OPENSEARCH_COLLECTION_ID")
vector_store_name = os.environ.get("OPENSEARCH_VECTOR_STORE_NAME")

region = "eu-west-2"
service = "es"
session = boto3.Session()
credentials = session.get_credentials()
auth_creds = AWS4Auth(
    region=region,
    service=service,
    refreshable_credentials=credentials,
)

# Create IAM client and Bedrock execution role
iam_client = session.client('iam')
sts_client = session.client('sts')
opensearch_client = session.client('opensearch')  # Changed from 'aoss' to 'opensearch'

# Variables for unique naming
suffix = random.randrange(200, 900)
account_number = sts_client.get_caller_identity().get('Account')
identity = sts_client.get_caller_identity()['Arn']
bedrock_execution_role_name = f'AmazonBedrockExecutionRoleForKnowledgeBase_{suffix}'
fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{suffix}'
s3_policy_name = f'AmazonBedrockS3PolicyForKnowledgeBase_{suffix}'
oss_policy_name = f'AmazonBedrockOSSPolicyForKnowledgeBase_{suffix}'
encryption_policy_name = f"bedrock-sample-rag-sp-{suffix}"
network_policy_name = f"bedrock-sample-rag-np-{suffix}"
access_policy_name = f'bedrock-sample-rag-ap-{suffix}'

def create_bedrock_execution_role(bucket_name):
    foundation_model_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v1"
                ]
            }
        ]
    }

    s3_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceAccount": f"{account_number}"
                    }
                }
            }
        ]
    }

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    # create policies based on the policy documents
    fm_policy = iam_client.create_policy(
        PolicyName=fm_policy_name,
        PolicyDocument=json.dumps(foundation_model_policy_document),
        Description='Policy for accessing foundation model',
    )

    s3_policy = iam_client.create_policy(
        PolicyName=s3_policy_name,
        PolicyDocument=json.dumps(s3_policy_document),
        Description='Policy for reading documents from s3')

    # create bedrock execution role
    bedrock_kb_execution_role = iam_client.create_role(
        RoleName=bedrock_execution_role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description='Amazon Bedrock Knowledge Base Execution Role for accessing OSS and S3',
        MaxSessionDuration=3600
    )

    # fetch arn of the policies and role created above
    bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
    s3_policy_arn = s3_policy["Policy"]["Arn"]
    fm_policy_arn = fm_policy["Policy"]["Arn"]

    # attach policies to Amazon Bedrock execution role
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=fm_policy_arn
    )
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=s3_policy_arn
    )
    return bedrock_kb_execution_role

def create_oss_policy_attach_bedrock_execution_role(collection_id, bedrock_kb_execution_role):
    # define oss policy document
    oss_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:APIAccessAll"
                ],
                "Resource": [
                    f"arn:aws:aoss:{region}:{account_number}:collection/{collection_id}"
                ]
            }
        ]
    }
    oss_policy = iam_client.create_policy(
        PolicyName=oss_policy_name,
        PolicyDocument=json.dumps(oss_policy_document),
        Description='Policy for accessing opensearch serverless',
    )
    oss_policy_arn = oss_policy["Policy"]["Arn"]
    print("Opensearch serverless arn: ", oss_policy_arn)

    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=oss_policy_arn
    )
    return None

def create_policies_in_oss(vector_store_name, aoss_client, bedrock_kb_execution_role_arn):
    encryption_policy = aoss_client.create_security_policy(
        name=encryption_policy_name,
        policy=json.dumps(
            {
                'Rules': [{'Resource': ['collection/' + vector_store_name],
                           'ResourceType': 'collection'}],
                'AWSOwnedKey': True
            }),
        type='encryption'
    )

    network_policy = aoss_client.create_security_policy(
        name=network_policy_name,
        policy=json.dumps(
            [
                {'Rules': [{'Resource': ['collection/' + vector_store_name],
                            'ResourceType': 'collection'}],
                 'AllowFromPublic': True}
            ]),
        type='network'
    )
    access_policy = aoss_client.create_access_policy(
        name=access_policy_name,
        policy=json.dumps(
            [
                {
                    'Rules': [
                        {
                            'Resource': ['collection/' + vector_store_name],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': ['index/' + vector_store_name + '/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'],
                            'ResourceType': 'index'
                        }],
                    'Principal': [identity, bedrock_kb_execution_role_arn],
                    'Description': 'Easy data policy'}
            ]),
        type='data'
    )
    return encryption_policy, network_policy, access_policy

def setup_bedrock_role_and_policies():
    bedrock_kb_execution_role = create_bedrock_execution_role(bucket_name)
    create_oss_policy_attach_bedrock_execution_role(collection_id, bedrock_kb_execution_role)
    bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
    create_policies_in_oss(vector_store_name, opensearch_client, bedrock_kb_execution_role_arn)

def generate_rolling_vectorstore_name():
    """Generates a name name for the index based on todays day

    Name is formatted as index_name_YYYYMMDD

    Returns:
        rolling_index_name (string): the name of the index with the date appended

    """
    name_suffix = os.environ.get("OPENSEARCH_INDEX_SUFFIX")
    today = datetime.now().strftime("%Y%m%d")
    rolling_index_name = f"{name_suffix}_{today}"
    logger.info(f"New index name: {rolling_index_name}")
    return rolling_index_name

def list_bedrock_models():
    bedrock_client = session.client('bedrock', region_name=region)
    response = bedrock_client.list_foundation_models()
    return response['Models']

def validate_model_id(model_id):
    models = list_bedrock_models()
    model_ids = [model['ModelId'] for model in models]
    if model_id not in model_ids:
        raise ValueError(f"Model ID {model_id} is not available. Available models: {model_ids}")

# Validate the model ID before use
validate_model_id(bedrock_model_id)

# Initialize the Bedrock embeddings model
embedding_model = BedrockEmbeddings(model_id=bedrock_model_id)

# Set up the IAM role and policies required for Bedrock and OpenSearch
setup_bedrock_role_and_policies()

# Generate the index name
index_name = generate_rolling_vectorstore_name()

# Set up the OpenSearch vector store
vectorstore = OpenSearchVectorSearch(
    index_name=index_name,
    opensearch_url=opensearch_https,
    http_auth=auth_creds,
    embedding_function=embedding_model,
    connection_class=RequestsHttpConnection,
)
