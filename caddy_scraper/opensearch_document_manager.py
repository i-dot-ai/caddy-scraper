import asyncio

import glob
import json
import os
from typing import List, Callable, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import BedrockEmbeddings
from opensearchpy import OpenSearch, helpers
import pandas as pd


class OpenSearchDocumentManager:
    def __init__(self, client: OpenSearch, index_name: str = "caddy-hybrid-search-index"):
        self.client = client
        self.index_name = index_name
        self.embedding_model = BedrockEmbeddings(
            model_id="cohere.embed-english-v3", region_name="eu-west-3"
        )
        
    def create_index(self):
        index_body = {
            "settings": {
                "index": {"knn": True}
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "domain": {"type": "keyword"},
                    "text_vector": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                        "engine": "faiss",
                        "space_type": "l2",
                        "name": "hnsw",
                        "parameters": {}
                    }
                    }
                }
            }
        }
        # Delete index if it exists
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        # Create new index
        self.client.indices.create(index=self.index_name, body=index_body)
    
    async def async_bulk_upload(self, file_path: str, domain: str = "citizen-advice"):
        json_files = glob.glob(os.path.join(file_path, "scrape_result_*.json"))
        for file in json_files:
            with open(file) as f:
                df = pd.DataFrame(json.load(f))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=100,
                length_function=len,
            )

            loader = DataFrameLoader(df, page_content_column="markdown")
            docs = loader.load()

            # Citizen Advice Specific Logic to remove low quality docs
            docs = [d for d in docs if d.metadata['markdown_length'] > 1000]
            docs = [d for d in docs if "cymraeg" not in d.metadata['source']]

            docs = text_splitter.split_documents(docs)

            embeddings = await self._gather_with_concurrency(
                10,
                *[
                    self.embedding_model.aembed_documents(
                        [d.page_content for d in docs]
                    )
                ],
            )
            success, failed = helpers.bulk(
                self.client, 
                self._generate_bulk_actions(docs, embeddings, domain)
            )
            print(f"File {file} - Uploaded: {success}, Failed: {failed}")

    async def _gather_with_concurrency(
        self, concurrency: int, *coroutines: List[Callable]
    ) -> List[Any]:
        """Run a number of async coroutines with a concurrency limit.

        Args:
            concurrency (int): max number of concurrent coroutine runs.
            coroutines (List[Callable]): list of coroutines to run asynchronously.

        Returns:
            List[Any]: list of coroutine results.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def semaphore_coroutine(coroutines):
            async with semaphore:
                return await coroutines

        return await asyncio.gather(*(semaphore_coroutine(c) for c in coroutines))

    def _generate_bulk_actions(self, documents, embeddings, domain="citizen-advice"):
        for i, (doc, vector) in enumerate(zip(documents, embeddings[0])):
            action = {
                    "_index": self.index_name,
                    "_id": str(i),
                    "_source": {
                        "text": doc.page_content,
                        "text_vector": vector,
                        "source": doc.metadata["source"],
                        "domain": domain
                    }
                }
            yield action
