"""Vector store manager module."""

import asyncio
import json
import os
from tqdm import tqdm
from typing import Any, Callable, List, Tuple

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import BedrockEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core import embeddings, documents
from langchain_text_splitters import RecursiveCharacterTextSplitter, base
from opensearchpy import RequestsHttpConnection
import pandas as pd

from core_utils import retry


class VectorStoreManager:
    """VectorStoreManager definition."""

    def __init__(
        self,
        index_name: str,
        authentication_creds: Tuple[str, str],
        opensearch_url: str,
        scrape_output_path: str = "scrape_result",
        embedding_model: str = "bedrock",
        delete_existing_index: bool = True,
    ):
        """Initialise VectorStoreManager.

        Args:
            index_name (str): name of open search index for vectorstore.
            authentication_creds (Tuple[str, str]): open search admin credentials (username, password).
            opensearch_url (str): open search url.
            scrape_output_path (str): dir containing json scrape results. Defaults to 'scrape_result/result.json'.
            embedding_model (str): which embedding model to use ['bedrock', 'huggingface']. Defaults to 'bedrock'.
            delete_existing_index (bool): whether or not to delete an index of the same name if it exists. Defaults to True.
        """
        self.index_name = index_name
        self.scrape_output_path = scrape_output_path
        self.embedding_model = self.get_embedding_model(embedding_model)
        self.vectorstore = self.get_vectorstore(
            index_name, opensearch_url, authentication_creds, delete_existing_index
        )
        self.text_splitter = self.get_text_splitter()

    async def run(self):
        """Run VectorStoreManager, loading documents and uploading them to vectorstore."""
        for file in tqdm(os.listdir(self.scrape_output_path)):
            path = os.path.join(self.scrape_output_path, file)
            docs = self.load_documents(path)
            await self.add_documents_to_vectorstore(docs)

    def get_embedding_model(self, embedding_model: str) -> embeddings.Embeddings:
        """Get an embedding model for the vectorstore.

        Args:
            embedding_model (str): which embedding model to use ['bedrock', 'huggingface']

        Returns:
            Embeddings: loaded embedding model.
        """
        if embedding_model == "bedrock":
            return BedrockEmbeddings(
                model_id="cohere.embed-english-v3", region_name="eu-west-3"
            )
        elif embedding_model == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(
                "Invalid embedding model, must be one of  ['bedrock', 'huggingface']."
            )

    def get_text_splitter(self) -> base.TextSplitter:
        """Get a text splitter for chunking of longer documents.

        Returns:
            base.TextSplitter: loaded text splitter
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=100,
            length_function=len,
        )

    def get_vectorstore(
        self,
        index_name: str,
        opensearch_url: str,
        authentication_creds: Tuple[str, str],
        delete_existing: bool,
    ) -> OpenSearchVectorSearch:
        """Get vectorstore, includes authentication.

        Args:
            index_name (str): name of index where documents will be stored.
            opensearch_url (str): opensearch endpoint.
            authentication_creds (Tuple[str, str]): open search admin credentials (username, password).
            delete_existing (bool): whether or not to delete an existing index with the same name.

        Returns:
            OpenSearchVectorSearch: loaded vectorstore.
        """
        vectorstore = OpenSearchVectorSearch(
            index_name=index_name,
            opensearch_url=opensearch_url,
            http_auth=authentication_creds,
            embedding_function=self.embedding_model,
            connection_class=RequestsHttpConnection,
        )
        if delete_existing:
            if vectorstore.index_exists(index_name):
                vectorstore.delete_index(index_name)
        return vectorstore

    def load_documents(self, file_path: str) -> List[documents.base.Document]:
        """Load documents from file.

        Args:
            file_path (str): file path to json storing scrape results.

        Returns:
            List[documents.base.Document]: list of loaded langchain documents.
        """
        with open(file_path) as f:
            df = pd.DataFrame(json.load(f))
        df["raw_markdown"] = df["markdown"]
        loader = DataFrameLoader(df, page_content_column="markdown")
        docs = loader.load()
        return self.text_splitter.split_documents(docs)

    @retry()
    async def add_documents_to_vectorstore(
        self, document_list: List[documents.base.Document], bulk_size: int = 500
    ):
        """Takes a list of documents, and adds them to the vectorstore in bulk

        Args:
            document_list (List[documents.base.Document]): list of documents
            bulk_size (int): the size of the bulk to add the documents in
        """
        embeddings = await self.gather_with_concurrency(
            10,
            *[
                self.embedding_model.aembed_documents(
                    [d.page_content for d in document_list]
                )
            ],
        )
        for i in range(0, len(document_list), bulk_size):
            doc_batch =  document_list[i : i + bulk_size]
            self.vectorstore.add_embeddings(
                text_embeddings=list(
                    zip([d.page_content for d in doc_batch], embeddings[0][i : i + bulk_size])
                ),
                metadatas=[d.metadata for d in doc_batch],
                index_name=self.index_name,
                bulk_size=20000,
            )

    async def gather_with_concurrency(
        self, concurrency: int, *coroutines: List[Callable]
    ) -> List[Any]:
        """ "Run a number of async coroutines with a concurrency limit.

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
