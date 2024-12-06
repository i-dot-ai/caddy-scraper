from opensearchpy import OpenSearch
import requests
from langchain_community.embeddings import BedrockEmbeddings
from langchain.schema import Document
from typing import List


class OpenSearchQueryEngine:
    def __init__(
        self,
        client: OpenSearch,
        index_name: str = "caddy-hybrid-search-index",
        lexical_weight: float = 0.3,
        vector_weight: float = 0.7
    ):
        self.client = client
        self.index_name = index_name
        self.embedding_model = BedrockEmbeddings(
            model_id="cohere.embed-english-v3", region_name="eu-west-3"
        )
        # Create the search pipeline for normalization
        self._create_search_pipeline(lexical_weight, vector_weight)

    def _create_search_pipeline(self, lexical_weight: float, vector_weight: float):
        """Create a search pipeline with normalization processor"""
        pipeline_config = {
            "description": "Post processor for hybrid search",
            "phase_results_processors": [{
                "normalization-processor": {
                    "normalization": {
                        "technique": "min_max"
                    },
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [lexical_weight, vector_weight]  
                        }
                    }
                }
            }]
        }
        
        try:
            self.client.transport.perform_request(
                'PUT',
                '/_search/pipeline/hybrid-search-pipeline',
                body=pipeline_config
            )
        except Exception as e:
            print(f"Warning: Failed to create search pipeline: {e}")

    def submit_hybrid_search(self, query_text: str, keywords: str, n_neighbours: int = 2, n_results: int = 5):
        """Perform hybrid search combining lexical and vector search with normalization.
        
        Args:
            query_text (str): Text to generate vector embedding from
            keywords (str): Keywords for lexical search
            n_neighbours (int): Number of nearest neighbors for KNN
            n_results (int): Number of results to return
            
        Returns:
            dict: OpenSearch response containing search results
        """
        query_vector = self.embedding_model.embed_query(query_text)
        
        search_query = {
            "size": n_results,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                "text": {
                                    "query": keywords
                                }
                            }
                        },
                        {
                            "knn": {
                                "text_vector": {
                                    "vector": query_vector,
                                    "k": n_neighbours
                                }
                            }
                        }
                    ]
                }
            }
        }

        # Use the search pipeline for normalization
        response = self.client.search(
            body=search_query,
            index=self.index_name,
            search_pipeline="hybrid-search-pipeline"
        )
        return self.convert_opensearch_to_langchain(response)

    def submit_vector_search(self, query_text: str, n_results: int = 5):
        """Perform pure vector search without lexical matching.
        
        Args:
            query_text (str): The search query text
            n_results (int): Number of results to return
            
        Returns:
            dict: OpenSearch response containing search results
        """
        query_vector = self.embedding_model.embed_query(query_text)
        
        search_query = {
            "size": n_results,
            "query": {
                "knn": {
                    "text_vector": {
                        "vector": query_vector,
                        "k": n_results,
                    }
                }
            }
        }

        response = self.client.search(
            body=search_query,
            index=self.index_name
        )
        return self.convert_opensearch_to_langchain(response)

    def convert_opensearch_to_langchain(self, opensearch_results) -> List[Document]:
        documents = []
        
        for hit in opensearch_results['hits']['hits']:
            source = hit['_source']
            
            doc = Document(
                page_content= source['text'],  
                metadata={
                    'score': hit['_score'],
                    'source': source['source'],
                }
            )
            documents.append(doc)
        
        return documents