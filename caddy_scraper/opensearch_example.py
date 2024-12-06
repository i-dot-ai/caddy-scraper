import asyncio

from opensearchpy import OpenSearch

from opensearch_document_manager import OpenSearchDocumentManager
from opensearch_query_engine import OpenSearchQueryEngine
from caddy_scraper import CaddyScraper

client = OpenSearch(
    hosts = [{'host': 'localhost', 'port': 9200}],
    http_auth = ('admin', 'Caddy_14211'),
    use_ssl = False,
    verify_certs = False,
)

async def upload_docs():
    # Upload documents to OpenSearch
    doc_manager = OpenSearchDocumentManager(client, index_name="caddy-hybrid-search-index")
    doc_manager.create_index()
    await doc_manager.async_bulk_upload(file_path="ca_test_scrape")


if __name__ == "__main__":
    # Scrape Documents
    scraper = CaddyScraper(
        base_url="https://www.citizensadvice.org.uk/",
        sitemap_url="https://www.citizensadvice.org.uk/sitemap.xml",
        crawling_method='sitemap',
        output_dir='ca_test_scrape',
        div_ids=["main-content", "cads-main-content"],
        div_classes=["main-content", "cads-main-content"],
        scrape_depth=1
    )
    scraper.run()
    
    # Upload Documents
    asyncio.run(upload_docs())

        
    # Query Documents
    query_engine = OpenSearchQueryEngine(
        client,
        index_name="caddy-hybrid-search-index",
        lexical_weight=0.3,
        vector_weight=0.7
    )
    results = query_engine.submit_hybrid_search(query_text="i am getting evicted for no reason what can I do", keywords="eviction", n_results=5)
    print(results)


   