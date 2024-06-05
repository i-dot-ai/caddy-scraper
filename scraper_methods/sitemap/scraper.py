from core_utils import (
    generate_vectorstore,
    crawl_url_batch,
    add_document_list_to_vectorstore,
    delete_duplicate_chunks_from_store,
)
from scraper_methods.sitemap.utils import get_urls_from_sitemap_as_batch


def crawl_with_sitemap(base_domain: str, domain_description: str, **kwargs):
    """
    takes a website url, and returns a dataframe with all the site content as markdown
    """
    authentication_cookie = kwargs.get("authentication_cookie", None)
    content_div_ids = kwargs.get("content_div_ids", None)
    content_div_classes = kwargs.get("content_div_classes", None)
    batch_size = kwargs.get("batch_size", None)
    sitemap_url = kwargs.get("sitemap_url", None)

    vectorstore = generate_vectorstore()

    url_batches = get_urls_from_sitemap_as_batch(base_domain, sitemap_url, batch_size)

    for batch in url_batches:
        docs_to_upload = crawl_url_batch(
            batch,
            domain_description,
            div_classes=content_div_classes,
            div_ids=content_div_ids,
            authentication_cookie=authentication_cookie,
        )
        add_document_list_to_vectorstore(docs_to_upload, vectorstore)

    delete_duplicate_chunks_from_store(vectorstore)
