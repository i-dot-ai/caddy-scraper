from core_utils import (
    add_document_list_to_vectorstore,
    crawl_url_batch,
    generate_vectorstore,
    delete_duplicate_urls_from_store,
    logger
)
from scraper_methods.brute.utils import brute_find_all_links


def brute_scrape_website(base_domain: str, domain_description: str, **kwargs):
    """
    takes a website url, and returns a dataframe with all the site content as markdown
    """

    logger.info(f"Brute force scraping for {
                domain_description} at {base_domain}")
    logger.info(f"Additional arguments: {kwargs}")

    scrape_depth = kwargs.get("scrape_depth", None)
    authentication_cookie = kwargs.get("authentication_cookie", None)
    content_div_ids = kwargs.get("content_div_ids", None)
    content_div_classes = kwargs.get("content_div_classes", None)
    batch_size = kwargs.get("batch_size", None)

    vectorstore = generate_vectorstore()

    all_urls_on_domain = brute_find_all_links(
        base_domain,
        scrape_depth=scrape_depth,
        authentication_cookie=authentication_cookie,
    )

    url_batches = [
        all_urls_on_domain[i: i + batch_size]
        for i in range(0, len(all_urls_on_domain), batch_size)
    ]

    for batch in url_batches:
        docs_to_upload = crawl_url_batch(
            batch,
            domain_description,
            content_div_classes,
            content_div_ids,
            authentication_cookie,
        )
        add_document_list_to_vectorstore(docs_to_upload, vectorstore)
    delete_duplicate_urls_from_store(vectorstore)
