from core_utils import get_all_urls, return_excluded_domains, retry
import pandas as pd


@retry()
def get_urls_from_sitemap_as_batch(
    base_domain: str, sitemap_url: str, batch_size: int = 1000
):
    """
    takes a sitemap url and returns a list of urls
    """

    df_of_urls = pd.concat(get_all_urls(sitemap_url))

    excluded_domain_list = return_excluded_domains()

    # remove any urls that are not on the base domain
    df_of_urls = df_of_urls[df_of_urls["loc"].str.contains(base_domain)]

    url_list = df_of_urls["loc"].tolist()

    # remove any urls that are in the excluded domain list
    url_list_without_excluded_domains = [
        url
        for url in url_list
        if not any(excluded_domain in url for excluded_domain in excluded_domain_list)
    ]

    # separate urls into batches
    url_batches = [
        url_list_without_excluded_domains[i : i + batch_size]
        for i in range(0, len(url_list_without_excluded_domains), batch_size)
    ]

    return url_batches
