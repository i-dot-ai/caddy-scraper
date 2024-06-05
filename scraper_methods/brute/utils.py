from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from tqdm import tqdm
from core_utils import (
    extract_urls,
    check_if_link_in_base_domain,
    return_excluded_domains,
    remove_anchor_urls,
)


def brute_find_all_links(base_url, scrape_depth, authentication_cookie=None):
    """given a base url, find all links on the domain

    Args:
        base_url (string): the base url of the domain to scrape
        scrape_depth (int): the maximum depth to scrape to
        authentication_cookie (string, optional): authentication cookie for accessing restricted pages

    Returns:
        links (list): a list of links found on the pages
    """

    bs_transformer = BeautifulSoupTransformer()

    excluded_domains = return_excluded_domains()

    if authentication_cookie:
        cookie_dict = {"Cookie": authentication_cookie}
        loader = AsyncHtmlLoader(base_url, header_template=cookie_dict)
    else:
        loader = AsyncHtmlLoader(base_url)

    docs = loader.load()

    root_page = docs[0]

    page_html = BeautifulSoup(root_page.page_content, features="lxml")

    extracted_links = bs_transformer.extract_tags(str(page_html), ["a"])

    links_list = extract_urls(base_url, extracted_links)

    if authentication_cookie:
        loader = AsyncHtmlLoader(links_list, header_template=cookie_dict)
    else:
        loader = AsyncHtmlLoader(links_list)

    links = [base_url]

    for _ in range(scrape_depth):
        all_pages = loader.load()

        for page in tqdm(all_pages):
            current_url = page.metadata["source"]

            page_html = BeautifulSoup(page.page_content, features="lxml")

            extracted_links = bs_transformer.extract_tags(str(page_html), ["a"])

            current_page_links = extract_urls(current_url, extracted_links)

            current_page_links = [
                link
                for link in current_page_links
                if check_if_link_in_base_domain(base_url, link)
            ]

            links += current_page_links

            # remove any links that are in the excluded domain list
            links = [
                link
                for link in links
                if not any(
                    excluded_domain in link for excluded_domain in excluded_domains
                )
            ]

            # remove anchor links
            links = remove_anchor_urls(links)

            # remove duplicates
            links = list(set(links))

            if authentication_cookie:
                loader = AsyncHtmlLoader(links, header_template=cookie_dict)
            else:
                loader = AsyncHtmlLoader(links)

    # take off any anchor links
    filtered_urls = remove_anchor_urls(links)

    return filtered_urls
