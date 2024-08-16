import json
import logging
import pandas as pd
from tqdm.auto import tqdm
import re
import requests
from urllib.parse import urljoin, urlparse

from typing import List

import html2text
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from joblib import Memory
from langchain_community.document_loaders import AsyncHtmlLoader, DataFrameLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from generate_vectorstore import vectorstore
from opensearchpy import helpers
import time
import functools


LOCATION = "./cachedir"
MEMORY = Memory(LOCATION, verbose=0)

load_dotenv()  # take environment variables from .env.

BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{BLUE}CADDY SCRAPER{RESET} | {GREEN}%(asctime)s{RESET} | {
            YELLOW}%(levelname)s{RESET} | {CYAN}%(message)s{RESET}",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def retry(num_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator

    Parameters:
    num_retries (int): Number of times to retry before giving up
    delay (int): Initial delay between retries in seconds
    backoff (int): Factor by which the delay should be multiplied each retry
    exceptions (tuple): Exceptions to trigger a retry
    """

    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            _num_retries, _delay = num_retries, delay
            while _num_retries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    _num_retries -= 1
                    if _num_retries == 0:
                        raise
                    time.sleep(_delay)
                    _delay *= backoff
                    logger.warning(
                        f"Retrying {_num_retries} more times after exception: {e}")

        return wrapper_retry

    return decorator_retry


def remove_anchor_urls(urls):
    """
    Removes anchor URLs (URLs with a # followed by text at the end) from a list of URLs.
    Args:
        urls (list): A list of URLs (strings).
    Returns:
        list: A new list containing only the URLs that are not anchor URLs.
    """
    anchor_pattern = re.compile(r"#.*$")
    cleaned_urls = []

    for url in urls:
        if not anchor_pattern.search(url):
            cleaned_urls.append(url)

    return cleaned_urls


def crawl_url_batch(
    url_list: List,
    domain_description: str,
    div_classes: List = None,
    div_ids: List = None,
    authentication_cookie: str = None,
):
    """Takes a list of URLS, iterartively scrapes the content of each page, and returns a list of langchain Documents"""

    if authentication_cookie:
        cookie_dict = {"Cookie": authentication_cookie}
        loader = AsyncHtmlLoader(url_list, header_template=cookie_dict)
    else:
        loader = AsyncHtmlLoader(url_list)

    docs = loader.load()

    scraped_pages = []

    for page in tqdm(docs):
        current_url = page.metadata["source"]

        # get main section of page
        soup = BeautifulSoup(page.page_content, "html.parser")

        main_section_html = ""

        if div_ids:
            for div_id in div_ids:
                selected_div_id_html = soup.find("div", id=div_id)
                if selected_div_id_html:
                    main_section_html += str(selected_div_id_html)

        if div_classes:
            for div_class in div_classes:
                selected_div_classes_html = soup.find("div", class_=div_class)
                if selected_div_classes_html:
                    main_section_html += str(selected_div_classes_html)

        if not main_section_html:
            main_section_html = str(soup)

        # page content
        current_page_markdown = html2text.html2text(str(main_section_html))
        page_dict = {"source_url": current_url,
                     "markdown": current_page_markdown}
        scraped_pages.append(page_dict)

    document_df = pd.DataFrame(scraped_pages)

    unique_pages = document_df.drop_duplicates(subset=["source_url"]).reset_index(
        drop=True
    )

    unique_pages["domain_description"] = domain_description
    unique_pages["scraped_at"] = pd.to_datetime("today")
    unique_pages["updated_at"] = pd.to_datetime("today")

    dataframe_loader = DataFrameLoader(
        unique_pages, page_content_column="markdown")

    docs_to_upload = dataframe_loader.load()

    return docs_to_upload


def get_sitemap(url):
    """Scrapes an XML sitemap from the provided URL and returns XML source.

    Args:
        url (string): Fully qualified URL pointing to XML sitemap.

    Returns:
        xml (string): XML source of scraped sitemap.
    """

    response = requests.get(url)  # nosec
    response.raise_for_status()  # Ensure we get a valid response or raise an HTTPError
    # Set the apparent encoding if not provided
    response.encoding = response.apparent_encoding
    xml = BeautifulSoup(response.content, "lxml-xml")
    return xml


def get_sitemap_type(xml):
    """Parse XML source and returns the type of sitemap.

    Args:
        xml (string): Source code of XML sitemap.

    Returns:
        sitemap_type (string): Type of sitemap (sitemap, sitemapindex, or None).
    """

    sitemapindex = xml.find_all("sitemapindex")
    sitemap = xml.find_all("urlset")

    if sitemapindex:
        return "sitemapindex"
    elif sitemap:
        return "urlset"
    else:
        return


def return_excluded_domains():
    """Returns the list of excluded domains from the json file as a list

    Returns:
    excluded_domains (list): a list of excluded domains for the domain
    """

    file_path = "excluded_domains.json"

    with open(file_path, "r") as file:
        data = json.load(file)
        excluded_domains = data["excluded_urls"]

    return excluded_domains


def get_child_sitemaps(xml):
    """Return a list of child sitemaps present in a XML sitemap file.

    Args:
        xml (string): XML source of sitemap.

    Returns:
        sitemaps (list): Python list of XML sitemap URLs.
    """

    sitemaps = xml.find_all("sitemap")

    output = []

    for sitemap in sitemaps:
        output.append(sitemap.findNext("loc").text)
    return output


def generate_vectorstore():
    """Creates the vectorstore with an initial set of documents, and returns the retriever"""

    return vectorstore


@retry()
def add_document_list_to_vectorstore(document_list, vectorstore, bulk_size=20000):
    """Takes a list of documents, and adds them to the vectorstore in bulk

    Args:
        document_list (list): list of documents
        vectorstore (vectorstore): vectorstore to add the documents to
        bulk_size (int, optional): the size of the bulk to add the documents in
        retry_count (int, optional): the number of times to retry adding documents in case of failure

    Returns:
        added_docs (int): the number of documents added to the vectorstore
    """

    added_docs = vectorstore.add_documents(document_list, bulk_size=bulk_size)
    return added_docs


def sitemap_to_dataframe(xml, name=None, verbose=False):
    """Read an XML sitemap into a Pandas dataframe.

    Args:
        xml (bs4): XML source of sitemap as a beau
        name (optional): Optional name for sitemap parsed.
        verbose (boolean, optional): Set to True to monitor progress.

    Returns:
        dataframe: Pandas dataframe of XML sitemap content.
    """

    urls = xml.find_all("url")

    # Prepare lists to collect data
    data = []

    for url in urls:
        loc = url.find("loc").text if url.find("loc") else ""
        domain = urlparse(loc).netloc if loc else ""
        changefreq = url.find("changefreq").text if url.find(
            "changefreq") else ""
        priority = url.find("priority").text if url.find("priority") else ""
        sitemap_name = name if name else ""

        row = {
            "domain": domain,
            "loc": loc,
            "changefreq": changefreq,
            "priority": priority,
            "sitemap_name": sitemap_name,
        }

        if verbose:
            logger.debug(row)

        data.append(row)

    # Create DataFrame from collected data
    df = pd.DataFrame(data)

    return df


def get_all_urls(url, domains_to_exclude=None):
    """Return a dataframe containing all of the URLs from a site's XML sitemaps.

    Args:
        url (string): URL of site's XML sitemap. Usually located at /sitemap.xml
        domains_to_exclude (list, optional): List of domains to exclude from the sitemap.

    Returns:
        list_of_dfs (list): a list of pandas dataframes

    """
    try:
        xml = get_sitemap(url)
        sitemap_type = get_sitemap_type(xml)

        if sitemap_type == "sitemapindex":
            sitemaps = get_child_sitemaps(xml)
        else:
            sitemaps = [url]

        list_of_dfs = []

        for sitemap in sitemaps:
            try:
                logger.info(f"Processing sitemap: {sitemap}")
                sitemap_xml = get_sitemap(sitemap)
                df_sitemap = sitemap_to_dataframe(sitemap_xml, name=sitemap)
                logger.info(f"Sitemap processed: {sitemap}")
                df = pd.DataFrame(
                    columns=["loc", "changefreq",
                             "priority", "domain", "sitemap_name"]
                )
                # remove any rows which contain any of the excluded domains
                if domains_to_exclude:
                    df_sitemap = df_sitemap[
                        ~df_sitemap["loc"].str.contains(
                            "|".join(domains_to_exclude))
                    ]

                df = pd.concat([df, df_sitemap], ignore_index=True)
                list_of_dfs.append(df)

            except Exception as e:
                logger.error(f"Error processing sitemap {sitemap}: {e}")

        return list_of_dfs

    except Exception as e:
        logger.error(f"Error initializing sitemap processing for {url}: {e}")
        return []


def find_all_urls_on_sitemap(url):
    df = get_all_urls(url)
    return df


def extract_urls(base_url, text):
    """Extracts URLs from a string of text.

    Args:
        base_url (string): The base URL of the page.
        text (string): The text to extract URLs from.

    Returns:
        urls (list): A list of URLs found in the text.
    """

    # Regular expression to find URLs in parentheses
    pattern = re.compile(r"\((.*?)\)")

    # Extract all URLs
    urls = pattern.findall(text)

    # if any urls start with a /, add the base_url
    urls = [urljoin(base_url, url) if url.startswith("/")
            else url for url in urls]

    # Remove any urls that don't start with http
    urls = [url for url in urls if url.startswith("http")]

    # remove duplicate urls
    urls = list(set(urls))

    return urls


def remove_markdown_index_links(markdown_text: str) -> str:
    """Clean markdown text by removing index links.

    Args:
        markdown_text (str): markdown text to clean.

    Returns:
        str: cleaned markdown string.
    """
    # Regex patterns
    list_item_link_pattern = re.compile(
        r"^\s*\*\s*\[[^\]]+\]\([^\)]+\)\s*$", re.MULTILINE
    )
    list_item_header_link_pattern = re.compile(
        r"^\s*\*\s*#+\s*\[[^\]]+\]\([^\)]+\)\s*$", re.MULTILINE
    )
    header_link_pattern = re.compile(
        r"^\s*#+\s*\[[^\]]+\]\([^\)]+\)\s*$", re.MULTILINE
    )
    # Remove matches
    cleaned_text = re.sub(list_item_header_link_pattern, "", markdown_text)
    cleaned_text = re.sub(list_item_link_pattern, "", cleaned_text)
    cleaned_text = re.sub(header_link_pattern, "", cleaned_text)
    # Removing extra newlines resulting from removals
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
    cleaned_text = re.sub(
        r"^\s*\n", "", cleaned_text, flags=re.MULTILINE
    )  # Remove leading newlines
    return cleaned_text


def check_if_link_in_base_domain(base_url, link):
    """checks if a link is in the same domain as the base url. If it is, returns the link"""

    if link.startswith(base_url):
        return link

    elif not link.startswith("http"):
        return f"{base_url}{link}"

    else:
        return False


def scrape_url_list(base_url, url_list, authentication_cookie=None):
    """takes a list of urls, and returns a dataframe of page content as markdown and urls, as well as any links found as a list

    Args:
        base_url (string): the base url of the domain
        url_list (list): a list of urls to scrape
        authentication_cookie (str, optional): a cookie to use when scraping

    Returns:
        unique_pages (dataframe): a dataframe of unique pages and their content
        links (list): a list of links found on the pages
    """

    bs_transformer = BeautifulSoupTransformer()

    if authentication_cookie:
        cookie_dict = {"Cookie": authentication_cookie}
        loader = AsyncHtmlLoader(url_list, header_template=cookie_dict)
    else:
        loader = AsyncHtmlLoader(url_list)
    docs = loader.load()

    pages = []
    links = []

    for page in tqdm(docs):
        current_url = page.metadata["source"]

        # get main section of page
        soup = BeautifulSoup(page.page_content)

        if url_list == [
            base_url
        ]:  # for base url (homepage), use whole page to get all links
            main_section_html = soup
        else:
            if soup.find("div", id=["main-content", "cads-main-content"]):
                main_section_html = soup.find(
                    "div", id=["main-content", "cads-main-content"]
                )
            elif soup.find("div", class_=["main-content", "cads-main-content"]):
                main_section_html = soup.find(
                    "div", class_=["main-content", "cads-main-content"]
                )
            else:
                main_section_html = soup

        # get links on main section of page
        extracted_links = bs_transformer.extract_tags(
            str(main_section_html), ["a"])

        # run extract_url on each url
        current_page_links = extract_urls(current_url, extracted_links)

        # add links if in base domain
        current_page_links = [
            link
            for link in current_page_links
            if check_if_link_in_base_domain(base_url, link)
        ]

        # add current page links to the link list
        links += current_page_links

        # remove duplicate links
        links = list(set(links))

        # page content
        current_page_markdown = html2text.html2text(str(main_section_html))
        page_dict = {"source_url": current_url,
                     "markdown": current_page_markdown}
        pages.append(page_dict)

    # Create a dataframe with page sources & contents
    document_df = pd.DataFrame(pages)

    unique_pages = document_df.drop_duplicates(subset=["source_url"]).reset_index(
        drop=True
    )

    logger.info(f"Number of pages scraped: {len(pages)}")

    return unique_pages, links


def delete_duplicate_urls_from_store(vectorstore):
    """Looks for duplicate source urls in the Opensearch vectorstore, and removes them, keeping only the most recent based on metadata.time_scraped"""

    index_name = vectorstore.index_name

    # Step 1: Aggregate to find potential duplicates
    agg_query = {
        "size": 0,
        "aggs": {
            "duplicate_urls": {
                "terms": {
                    "field": "metadata.source_url.keyword",
                    "min_doc_count": 2,
                    "size": 10000,  # Adjust size based on expected number of unique URLs
                }
            }
        },
    }

    agg_result = vectorstore.client.search(index=index_name, body=agg_query)

    # Step 2: For each duplicate, find the most recent document and prepare to delete the rest
    delete_candidates = []

    for bucket in agg_result["aggregations"]["duplicate_urls"]["buckets"]:
        url = bucket["key"]

        # Find documents with this URL, sorted by time_scraped in descending order
        search_query = {
            "size": bucket["doc_count"],
            "query": {"term": {"metadata.source_url.keyword": url}},
            "sort": [
                {"metadata.scraped_at": {"order": "desc"}}  # Sort by time_scraped
            ],
            # Retrieve ID and time_scraped
            "_source": ["_id", "metadata.scraped_at"],
        }

        search_result = vectorstore.client.search(
            index=index_name, body=search_query)
        doc_ids = [hit["_id"] for hit in search_result["hits"]["hits"]]

        # Keep the first ID (most recent) and mark the rest for deletion
        delete_candidates.extend(doc_ids[1:])

    # Step 3: Delete duplicates
    for doc_id in delete_candidates:
        vectorstore.client.delete(index=index_name, id=doc_id)

    logger.info(f"Deleted {len(delete_candidates)} duplicate documents")

    def fetch_entries_to_dataframe(opensearch_client, index_name):
        """
        Fetch all entries from the specified OpenSearch index, extract ID and URL,
        and return a pandas DataFrame containing these details using pandas.concat for efficiency.
        Includes a progress bar using tqdm.

        Parameters:
        - opensearch_client: An instance of OpenSearch client.
        - index_name: The name of the OpenSearch index from which to fetch entries.

        Returns:
        - A pandas DataFrame with columns 'ID' and 'URL'.
        """

        records = []  # Initialize a list to hold all the records

        # Prepare the initial search
        page = opensearch_client.search(
            index=index_name,
            scroll="2m",  # Keep the search context open for 2 minutes
            size=1000,  # Adjust size per page as needed
            body={"query": {"match_all": {}}},
        )

        scroll_id = page["_scroll_id"]
        total_hits = (
            page["hits"]["total"]["value"]
            if "value" in page["hits"]["total"]
            else page["hits"]["total"]
        )  # Compatibility with different versions of OpenSearch/Elasticsearch

        with tqdm(total=total_hits, desc="Fetching records") as pbar:
            while len(page["hits"]["hits"]):
                for hit in page["hits"]["hits"]:
                    doc_id = hit["_id"]
                    url = (
                        hit["_source"]["metadata"]["source_url"]
                        if "metadata" in hit["_source"]
                        and "source_url" in hit["_source"]["metadata"]
                        else "URL Not Available"
                    )
                    records.append({"ID": doc_id, "URL": url})

                pbar.update(len(page["hits"]["hits"]))

                # Fetch the next page of results
                page = opensearch_client.scroll(
                    scroll_id=scroll_id, scroll="2m")

                if not page["hits"]["hits"]:
                    break

        # Correctly clear the scroll
        opensearch_client.clear_scroll(body={"scroll_id": [scroll_id]})

        # Convert the list of records into a DataFrame
        df = pd.DataFrame(records, columns=["ID", "URL"])

        return df

    def bulk_delete_by_ids(opensearch_client, index_name, document_ids):
        """
        Delete documents from the specified OpenSearch index based on a list of document IDs.

        Parameters:
        - opensearch_client: An instance of OpenSearch client.
        - index_name: The name of the OpenSearch index from which to delete documents.
        - document_ids: A list of document IDs to be deleted.
        """
        # Create a generator of delete actions
        actions = (
            {"_op_type": "delete", "_index": index_name, "_id": doc_id}
            for doc_id in document_ids
        )

        # Perform the bulk delete operation
        response = helpers.bulk(opensearch_client, actions)

        return response

    def delete_excluded_domains(vectorstore):
        """Given a domain from the scrape, search for any excluded domains and delete them all.

        Args:
            vectorstore: The vectorstore object.
            index_name (str): The name of the index.

        Returns:
            deleted_pages (list): A list of dictionaries containing the ID and details of each deleted page.
        """

        index_name = vectorstore.index_name

        try:
            file_path = "scrape/excluded_domains.json"
            with open(file_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            file_path = "excluded_domains.json"
            with open(file_path, "r") as file:
                data = json.load(file)

        combined_list = []
        for key in data:
            combined_list.extend(data[key])

        entry_df = fetch_entries_to_dataframe(vectorstore.client, index_name)

        excluded_entries = entry_df[
            entry_df["URL"].str.contains("|".join(combined_list))
        ]

        ids_to_delete = excluded_entries["ID"].tolist()

        deleted_docs = bulk_delete_by_ids(
            vectorstore.client, index_name, ids_to_delete)

        num_deleted_domains = len(deleted_docs)
        logger.info(f"Total number of deleted domains: {num_deleted_domains}")

        return deleted_docs


def fetch_entries_to_dataframe(opensearch_client, index_name):
    """
    Fetch all entries from the specified OpenSearch index, extract ID and URL,
    and return a pandas DataFrame containing these details using pandas.concat for efficiency.
    Includes a progress bar using tqdm.

    Parameters:
    - opensearch_client: An instance of OpenSearch client.
    - index_name: The name of the OpenSearch index from which to fetch entries.

    Returns:
    - A pandas DataFrame with columns 'ID' and 'URL'.
    """
    records = []  # Initialize a list to hold all the records

    # Prepare the initial search
    page = opensearch_client.search(
        index=index_name,
        scroll="2m",  # Keep the search context open for 2 minutes
        size=1000,  # Adjust size per page as needed
        body={"query": {"match_all": {}}},
    )

    scroll_id = page["_scroll_id"]
    total_hits = (
        page["hits"]["total"]["value"]
        if "value" in page["hits"]["total"]
        else page["hits"]["total"]
    )  # Compatibility with different versions of OpenSearch/Elasticsearch

    with tqdm(total=total_hits, desc="Fetching records") as pbar:
        while len(page["hits"]["hits"]):
            for hit in page["hits"]["hits"]:
                doc_id = hit["_id"]
                url = (
                    hit["_source"]["metadata"]["source_url"]
                    if "metadata" in hit["_source"]
                    and "source_url" in hit["_source"]["metadata"]
                    else "URL Not Available"
                )
                records.append({"ID": doc_id, "URL": url})

            pbar.update(len(page["hits"]["hits"]))

            # Fetch the next page of results
            page = opensearch_client.scroll(scroll_id=scroll_id, scroll="2m")

            if not page["hits"]["hits"]:
                break

    # Correctly clear the scroll
    opensearch_client.clear_scroll(body={"scroll_id": [scroll_id]})

    # Convert the list of records into a DataFrame
    df = pd.DataFrame(records, columns=["ID", "URL"])

    return df
