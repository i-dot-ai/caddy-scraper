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
from langchain_community.document_loaders import AsyncHtmlLoader, DataFrameLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

from generate_vectorstore import vectorstore, embedding_model
from chunking import (
    split_short_and_long_documents,
    chunk_document_embedding_and_add_to_db,
)
from opensearchpy import helpers
import time
import functools
import traceback


load_dotenv()  # take environment variables from .env.

# Configure logging
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


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
                    print(f"Retrying {_num_retries} more times after exception: {e}")

        return wrapper_retry

    return decorator_retry


def remove_duplicate_chunks(markdown_text):
    # Split the text into chunks based on line breaks and heading markers
    chunks = re.split(r"\n\s*\n+|\n(?=#)", markdown_text)

    # Create a set to store unique chunks
    unique_chunks = set()

    # Create a list to store the result
    result = []

    # Iterate through the chunks
    for chunk in chunks:
        # If the chunk is not empty and not a duplicate, add it to the result
        if chunk.strip() and chunk.strip() not in unique_chunks:
            unique_chunks.add(chunk.strip())
            result.append(chunk)

    # Join the remaining chunks back into a single string
    return "\n\n".join(result)


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

        if soup.find("div", id=div_ids):
            main_section_html = soup.find("div", id=div_ids)
        elif soup.find("div", class_=div_classes):
            main_section_html = soup.find("div", class_=div_classes)
        else:
            main_section_html = soup

        # Parse the combined HTML content again to extract paragraphs
        if len(main_section_html) > 0:
            current_page_markdown = html2text.html2text(str(main_section_html))
            page_dict = {"source_url": current_url, "markdown": current_page_markdown}
            scraped_pages.append(page_dict)

    document_df = pd.DataFrame(scraped_pages)

    unique_pages = document_df.drop_duplicates(subset=["source_url"]).reset_index(
        drop=True
    )

    unique_pages["domain_description"] = domain_description
    unique_pages["scraped_at"] = pd.to_datetime("today")
    unique_pages["updated_at"] = pd.to_datetime("today")

    # TODO REMOVE this once debugging done
    unique_pages.to_csv("unique_pages.csv")
    unique_pages.to_pickle("unique_pages.pkl")

    # unique_pages['token_count'] = unique_pages['markdown'].apply(lambda x: num_tokens_from_string(x, default_encoder))

    dataframe_loader = DataFrameLoader(unique_pages, page_content_column="markdown")

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
    response.encoding = (
        response.apparent_encoding
    )  # Set the apparent encoding if not provided
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
def add_document_list_to_vectorstore(
    document_list,
    vectorstore,
    batch_size=50,
    retry_count=3,
    maximum_token_length=512,
    maximum_character_length=2000,
    request_timeout=90,
):
    """Takes a list of documents, and adds them to the vectorstore in batches

    Args:
        document_list (list): list of documents
        vectorstore (vectorstore): vectorstore to add the documents to
        batch_size (int, optional): the size of each batch to add the documents in
        retry_count (int, optional): the number of times to retry adding documents in case of failure

    Returns:
        added_docs (int): the number of documents added to the vectorstore
    """

    print("beginning to shorten long documents")
    print("initial document count:", len(document_list))

    list_of_short_docs, list_of_too_long_docs = split_short_and_long_documents(
        document_list, maximum_token_length, max_chars=maximum_character_length
    )

    print("too long document count:", len(list_of_too_long_docs))
    print("short document count:", len(list_of_short_docs))

    print("beginning to add short documents to vectorstore")

    docs_to_add = list_of_short_docs

    added_docs = 0
    num_docs = len(docs_to_add)
    num_batches = (
        num_docs + batch_size - 1
    ) // batch_size  # Changed to ensure correct batch count

    print(f"Adding {num_docs} documents to the vectorstore in {num_batches} batches")

    for i in range(num_batches):
        print(f"Adding batch {i+1} of {num_batches}")
        start = i * batch_size
        end = min((i + 1) * batch_size, num_docs)
        batch = docs_to_add[start:end]

        retry = 0
        while retry < retry_count:
            try:
                vectorstore.add_documents(
                    batch, bulk_size=2000, timeout=request_timeout
                )
                added_docs += len(batch)
                print("added batch", i + 1)
                break
            except Exception:
                retry += 1
                print(f"Failed to add batch {i+1}. Retrying... ({retry}/{retry_count})")

                print(batch)
                # print the full text of the exception
                print(traceback.format_exc())
                time.sleep(1)

    print(f"Added {added_docs} documents to the vectorstore")

    print("beggining to add long documents to vectorstore")
    chunk_document_embedding_and_add_to_db(
        list_of_too_long_docs,
        vectorstore,
        max_chars=maximum_character_length,
        embedding_model=embedding_model,
    )
    print("added long documents to vectorstore")

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
        changefreq = url.find("changefreq").text if url.find("changefreq") else ""
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
            print(row)

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
                print("Processing sitemap: ", sitemap)
                sitemap_xml = get_sitemap(sitemap)
                df_sitemap = sitemap_to_dataframe(sitemap_xml, name=sitemap)
                print("Sitemap processed: ", sitemap)
                df = pd.DataFrame(
                    columns=["loc", "changefreq", "priority", "domain", "sitemap_name"]
                )
                # remove any rows which contain any of the excluded domains
                if domains_to_exclude:
                    df_sitemap = df_sitemap[
                        ~df_sitemap["loc"].str.contains("|".join(domains_to_exclude))
                    ]

                df = pd.concat([df, df_sitemap], ignore_index=True)
                list_of_dfs.append(df)

            except Exception as e:
                print(f"Error processing sitemap {sitemap}: {e}")

        return list_of_dfs

    except Exception as e:
        print(f"Error initializing sitemap processing for {url}: {e}")
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
    urls = [urljoin(base_url, url) if url.startswith("/") else url for url in urls]

    # Remove any urls that don't start with http
    urls = [url for url in urls if url.startswith("http")]

    # remove duplicate urls
    urls = list(set(urls))

    return urls


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
        extracted_links = bs_transformer.extract_tags(str(main_section_html), ["a"])

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
        page_dict = {"source_url": current_url, "markdown": current_page_markdown}
        pages.append(page_dict)

    # Create a dataframe with page sources & contents
    document_df = pd.DataFrame(pages)

    unique_pages = document_df.drop_duplicates(subset=["source_url"]).reset_index(
        drop=True
    )

    print(f"Number of pages scraped: {len(pages)}")

    return unique_pages, links


def delete_duplicate_chunks_from_store(vectorstore):
    """Looks for duplicate source urls and text chunks in the Opensearch vectorstore, and removes them, keeping only the most recent based on metadata.time_scraped"""

    index_name = vectorstore.index_name

    # Step 1: Aggregate to find potential duplicates
    agg_query = {
        "size": 0,
        "aggs": {
            "duplicate_chunks": {
                "composite": {
                    "sources": [
                        {
                            "source_url": {
                                "terms": {"field": "metadata.source_url.keyword"}
                            }
                        },
                        {"text": {"terms": {"field": "text.keyword"}}},
                    ],
                    "size": 10000,  # Adjust size based on expected number of unique chunks
                }
            }
        },
    }

    agg_result = vectorstore.client.search(index=index_name, body=agg_query)

    # Step 2: For each duplicate, find the most recent document and prepare to delete the rest
    delete_candidates = []

    for bucket in agg_result["aggregations"]["duplicate_chunks"]["buckets"]:
        source_url = bucket["key"]["source_url"]
        text = bucket["key"]["text"]

        # Find documents with this URL and text, sorted by time_scraped in descending order
        search_query = {
            "size": bucket["doc_count"],
            "query": {
                "bool": {
                    "must": [
                        {"term": {"metadata.source_url.keyword": source_url}},
                        {"term": {"text.keyword": text}},
                    ]
                }
            },
            "sort": [
                {"metadata.scraped_at": {"order": "desc"}}  # Sort by time_scraped
            ],
            "_source": ["_id", "metadata.scraped_at"],  # Retrieve ID and time_scraped
        }

        search_result = vectorstore.client.search(index=index_name, body=search_query)
        doc_ids = [hit["_id"] for hit in search_result["hits"]["hits"]]

        # Keep the first ID (most recent) and mark the rest for deletion
        delete_candidates.extend(doc_ids[1:])

    # Step 3: Delete duplicates
    for doc_id in delete_candidates:
        vectorstore.client.delete(index=index_name, id=doc_id)

    print(f"Deleted {len(delete_candidates)} duplicate documents")

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

        deleted_docs = bulk_delete_by_ids(vectorstore.client, index_name, ids_to_delete)

        num_deleted_domains = len(deleted_docs)
        print(f"Total number of deleted domains: {num_deleted_domains}")

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
