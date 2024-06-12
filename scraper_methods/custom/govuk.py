from datetime import datetime
from core_utils import get_all_urls

import pandas as pd
from tqdm import tqdm
from langchain.schema import Document
import requests
from langchain_community.document_transformers import Html2TextTransformer
import logging
from core_utils import (
    return_excluded_domains,
    generate_vectorstore,
    add_document_list_to_vectorstore,
)


# Configure logging
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def get_url_content_from_content_api(url, minimum_body_length=10):
    url_start = "https://www.gov.uk/"
    api_call_start = "https://www.gov.uk/api/content/"

    base_govuk = "https://www.gov.uk"

    html2text = Html2TextTransformer()

    if url.startswith(url_start):
        # replace the url_start with api_call_start
        api_call = url.replace(url_start, api_call_start)
        api_call
    else:
        # raise exception
        raise Exception("not gov.uk url")

    response = requests.get(api_call, timeout=100)
    # check the status code
    response_code = response.status_code

    # if does not return 200, then raise exception
    if response_code != 200:
        logging.debug("API call failed")
        # log the response and code
        logging.debug(response_code)
        logging.debug(response)
        return None

    json = response.json()

    # Assuming 'json' is your JSON object
    details = json.get("details", {})

    # Check if 'body' exists, otherwise concatenate 'parts'
    json_body = details.get("body")
    if json_body is None:
        # Using join() for efficient string concatenation
        parts = details.get("parts", [])
        json_body = "".join(part.get("body", "") for part in parts)

    if len(json_body) < minimum_body_length:
        return None

    langchain_doc = Document(
        page_content=json_body,
        metadata={
            "source_url": base_govuk + json["base_path"],
            "updated_at": json["updated_at"],
            "scraped_at": datetime.now(),
            "title": json["title"],
            "domain_description": "GOV.UK",
        },
    )

    docs = [langchain_doc]

    docs_to_markdown = html2text.transform_documents(docs)

    return docs_to_markdown


def expand_lists_in_docs(document_list):
    """takes a list of documents, and expands any lists within the list, and removes any None values

    Args:
        document_list (list): list of documents

        Returns:
        new_list (list): list of documents with lists expanded and None values removed

    """

    new_list = []
    for item in document_list:
        if isinstance(item, list):
            new_list = new_list + item
        # if item is None, remove it
        elif item is None:
            pass
        else:
            new_list.append(item)
    return new_list


def scrape_govuk_child_sitemap_df(
    site_df, vectorstore, batch_size=1000, retry_attempts=3, token_chunk_size=512
):
    """takes a dataframe generated from a govuk child sitemap, and then scrapes each url in turn

    Args:
        site_df (dataframe): dataframe generated from a govuk child sitemap
        vectorstore (vectorstore): vectorstore to add the scraped content to
        regex_pattern_to_exclude (list, optional): list of regex patterns to exclude from the scrape. Defaults to exclude_pattern.

    Returns:
        None

    """

    # log start time and number of rows in the dataframe
    print(f"Starting at {datetime.now()}")
    print(f"Number of rows in dataframe: {len(site_df)}")

    # split dataframe into batches
    batches = [site_df[x : x + batch_size] for x in range(0, len(site_df), batch_size)]

    for batch in tqdm(batches):
        print("querying API")
        for attempt in range(1, retry_attempts + 1):
            try:
                urls_to_documents = (
                    batch["loc"].apply(get_url_content_from_content_api).tolist()
                )
                break  # Exit loop if successful
            except Exception as e:
                logging.debug(f"Attempt {attempt} failed: {e}")
                if attempt == retry_attempts:
                    logging.debug(f"Failed after {retry_attempts} attempts")
                    raise e  # Raising the exception to terminate the function

        document_list = expand_lists_in_docs(urls_to_documents)
        docs_with_content = [
            doc for doc in document_list if doc.page_content is not None
        ]
        logging.debug(f"Number of docs to add: {len(docs_with_content)}")

        for attempt in range(1, retry_attempts + 1):
            try:
                add_document_list_to_vectorstore(
                    docs_with_content, vectorstore, batch_size=batch_size
                )
                break  # Exit loop if successful
            except Exception as e:
                logging.debug(f"Attempt {attempt} failed: {e}")
                if attempt == retry_attempts:
                    logging.debug(f"Failed after {retry_attempts} attempts")
                    raise e  # Raising the exception to terminate the function

        print("Added to DB")


def iterative_govuk_scrape(domains_to_exclude=None, retry_attempts=3, **kwargs):
    """Iterates over a sitemap, and then conducts a scrape of each in turn.

    Args:
        url (string): URL of site's XML sitemap. Usually located at /sitemap.xml
        vectorstore (vectorstore): vectorstore to add the scraped content to
        domains_to_exclude (list of str, optional):  patterns to exclude certain URLs.
        retry_attempts (int, optional): Number of times to retry scraping a sitemap in case of failure.

    Returns:
        None

    """

    batch_size = kwargs.get("batch_size", 500)

    govuk_sitemap = "https://www.gov.uk/sitemap.xml"

    website_url_df_list = get_all_urls(govuk_sitemap)

    domains_to_exclude = return_excluded_domains()

    vectorstore = generate_vectorstore()

    logging.debug(f"Number of sitemaps found: {len(website_url_df_list)}")
    print(f"Number of sitemaps found: {len(website_url_df_list)}")

    for sitemap in tqdm(website_url_df_list):
        df = pd.DataFrame(
            columns=["loc", "changefreq", "priority", "domain", "sitemap_name"]
        )

        df = pd.concat([df, sitemap], ignore_index=True)

        if domains_to_exclude:
            # remove any rows where the loc contains any of the excluded domains
            df = df[~df["loc"].str.contains("|".join(domains_to_exclude))]

        logging.debug(f"Number of rows in dataframe: {len(df)}")
        scrape_govuk_child_sitemap_df(
            df, vectorstore, retry_attempts=retry_attempts, batch_size=batch_size
        )


"""    for sitemap in tqdm(website_url_df_list):
        try:
            df = pd.DataFrame(
                columns=["loc", "changefreq", "priority", "domain", "sitemap_name"]
            )

            df = pd.concat([df, sitemap], ignore_index=True)

            if domains_to_exclude:
                # remove any rows where the loc contains any of the excluded domains
                df = df[~df["loc"].str.contains("|".join(domains_to_exclude))]

            logging.debug(f"Number of rows in dataframe: {len(df)}")
            scrape_govuk_child_sitemap_df(
                df, vectorstore, retry_attempts=retry_attempts, batch_size=batch_size
            )

        except Exception as e:
            print(f"Error processing sitemap {sitemap}: {e}")
            # print error
            logging.debug(e)"""
