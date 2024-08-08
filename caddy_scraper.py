""""Caddy scraper module."""

import json
import os
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
import html2text
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
import pandas as pd
import requests
from tqdm import tqdm

from core_utils import (
    extract_urls,
    check_if_link_in_base_domain,
    get_all_urls,
    remove_anchor_urls,
    remove_markdown_index_links,
    retry,
)


class CaddyScraper:
    """CaddyScraper definition."""

    def __init__(
        self,
        base_url: str,
        sitemap_url: Optional[str] = None,
        crawling_method: str = "brute",
        downloading_method: str = "scrape",
        scrape_depth: Optional[int] = 4,
        div_classes: Optional[List] = ["main-content", "cads-main-content"],
        div_ids: Optional[List] = ["main-content", "cads-main-content"],
        batch_size: int = 1000,
        output_dir: str = "scrape_result",
    ):
        """Initialise CaddyScraper.

        Args:
            base_url (str): base URL of the website to be scraped
            sitemap_url (Optional[str]): sitemap of domain, if available. Defaults to None.
            crawling_method (str): Crawling method for fetching URLS to download options: ["brute", "sitemap"]. Defaults to "brute".
            downloading_method (str): Downloading method for fetched URLS: ["scrape", "api"]. Defaults to "scrape".
            scrape_depth (Optional[int]): depth of recursive crawler used by "brute" crawling method. Defaults to 4.
            div_classes: (Optional[List]): HTML div classes to scrape. Defaults to ["main-content", "cads-main-content"],
            div_ids: (Optional[List]) = HTML div ids to scrape. Defaults to ["main-content", "cads-main-content"],
            batch_size (int): Number of URLS to be scraped in a batch. Defaults to 1000.
            output_dir (str): output directory to store scraper results. Defaults to "scrape_result".
        """
        self.base_url = base_url
        self.sitemap_url = sitemap_url
        self.crawling_method = crawling_method
        self.downloading_method = downloading_method
        self.scrape_depth = scrape_depth
        self.div_classes = div_classes
        self.div_ids = div_ids
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.excluded_domains = self.get_excluded_domains()

    def run(self):
        """Run the caddy scrapper, fetching relevant urls then downloading and saving their scraped content."""
        urls = self.fetch_urls()
        print(f"{len(urls)} urls fetched")
        self.download_urls(urls)

    def get_excluded_domains(self) -> List[str]:
        """Returns the list of excluded domains from the json file as a list

        Returns:
            excluded_domains (List[str]): a list of excluded domains for the domain
        """
        file_path = "excluded_domains.json"
        with open(file_path, "r") as file:
            data = json.load(file)
            excluded_domains = data["excluded_urls"]
        return excluded_domains

    def get_authentication_cookie(self, authenticate: bool) -> Optional[str]:
        """Get authentication cookie for domain access.

        Args:
            authenticate (bool): flag indicating whether or not cookie should be generated.

        Returns:
            Optional[str]: authentication cookie.
        """
        if authenticate:
            return os.getenv("ADVISOR_NET_AUTHENTICATION")
        else:
            return None

    def fetch_urls(self) -> List[str]:
        """Fetch list of urls to scrape.

        Raises:
            ValueError: scrape_depth attribute must be set to use brute crawling_method.
            ValueError: sitemap_url attribute must be set to use sitemap crawling_method.

        Returns:
            List[str]: list of fetched urls.
        """
        if self.crawling_method == "brute":
            if not isinstance(self.scrape_depth, int):
                raise ValueError(
                    "Valid int must be passed as scrape depth for brute crawling method."
                )
            return self.recursive_crawler()
        elif self.crawling_method == "sitemap":
            if not isinstance(self.sitemap_url, str):
                raise ValueError(
                    "Valid string must be passed as sitemap_url for sitemap crawling method."
                )
            return self.fetch_urls_from_sitemap()
        else:
            return []

    def recursive_crawler(self) -> List[str]:
        """Starting from a base url, recursively crawl a domain for urls.

        Returns:
            links (List[str]): a list of links found on the pages.
        """
        bs_transformer = BeautifulSoupTransformer()
        if "advisernet" in self.base_url:
            cookie_dict = {"Cookie": os.getenv("ADVISOR_NET_AUTHENTICATION")}
            loader = AsyncHtmlLoader(self.base_url, header_template=cookie_dict)
        else:
            loader = AsyncHtmlLoader(self.base_url)
        docs = loader.load()
        root_page = docs[0]
        page_html = BeautifulSoup(root_page.page_content, features="lxml")
        extracted_links = bs_transformer.extract_tags(str(page_html), ["a"])
        links_list = extract_urls(self.base_url, extracted_links)
        if cookie_dict:
            loader = AsyncHtmlLoader(links_list, header_template=cookie_dict)
        else:
            loader = AsyncHtmlLoader(links_list)
        links = [self.base_url]
        for _ in range(self.scrape_depth):
            all_pages = loader.load()
            for page in tqdm(all_pages):
                current_url = page.metadata["source"]
                page_html = BeautifulSoup(page.page_content, features="lxml")
                extracted_links = bs_transformer.extract_tags(str(page_html), ["a"])
                current_page_links = extract_urls(current_url, extracted_links)
                current_page_links = [
                    link
                    for link in current_page_links
                    if check_if_link_in_base_domain(self.base_url, link)
                ]
                links += current_page_links
                links = self.remove_excluded_domains(links)
                links = remove_anchor_urls(list(set(links)))
                if cookie_dict:
                    loader = AsyncHtmlLoader(links, header_template=cookie_dict)
                else:
                    loader = AsyncHtmlLoader(links)
        return remove_anchor_urls(links)

    def fetch_urls_from_sitemap(self) -> List[str]:
        """Fetch urls from a sitemap.

        Returns:
            List[str]: list of urls extracted from sitemap.
        """
        df_of_urls = pd.concat(get_all_urls(self.sitemap_url))
        df_of_urls = df_of_urls[df_of_urls["loc"].str.contains(self.base_url)]
        return self.remove_excluded_domains(df_of_urls["loc"].tolist())

    def remove_excluded_domains(self, url_list: List[str]) -> List[str]:
        """Remove excluded domains from a list of urls.

        Args:
            url_list (List[str]): list of urls to process.

        Returns:
            List[str]: processed list of urls.
        """
        return [
            url
            for url in url_list
            if not any(
                excluded_domain in url for excluded_domain in self.excluded_domains
            )
        ]

    def download_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """Split urls into batches and scrape their content.

        Args:
            urls (List[str]): urls to scrape.

        Returns:
            List[Dict[str, str]]: List of dicts containing scraped data from urls.
        """
        url_batches = [
            urls[i : i + self.batch_size] for i in range(0, len(urls), self.batch_size)
        ]
        for ind, url_batch in enumerate(url_batches):
            if self.downloading_method == "scrape":
                results = self.scrape_url_batch(url_batch)
            elif self.downloading_method == "api":
                results = self.download_batch_from_govuk_api(url_batch)
            self.save_results(results, ind)

    @retry()
    def scrape_url_batch(self, url_list: List[str]) -> List[Dict[str, str]]:
        """Takes a batch of urls, iteratively scrapes the content of each page.

        Args:
            url_list (List[str]): list of urls in batch.

        Returns:
            List[Dict[str, str]]: List of dicts containing scraped data from urls.
        """
        if "advisernet" in self.base_url:
            cookie_dict = {"Cookie": os.getenv("ADVISOR_NET_AUTHENTICATION")}
            loader = AsyncHtmlLoader(url_list, cookie_dict)
        else:
            loader = AsyncHtmlLoader(url_list)
        docs = loader.load()
        scraped_pages = []
        for page in tqdm(docs):
            current_url = page.metadata["source"]
            soup = BeautifulSoup(page.page_content, "html.parser")
            if soup.find("div", id=self.div_ids):
                main_section_html = soup.find("div", id=self.div_ids)
            elif soup.find("div", class_=self.div_classes):
                main_section_html = soup.find("div", class_=self.div_classes)
            else:
                main_section_html = soup
            if len(main_section_html) > 0:
                current_page_markdown = html2text.html2text(str(main_section_html))
                current_page_markdown = remove_markdown_index_links(
                    current_page_markdown
                )
                page_dict = {
                    "source_url": current_url,
                    "markdown": current_page_markdown,
                    "markdown_length": len(current_page_markdown),
                }
                scraped_pages.append(page_dict)
        return scraped_pages

    @retry()
    def download_batch_from_govuk_api(
        self, url_list: List[str], min_body_length: int = 50
    ) -> List[Dict[str, str]]:
        """Use the gov.uk api to download url content.

        Args:
            url_list (List[str]): list of urls to download.
            min_body_length (int): min body length for storage. Defaults to 10.

        Raises:
            ValueError: will be raised if gov.uk url is not the base_url.
            ValueError: will be raised if non gov.uk url is passed to the api.
            requests.exceptions.HTTPError: will be raised if api call fails.

        Returns:
            List[Dict[str, str]]: list of dictionaries containing page content and metadata.
        """
        if "gov.uk" not in self.base_url:
            raise ValueError("Can only use gov.uk API to scrape government urls.")
        pages = []
        for url in url_list:
            if url.startswith(self.base_url):
                api_call = url.replace(self.base_url, "https://www.gov.uk/api/content/")
            else:
                raise ValueError("Non gov.uk url passed to api.")
            response = requests.get(api_call, timeout=100)
            if response.status_code != 200:
                raise requests.exceptions.HTTPError
            response_json = response.json()
            details = response_json.get("details", {})
            json_body = details.get("body")
            if json_body is None:
                parts = details.get("parts", [])
                json_body = "".join(part.get("body", "") for part in parts)
            if len(json_body) > min_body_length:
                linked_urls = re.findall(r"(?P<url>https?://[^\s]+)", json_body)
                if len(linked_urls) > 0:
                    linked_urls = [u[:-1] for u in set(linked_urls)]
                pages.append(
                    {
                        "source_url": url,
                        "markdown": json_body,
                        "markdown_length": len(json_body),
                        "linked_urls": linked_urls,
                    }
                )
        return pages

    def save_results(
        self, pages: List[Dict[str, str]], file_index: Optional[int] = None
    ):
        """Save downloaded pages to file.

        Args:
            pages (List[Dict[str, str]]): downloaded web pages including meta-data.
            file_index (Optional[int]): batch index, used to create file name.
        """
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        with open(f"{self.output_dir}/scrape_result_{file_index}.json", "w+") as f:
            json.dump(pages, f)
