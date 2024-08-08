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
from tqdm import tqdm

from core_utils import extract_urls, check_if_link_in_base_domain, get_all_urls, remove_anchor_urls, retry


class CaddyScraper:
    """CaddyScraper definition."""

    def __init__(
        self,
        base_url: str,
        sitemap_url: Optional[str] = None,
        crawling_method: str = "brute",
        scrape_depth: Optional[int] = 4,
        div_classes: Optional[List] = None,
        div_ids: Optional[List] = None,
        batch_size: int = 1000,
        output_dir: str = "scrape_result",
        authenticate: bool = False,
    ):
        """Initialise CaddyScraper.

        Args:
            base_url (str): base URL of the website to be scraped
            sitemap_url (Optional[str]): sitemap of domain, if available. Defaults to None.
            crawling_method (str): Crawling method for fetching URLS to scrape options: ["brute", "sitemap"]. Defaults to "brute".
            scrape_depth (Optional[int]): depth of recursive crawler used by "brute" crawling method. Defaults to 4.
            div_classes: (Optional[List]): HTML div classes to scrape. Defaults to None,
            div_ids: (Optional[List]) = HTML div ids to scrape. Defaults to None,
            batch_size (int): Number of URLS to be scraped in a batch. Defaults to 1000.
            output_dir (str): output directory to store scraper results. Defaults to "scrape_result".
            authenticate (bool): whether or not to produce an authentication cookie for domain access. Defaults to False.
        """
        self.base_url = base_url
        self.sitemap_url = sitemap_url
        self.crawling_method = crawling_method
        self.scrape_depth = scrape_depth
        self.div_classes = div_classes
        self.div_ids = div_ids
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.excluded_domains = self.get_excluded_domains()
        self.authentication_cookie = self.get_authentication_cookie(authenticate)

    def run(self):
        """Run the caddy scrapper, fetching relevant urls and saving their scraped content."""
        urls = self.fetch_urls()
        print(f"{len(urls)} urls fetched")
        self.scrape_urls(urls)

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
        if self.authentication_cookie:
            print(self.authentication_cookie)
            cookie_dict = {"Cookie": self.authentication_cookie}
            loader = AsyncHtmlLoader(self.base_url, header_template=cookie_dict)
        else:
            loader = AsyncHtmlLoader(self.base_url)
        docs = loader.load()
        root_page = docs[0]
        page_html = BeautifulSoup(root_page.page_content, features="lxml")
        extracted_links = bs_transformer.extract_tags(str(page_html), ["a"])
        links_list = extract_urls(self.base_url, extracted_links)
        if self.authentication_cookie:
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
                if self.authentication_cookie:
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

    def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
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
            scraped_pages = self.scrape_url_batch(url_batch)
            self.save_scrape_results(scraped_pages, ind)

    @retry()
    def scrape_url_batch(self, url_list: List[str]) -> List[Dict[str, str]]:
        """Takes a batch of urls, iteratively scrapes the content of each page.

        Args:
            url_list (List[str]): list of urls in batch.

        Returns:
            List[Dict[str, str]]: List of dicts containing scraped data from urls.
        """
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
                current_page_markdown = self.remove_markdown_index_links(
                    current_page_markdown
                )
                page_dict = {
                    "source_url": current_url,
                    "markdown": current_page_markdown,
                    "markdown_length": len(current_page_markdown),
                }
                scraped_pages.append(page_dict)
        return scraped_pages

    def remove_markdown_index_links(self, markdown_text: str) -> str:
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

    def save_scrape_results(
        self, scraped_pages: List[Dict[str, str]], file_index: Optional[int] = None
    ):
        """Save results of a scrape to file.

        Args:
            scraped_pages (List[Dict[str, str]]): scraped web page data.
            file_index (Optional[int]): batch index, used to create file name.
        """
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        with open(f"{self.output_dir}/scrape_result_{file_index}.json", "w+") as f:
            json.dump(scraped_pages, f)
