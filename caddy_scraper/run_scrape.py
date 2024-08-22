"""Script to pull and upload caddy data."""
import asyncio
import datetime
import json
import os

import boto3
from requests_aws4auth import AWS4Auth

from caddy_scraper import CaddyScraper
from vectorstore_manager import VectorStoreManager
from core_utils import logger

with open("scrape_config.json", "r+") as f:
    scrap_configs = json.load(f)
session = boto3.Session()
credentials = session.get_credentials()
auth_creds = AWS4Auth(
    region="eu-west-3",
    service="aoss",
    refreshable_credentials=credentials,
)

if __name__ == "__main__":
    logger.info(f"Starting Caddy Scrape")
    for config in scrap_configs:
        logger.info(f"Scraping {config['base_url']}")
        scraper = CaddyScraper(**config)
        scraper.run()
        logger.info(f"Scrape completed for {
                    config['base_url']}. Output directory: {scraper.output_dir}")
        if not os.path.exists(scraper.output_dir):
            logger.warning(f"Output directory {
                           scraper.output_dir} does not exist after scraping. Creating...")
            os.makedirs(scraper.output_dir)

    for scrape_dir in ['citizensadvice_scrape', 'advisernet_scrape', "govuk_scrape"]:
        logger.info(f"Uploading {scrape_dir}")
        if not os.path.exists(scrape_dir):
            logger.error(
                f"Directory {scrape_dir} does not exist. Skipping upload.")
            continue
        manager = VectorStoreManager(
            index_name=f"{scrape_dir}_db",
            scrape_output_path=scrape_dir,
            opensearch_url=os.getenv("OPENSEARCH_URL"),
            authentication_creds=auth_creds,
            delete_existing_index=False
        )
        asyncio.run(manager.run())
    logger.info(f"Finished Caddy Scrape")
