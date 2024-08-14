"""Script to pull and upload caddy data."""
import asyncio
import datetime
import json
import os

import boto3
from requests_aws4auth import AWS4Auth

from caddy_scraper import CaddyScraper
from vectorstore_manager import VectorStoreManager

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
    print(f"Script start time: {datetime.datetime.now()}")
    for config in scrap_configs:
        print(f"Scraping {config['base_url']} \n start time: {datetime.datetime.now()} \n")
        scraper = CaddyScraper(**config)
        scraper.run()

    for scrape_dir in ['citizensadvice_scrape', 'advisernet_scrape', "govuk_scrape"]:
        print(f"Uploading {scrape_dir} \n start time: {datetime.datetime.now()} \n")
        manager = VectorStoreManager(
            index_name=f"{scrape_dir}_db",
            scrape_output_path=scrape_dir,
            opensearch_url="https://mll38hfhhgihe88ldxgj.eu-west-3.aoss.amazonaws.com",
            authentication_creds=auth_creds,
            delete_existing_index=False
        )
        asyncio.run(manager.run())
    print(f"Script finishing time: {datetime.datetime.now()}")