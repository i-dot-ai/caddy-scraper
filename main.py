import json
from scraper_registry import scraper_registry
import os
from core_utils import logger


def load_scrape_configs(file_path: str):
    with open(file_path, "r") as file:
        logger.debug(f"{os.getcwd()}")
        return json.load(file)


def execute_scraper(config):
    scrape_method = config["scrape_method"]
    scraper_func = scraper_registry.get(scrape_method)

    if scraper_func:
        scrape_args = config.get("scrape_args", {})
        scraper_func(
            base_domain=config["base_domain"],
            domain_description=config["domain_description"],
            **scrape_args,
        )
    else:
        logger.error(f"No scraper function found for method: {scrape_method}")


def main():
    file_path = "scraper_config.json"
    configs = load_scrape_configs(file_path)

    for config in configs:
        execute_scraper(config)


if __name__ == "__main__":
    main()
