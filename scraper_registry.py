from scraper_methods.brute.scraper import brute_scrape_website
from scraper_methods.sitemap.scraper import crawl_with_sitemap
from scraper_methods.custom.govuk import iterative_govuk_scrape


scraper_registry = {
    "sitemap": crawl_with_sitemap,
    "brute": brute_scrape_website,
    "govuk": iterative_govuk_scrape,
}
