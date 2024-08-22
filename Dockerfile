FROM python:3.12.5-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY caddy_scraper ./caddy_scraper
COPY core_utils.py ./caddy_scraper/core_utils.py
COPY excluded_domains.json ./caddy_scraper/excluded_domains.json

WORKDIR caddy_scraper

RUN mkdir -p citizensadvice_scrape advisernet_scrape govuk_scrape

ENTRYPOINT ["python", "run_scrape.py"]
