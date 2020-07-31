# scrapy-getting-started

This is a scraping tool implemented with the Scrapy web-crawling framework.

To set up this scrapy project, first run the following in terminal: 

`scrapy startproject platforms`

This creates the `platforms/platforms/spiders` directory. Make a copy of foodpanda_spider.py in this directory:

`cp foodpanda_spider.py platforms/platforms/spiders`

To run the spider and sent the output of the yield expression to `foodpanda.json`, run the following in terminal at the top directory of the platforms project where `scrapy.cfg` is located:

`cd platforms`

`scrapy crawl foodpanda -o foodpanda.json`

(remember: Scrapy appends to a given file instead of overwriting its contents. If you run this command twice without removing the file before the second time, you’ll end up with a broken JSON file)

(remember: Always check the robots.txt at the root of each website that you intend to scrape, before you scrape)
