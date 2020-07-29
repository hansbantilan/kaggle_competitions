To set up this scrapy project, run the following in terminal then place source code, e.g. foodpanda_spider.py` with `name = "foodpanda"' attribute, in the /platforms/spiders`directory of the platforms project:

`scrapy startproject platforms`

To run the spider and sent the output of the yield expression to `foodpanda.json`, run the following in terminal at the top directory of the platforms project:

`scrapy crawl foodpanda -o foodpanda.json`

(remember: Scrapy appends to a given file instead of overwriting its contents. If you run this command twice without removing the file before the second time, you’ll end up with a broken JSON file)
(remember: Always check the robots.txt at the root of each website that you intend to scrape, before you scrape)
