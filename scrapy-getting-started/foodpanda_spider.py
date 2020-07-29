import scrapy

class FoodpandaSpider(scrapy.Spider):
	name = "foodpanda"
	
	def start_requests(self):
		urls = [
			'https://www.foodpanda.com/'
		]
		for url in urls:
			yield scrapy.Request(url=url, callback=self.parse)
	
	def parse(self, response):
	    # Write to file an image of the site's html
		self.html_image(response) 
		# Yield a dictionary for each country in a list of strings
		for country in response.css('.radius img').xpath('@alt').getall():
			yield {'country': country}

	def html_image(self,response):
		filename = 'image_foodpanda.html'
		with open(filename, 'wb') as f:
			f.write(response.body)
		self.log('Saved file %s' % filename)
