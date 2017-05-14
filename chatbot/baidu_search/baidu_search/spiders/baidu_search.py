import scrapy

class BaiduSearchSpider(scrapy.Spider):
    name = "baidu_search"
    allowed_domains = ["baidu.com"]
    start_urls = [
            "https://www.baidu.com/s?wd=机器学习"
    ]

    count = 0

    def parse(self, response):
        filename = "result.html"
        with open(filename, 'wb') as f:
            f.write(response.body)
        hrefs = response.selector.xpath('//div[contains(@class, "c-container")]/h3/a/@href').extract()
        for href in hrefs:
            # print(href)
            yield scrapy.Request(href, callback=self.parse_url)

    def parse_url(self, response):
        print(response.url + ": " + str(len(response.body)))
        with open("result_" + str(self.count) + ".html", 'wb') as f:
            f.write(response.body)
        self.count += 1
