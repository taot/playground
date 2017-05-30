# import importlib,sys
# importlib.reload(sys)
# sys.setdefaultencoding( "utf-8" )

import scrapy
import re
from w3lib.html import remove_tags

from baidu_search.items import BaiduSearchItem

remove_space_pattern = re.compile('\s+')
def remove_space(s):
    return re.sub(remove_space_pattern, '', s)

class BaiduSearchSpider(scrapy.Spider):
    name = "baidu_search"
    allowed_domains = ["baidu.com"]
    start_urls = [
            "https://www.baidu.com/s?wd=机器学习"
    ]

    def parse(self, response):
        containers = response.selector.xpath('//div[contains(@class, "c-container")]')
        for container in containers:
            href = container.xpath('h3/a/@href').extract()[0]
            title = remove_tags(container.xpath('h3/a').extract()[0])
            c_abstract = container.xpath('div/div/div[contains(@class, "c-abstract")]').extract()
            c2_abstract = container.xpath('div/div/p').extract()
            # c2_abstract = remove_tags(container.xpath('div/div/p').extract_first()).strip()
            abstract = ""
            if len(c2_abstract) > 0:
                abstract = remove_tags(c2_abstract[0])
            if len(c_abstract) > 0:
                abstract = remove_tags(c_abstract[0]).strip()
            abstract = remove_space(abstract)
            request = scrapy.Request(href, callback=self.parse_url)
            request.meta['title'] = title
            request.meta['abstract'] = abstract
            yield request

    def parse_url(self, response):
        print("url: " + response.url)
        print("title: " + response.meta['title'])
        print("abstract: " + response.meta['abstract'])
        content = remove_tags(response.selector.xpath('//body').extract()[0])
        content = remove_space(content)
        print("content_len: " + str(len(content)))
        item = BaiduSearchItem()
        item["url"] = response.url
        item["title"] = response.meta['title']
        item["abstract"] = response.meta['abstract']
        item["content"] = content
        yield item
