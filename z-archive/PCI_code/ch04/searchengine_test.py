# -*- coding: utf8 -*-

import searchengine

reload(searchengine)
crawler = searchengine.crawler('searchengine')

pages = ['https://en.wikipedia.org/wiki/Julia_(programming_language)']
crawler.crawl(pages)

reload(searchengine)
e = searchengine.searcher()
e.query('functional programming')
