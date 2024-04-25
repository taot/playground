# -*- coding: utf8 -*-

import urllib2
from bs4 import *    # beautiful-soup
from urlparse import urljoin
import MySQLdb
import re

import nn

# 构造一个单词列表，这些单词将被忽略
ignorewords = set(['the', 'of', 'to', 'and', 'a', 'in', 'is', 'it'])

mynet = nn.searchnet()

class crawler:
    # 初始化 crawler 类并传入数据库名称
    def __init__(self, dbname = "searchengine"):
        self.con = MySQLdb.connect(user = "root", passwd = "123456", db = dbname)

    def __del__(self):
        self.con.close()

    def dbcommit(self):
        self.con.commit()

    # 辅助函数，用于获取条目的 id，并且如果条目不存在，就将其加入数据库中
    def getentryid(self, table, field, value, createnew = True):
        cur = self.con.cursor()
        cur.execute("select rowid from %s where %s = '%s'" % (table, field, value))
        res = cur.fetchone()
        if res == None:
            cur.execute("insert into %s (%s) values ('%s')" % (table, field, value))
            return cur.lastrowid
        else:
            return res[0]

    # 为每个网页建立索引
    def addtoindex(self, url, soup):
        if self.isindexed(url):
            return
        print 'Indexing %s' % url
        # 获取每个单词
        text = self.gettextonly(soup)
        words = self.separatewords(text)
        print "Got ", len(words), " words on ", url
        # 得到 URL 的 id
        urlid = self.getentryid('urllist', 'url', url)

        # 将每个单词与该 url 关联
        cur = self.con.cursor()
        for i in range(len(words)):
            word = words[i]
            if word in ignorewords:
                continue
            wordid = self.getentryid('wordlist', 'word', word)
            cur.execute("insert into wordlocation (urlid, wordid, location) values (%d, %d, %d)" % (urlid, wordid, i))
            if i % 100 == 0:
                print "insert into wordlocation %d" % i

    # 从一个 HTML 网页中提取文字（不带标签的）
    def gettextonly(self, soup):
        v = soup.string
        if v == None:
            c = soup.contents
            resulttext = ''
            for t in c:
                subtext = self.gettextonly(t)
                resulttext += subtext + '\n'
            return resulttext
        else:
            return v.strip()

    # 根据任何非空白字符进行分词处理
    def separatewords(self, text):
        splitter = re.compile('\\W*')
        return [s.lower() for s in splitter.split(text) if s != '']

    # 如果 url 已经建过索引，则返回 True
    def isindexed(self, url):
        cur = self.con.cursor()
        cur.execute("select rowid from urllist where url = '%s'" % self.con.escape_string(url))
        u = cur.fetchone()
        if u != None:
            cur.execute("select * from wordlocation where urlid = %d" % u[0])
            v = cur.fetchone()
            if v != None:
                return True
        return False

    # 添加一个关联两个网页的链接
    def addlinkref(self, urlfrom, urlto, linktext):
        """Add a link between two pages."""
        fromid = self.getentryid('urllist', 'url', urlfrom)
        toid = self.getentryid('urllist', 'url', urlto)
        if fromid == toid:
            return
        cur = self.con.cursor()
        cur.execute('insert into link (fromid, toid) values (%d, %d)' % (fromid, toid))

        linkid = cur.lastrowid
        # Remember each word in link text
        for word in self.separatewords(linktext):
            if word in ignorewords: continue
            wordid = self.getentryid('wordlist', 'word', word)
            cur.execute('insert into linkwords (wordid, linkid) values (%d, %d)' % (wordid, linkid))

    # 从一小组网页开始进行广度优先搜索，直至某一给定深度，期间为网页建立索引
    def crawl(self, pages, depth = 2):
        for i in range(depth):
            newpages = set()
            for page in pages:
                try:
                    c = urllib2.urlopen(page)
                except:
                    print "Could not open %s" % page
                    continue
                soup = BeautifulSoup(c.read())
                self.addtoindex(page, soup)
                self.dbcommit()

                links = soup('a')
                count = 0
                for link in links:
                    if ('href' in dict(link.attrs)):
                        url = urljoin(page, link['href'])
                        if url.find("''") != -1:
                            continue
                        url = url.split('#')[0]     # 去掉位置部分
                        if url[0:4] == 'http' and not self.isindexed(url):
                            newpages.add(url)
                        linkText = self.gettextonly(link)
                        self.addlinkref(page, url, linkText)

                        self.dbcommit()
                        count += 1
                        if count % 100 == 0:
                            print "addlinkref %d" % count

            pages = newpages

    def calculatepagerank(self, iterations = 20):
        # 清除当前的 PageRank 表
        cur = self.con.cursor()
        cur.execute('drop table if exists pagerank')
        cur.execute('create table pagerank (urlid bigint not null primary key, score decimal(20,10) not null)')
        # 初始化每个 url，令其 PageRnak 值为 1
        cur.execute('insert into pagerank select rowid, 1.0 from urllist')
        self.dbcommit()

        cur.execute('select rowid from urllist')
        urllist = cur.fetchall()
        for i in range(iterations):
            print "Iteration %d" % i

            for (urlid, ) in urllist:
                pr = 0.15
                cur.execute('select distinct fromid from link where toid = %d' % urlid)
                fromids = cur.fetchall()
                # 循环遍历指向当前网页的所有其他网页
                for (linker, ) in fromids:
                    # 得到链接源对应网页的 PageRank 值
                    cur.execute('select score from pagerank where urlid = %d' % linker)
                    linkingpr = float(cur.fetchone()[0])
                    # 根据链接源，求得总的链接数
                    cur.execute('select count(*) from link where fromid = %d' % linker)
                    linkingcount = float(cur.fetchone()[0])
                    pr += 0.85 * (linkingpr / linkingcount)

                cur.execute('update pagerank set score = %f where urlid = %d' % (pr, urlid))

            self.dbcommit()


class searcher:
    def __init__(self, dbname = "searchengine"):
        self.con = MySQLdb.connect(user = "root", passwd = "123456", db = dbname)

    def __del__(self):
        self.con.close()

    def getmatchrows(self, q):
        # 构造查询的字符串
        fieldlist = 'w0.urlid'
        tablelist = ''
        clauselist = ''
        wordids = []
        # 根据空格拆分单词
        words = q.split(' ')
        tablenumber = 0

        cur = self.con.cursor()
        for word in words:
            # 获取单词的 id
            cur.execute("select rowid from wordlist where word = '%s'" % word)
            wordrow = cur.fetchone()
            if wordrow != None:
                wordid = wordrow[0]
                wordids.append(wordid)
                if tablenumber > 0:
                    tablelist += ','
                    clauselist += ' and '
                    clauselist += 'w%d.urlid = w%d.urlid and ' % (tablenumber - 1, tablenumber)
                fieldlist += ', w%d.location' % tablenumber
                tablelist += 'wordlocation w%d' % tablenumber
                clauselist += 'w%d.wordid = %d' % (tablenumber, wordid)
                tablenumber += 1

        # 根据各个分组，建立查询
        fullquery = 'select %s from %s where %s' % (fieldlist, tablelist, clauselist)
        cur.execute(fullquery)
        rows = [row for row in cur]

        return rows, wordids

    def normalizescores(self, scores, smallisbetter = 0):
        vsmall = 0.00001    # 避免被零除
        if smallisbetter:
            minscore = min(scores.values())
            return dict([(u, float(minscore) / max(vsmall, l)) for (u, l) in scores.items()])
        else:
            maxscore = max(scores.values())
            if maxscore == 0:
                maxscore = vsmall
            return dict([(u, float(c) / maxscore) for (u, c) in scores.items()])

    def frequencyscore(self, rows):
        counts = dict([(row[0], 0) for row in rows])
        for row in rows:
            counts[row[0]] += 1
        return self.normalizescores(counts)

    def locationscore(self, rows):
        locations = dict([(row[0], 1000000) for row in rows])
        for row in rows:
            loc = sum(row[1:])
            if loc < locations[row[0]]:
                locations[row[0]] = loc
        return self.normalizescores(locations, smallisbetter = 1)

    def distancescore(self, rows):
        # 如果仅有一个单词，则得分都一样
        if len(rows[0]) <= 2:
            return dict([(row[0], 1.0) for row in rows])
        # 初始化字典，并填入一个很大的数
        mindistance = dict([(row[0], 1000000) for row in rows])
        for row in rows:
            dist = sum([abs(row[i] - row[i - 1]) for i in range(2, len(row))])
            if dist < mindistance[row[0]]:
                mindistance[row[0]] = dist
        return self.normalizescores(mindistance, smallisbetter = 1)

    def countinboundlink(self, toid):
        cur = self.con.cursor()
        cur.execute('select count(*) from link where toid = %d' % toid)
        res = cur.fetchone()
        return res[0]

    def inboundlinkscore(self, rows):
        uniqueurls = set([row[0] for row in rows])
        cur = self.con.cursor()
        inboundcount = dict([(u, self.countinboundlink(u)) for u in uniqueurls])
        return self.normalizescores(inboundcount)

    def getpagerankscore(self, cur, urlid):
        cur.execute('select score from pagerank where urlid = %d' % urlid)
        return cur.fetchone()[0]

    def pagerankscore(self, rows):
        cur = self.con.cursor()
        pageranks = dict([(row[0], self.getpagerankscore(cur, row[0])) for row in rows])
        maxrank = float(max(pageranks.values()))
        normalizedscores = dict([(u, float(l) / maxrank) for (u, l) in pageranks.items()])
        return normalizedscores

    def linktextscore(self, rows, wordids):
        linkscores = dict([(row[0], 0) for row in rows])
        cur = self.con.cursor()
        for wordid in wordids:
            cur.execute('select link.fromid, link.toid from linkwords, link where wordid = %d and linkwords.linkid = link.rowid' % wordid)
            fromidtoids = cur.fetchall()
            for (fromid, toid) in fromidtoids:
                if toid in linkscores:
                    cur.execute('select score from pagerank where urlid = %d' % fromid)
                    pr = float(cur.fetchone()[0])
                    linkscores[toid] += pr
        maxscore = max(linkscores.values())
        normalizedscores = dict([(u, float(l) / maxscore) for (u, l) in linkscores.items()])
        return normalizedscores

    def nnscore(self, rows, wordids):
        urlids = [urlid for urlid in set([row[0] for row in rows])]
        nnres = mynet.getresult(wordids, urlids)
        scores = dict([(urlids[i], nnres[i]) for i in range(len(urlids))])
        return self.normalizescores(scores)

    def getscoredlist(self, rows, wordids):
        totalscores = dict([(row[0], 0) for row in rows])

        # weights = [(1.0, self.frequencyscore(rows))]
        # weights = [(1.0, self.locationscore(rows))]
        # weights = [(1.0, self.inboundlinkscore(rows))]
        weights = [
            (1.0, self.frequencyscore(rows)),
            (1.0, self.locationscore(rows)),
            (1.0, self.pagerankscore(rows)),
            (1.0, self.linktextscore(rows, wordids))
        ]

        for (weight, scores) in weights:
            for url in totalscores:
                totalscores[url] += weight * scores[url]

        return totalscores

    def geturlname(self, id):
        cur = self.con.cursor()
        cur.execute("select url from urllist where rowid = %d" % id)
        res = cur.fetchone()
        return res[0]

    def query(self, q):
        rows, wordids = self.getmatchrows(q)
        scores = self.getscoredlist(rows, wordids)
        rankedscores = sorted([(score, url) for (url, score) in scores.items()], reverse = 1)
        for (score, urlid) in rankedscores[0:10]:
            print '%f\t%s' % (score, self.geturlname(urlid))

        return wordids, [r[1] in rankedscores[0:10]]
