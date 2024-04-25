import docclass

# Page 121
reload(docclass)
cl = docclass.classifier(docclass.getwords)
cl.train('the quick brown fox jumps over the lazy dog', 'good')
cl.train('make quick money in the online casino', 'bad')
cl.fcount('quick', 'good')

c.fcount('quick', 'bad')

# Page 122
reload(docclass)
cl = docclass.classifier(docclass.getwords)
docclass.sampletrain(cl)
cl.fprob('quick', 'good')

# Page 125
reload(docclass)
cl = docclass.naivebayes(docclass.getwords)
docclass.sampletrain(cl)

cl.prob('quick rabbit', 'good')
