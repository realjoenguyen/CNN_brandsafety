# f = open('stop_words.txt','r')
# STOPWORDS = f.read()
# import re
# import string
#
# def strip_punctuation(str):
#     # deleted_not_dot = string.punctuation.replace('.', '')
#     regex = re.compile('[%s]' % re.escape(string.punctuation))
#     return regex.sub('', str)
#
# for e in set(STOPWORDS.split()):
#     e = strip_punctuation(e)
#     print "'{0}', ".format(e)
# f = open('stop_words.txt','r')
# STOPWORDS = f.read()
# import re
# import string
#
# for e in set(STOPWORDS.split()):
#     print "\"{0}\", ".format(e)
# import sys
# 
# reload(sys)
# sys.setdefaultencoding('utf8')
# def Cal():
#     from nltk.corpus import stopwords
#     f = open('stop_words.txt', 'r')
#     content = f.read()
#     STOP_WORDS = [e.encode('ascii', 'ignore') for e in content.split()]
#     STOP_WORDS = list(set(stopwords.words('english')).union(STOP_WORDS))
#     return STOP_WORDS
# 
# STOP_WORDS = list(Cal())
# print STOP_WORDS

STOP_WORDS = ['their',
'have',
'ain',
'aren',
'couldn',
'hadn',
'hasn',
'haven',
'isn',
'mightn',
'mustn',
'needn',
'shan',
'shouldn',
'wasn',
'weren',
'won',
'wouldn']

# from nltk import corpus
# nltk_stopwords = corpus.stopwords.words("english")
# STOP_WORDS.extend(nltk_stopwords)
# STOP_WORDS = list(set(STOP_WORDS))
# for e in STOP_WORDS:
#     print "\"{0}\", ".format(e)