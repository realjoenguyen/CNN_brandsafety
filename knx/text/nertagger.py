import logging
import nltk
import requests
import codecs

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

class NERTagger(object):
	"""Base class for named entity recognition (NER) tagger objects

	Returns
	=======
	tag : NERtagger
	NPChunker object, typically used by calling its chunk() method
	"""

	def tag(self, text):
		"""Tag the given text, producing list of named entities

		Parameters
		----------
		text: str or iterable
		The sentence to be parsed. If an iterable, it's assumed to be tokenized

		Returns
		-------
		named_entities : list of str
		"""
		return text

class KnorexNERTagger(NERTagger):
	"""Train a maxent NP chunker based on CONLL dataset.

	Parameters
	==========

	Returns
	=======
	chunker : NPChunker
	NPChunker object, typically used by calling its chunk() method
	"""

	def __init__(self):
		self.url = "http://lumina.knorex.com/nerweb/article/getNamedEntities"
		# self.url = "http://lum-demo.knorex.com/nerweb/article/getNamedEntities"
		# self.url = "http://lum-staging.knorex.com/nerweb/article/getNamedEntities"

	def tag(self, text):
		"""Tag the given text, producing list of named entities

		Parameters
		----------
		text: str or iterable
		The sentence to be parsed. If an iterable, it's assumed to be tokenized

		Returns
		-------
		named entities : list of tuple (startPos, endPos, text, type)
		duplicated named entities: list of dictionary
		"""
		payload = {'content':text}
		r = requests.post(self.url, data=payload)
		res, dup_res = list(), list()
		rjson = r.json()
		if u'results' not in rjson:
			return [], []
		for x in rjson[u'results']:
			if u'annotations' not in x or x[u'type'] == u'Address':
				continue
			tmpset, tmplist = set(), list()
			for y in x[u'annotations']:
				text = y[u'text']
				res.append((y[u'startPos'], y[u'endPos'], text, x[u'type']))
				tmplist.append((y[u'startPos'], y[u'endPos'], text, x[u'type']))
				tmpset.add(text)
			if len(tmpset) > 1:
				dup_res.append(tmplist)
		return res, dup_res
