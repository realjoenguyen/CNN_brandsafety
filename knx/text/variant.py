#!/usr/bin/python
import os
import logging
import re
import codecs
from pymongo import MongoClient

# from knx.text.doc_to_feature import DocToFeature

DEBUG = False
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)
if DEBUG:
    LOGGER.setLevel(logging.DEBUG)

client = MongoClient('localhost', 27017)
db = client['wikipedia']
coll = db['TitleId']
# dtf = DocToFeature()
# tokenizer = dtf.get_tokenizer()

seperators = {'/', '-'}
def contain_sep(token):
	for sep in seperators:
		if sep in token:
			return sep
	return str()

def SymbolicVarTok(token1, token2):	
	sep1, sep2 = contain_sep(token1), contain_sep(token2)
	if not sep1 and not sep2 :
		return False
	if sep1 and token1.replace(sep1, sep2) == token2:
		return True
	if sep2 and token2.replace(sep2, sep1) == token1:
		return True
	return False

def SymbolicVar(tokens1, tokens2):
	res = []
	for t1 in tokens1:
		for t2 in tokens2:
			if SymbolicVarTok(t1, t2):
				res.append((t1, t2))
	return res

def CompoundVar(tokens1, tokens2):
	i1, i2 = 0, 0
	l1, l2 = len(tokens1), len(tokens2)
	res = []
	while True:
		ti1, ti2 = tokens1[i1], tokens2[i2]
		id12, id21 = ti1.find(ti2), ti2.find(ti1)
		if id12 == 0 and len(ti1) > len(ti2):
			sep = contain_sep(ti1)
			l = 0
			while True:
				i2 += 1
				if i2 >= l2:
					break
				l += 1
				ti2 += sep + tokens2[i2]
				if ti1.find(ti2) != 0:
					break
				if ti1 == ti2:
					res.append((tokens2[i2 - l : i2 + 1], ti1))
		elif id21 == 0 and len(ti2) > len(ti1):
			sep = contain_sep(ti2)
			l = 0
			while True:
				i1 += 1
				if i1 >= l1:
					break
				l += 1
				ti1 += sep + tokens1[i1]
				if ti2.find(ti1) != 0:
					break
				if ti1 == ti2:
					res.append((tokens1[i1 - l : i1 + 1], ti2))
		else:
			i1 += 1
			i2 += 1
		if i1 >= l1 or i2 >= l2:
			break
	return res

def InflectionVar(tokens1, tokens2):
	"""
		Only consider the last words of phrase which ends with 's'
	"""
	def compare(t1, t2):
		if t1[-1] == 's' and t1[0:-1] == t2 and t1[-2] == t2[-1]:  return True
		if t1[-2:] == 'es':
			if t1[:-2] == t2:  return True
			if t1[:-3] == t2[:-1] and t1[-3] == 'i' and t2[-1] == 'y':  return True
		return False

	if len(tokens1) != len(tokens2):
		return False
	for i in range(len(tokens1) - 1):
		if tokens1[i].lower() != tokens2[i].lower():
			return False

	t1, t2 = tokens1[-1].lower(), tokens2[-1].lower()
	return compare(t1, t2) or compare(t2, t1)

def wikiId(phrase):
	s = phrase.upper().replace(u' ', u'_')
	f = coll.find_one({'title': s}, {'_id': False, 'title': False})
	if f == None:
		return unicode()
	else:
		return f[u'id']

def updatePhrases(phrases, statFile):
	f = codecs.open(statFile, "r", "utf-8")
	for line in f:
		s = line.strip(u'\n ')
		if s in phrases:
			phrases[s][0] +=1
		else:
			phrases[s] = [1, wikiId(s), s.split()]
	f.close()

def updateMapping(phrases):
	def CountPhrase(phrase):
		return phrases.get(phrase, 0)

	mappingP = dict()
	inv = dict()
	for phrase in phrases:
		count, wiki, tmp = phrases[phrase]
		if not wiki:
			continue
		if wiki in inv:
			inv[wiki].add(phrase)
		else:
			inv[wiki] = {phrase}

	for wiki in inv:
		dupset = inv[wiki]
		maxEle, countMax = str(), 0
		for i, ele in enumerate(dupset):
			if i == 0 or CountPhrase(ele) > countMax:
				maxEle = ele
				countMax = CountPhrase(maxEle)
		for ele in dupset:
			if ele != maxEle:
				mappingP[ele] = maxEle

	# Normalize based on morphological features
	# for i1, phrase1 in enumerate(phrases):
	# 	for i2, phrase2 in enumerate(phrases):
	# 		if i2 <= i1:
	# 			continue
	# 		tokens1, tokens2 = phrases[phrase1][2], phrases[phrase2][2]
	# 		if InflectionVar(tokens1, tokens2):
	# 			if CountPhrase(phrase1) > CountPhrase(phrase2):
	# 				mappingP[phrase2] = phrase1
	# 			else:
	# 				mappingP[phrase1] = phrase2
	return mappingP

class Normalizer:
	"""Normalizer transforms phrases to canonical forms"""
	def __init__(self, statFile=str()):
		self.phrases = dict()
		self.mappingP = dict()
		if statFile:
			updatePhrases(self.phrases, statFile)
			print "Done"
			self.mappingP = updateMapping(self.phrases)

	def clearMem(self):
		self.phrases.clear()
		self.mappingP.clear()

	def addFile(self, statFile):
		updatePhrases(self.phrases, statFile)

	def normalize(self, lPhrases):
		"""
			Return a new list of normalized phrases
		"""
		phrases = self.phrases
		mappingP = self.mappingP

		for i, phrase in enumerate(lPhrases):
			if phrase in mappingP:
				lPhrases[i] = mappingP[phrase]

		return lPhrases
