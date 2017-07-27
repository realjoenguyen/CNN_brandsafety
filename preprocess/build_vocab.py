#I'd suggest to reduce the size of the vocabulary by selecting top-k most frequent words / top-k by TfIdf. ' \
# 'Most words occur only once or few times in your dataset.

from collections import Counter
import cPickle as pkl
from os import listdir
from os.path import isfile, join
import sys
from gensim import parsing
from nltk.stem.porter import *
from knx.text.preprocess_text import NormalizationText

REPLACE_WORDS = [
    'editing by',
    'editing by',
    'editing by',
    'reporting by',
    'reporting by',
    'reporting by',
    'compiled by',
    'compiled by',
    'compiled by',
    'writing by',
    'writing by',
    'writing by',
    'u.s.',
    'u.n.'
]
from stop_words import STOP_WORDS
traindir = sys.argv[1] 
vocab_output = sys.argv[2]
train_file_list = [ f for f in listdir(traindir) if isfile(join(traindir,f)) ]
vocab = Counter()
max_words_len, accumulated_words_len = 0, 0

print "Building vocab from", traindir

def contain_digit(string):
    for e in string:
        if e.isdigit():
            return True
    return False

from unidecode import unidecode
from knx.util.logging import Timing

df = Counter()
with Timing("Evaluating model ..."):
    for true_filename in train_file_list:
        file_path = join(traindir, true_filename)
        f_reader = open(file_path, "r")

        words = []
        for line in f_reader:
            if 'preprocess' in traindir:
                # vocab.update(line.split())
                words = [e for e in line.split() and e not in STOP_WORDS] 
            else:
                text_line = NormalizationText.normalize_to_unicode(line).strip()
                words = parsing.preprocessing.preprocess_string(text_line)
                words = [e for e in words if e.lower() not in STOP_WORDS and len(e) < 15]

        df.update(set(words))
        vocab.update(words)
        f_reader.close()

for k, v in df.items():
    if v < 3 and v > len(train_file_list) / 2:
        del vocab[k]

print "Vocab size:", len(vocab)
print vocab.most_common(100)
print "Dumping vocab files into", vocab_output, '\n'
pkl.dump(vocab,open(vocab_output,'wb'))
