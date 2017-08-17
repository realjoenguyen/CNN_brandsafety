from gensim import parsing
import cPickle as pkl
from os import listdir
from os.path import isfile, join
import sys
from knx.text.preprocess_text import NormalizationText
import re
import nltk
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
from stop_words import STOP_WORDS_origin

source_dir = sys.argv[1]
output_dir = sys.argv[2]
file_list = [ f for f in listdir(source_dir) if isfile(join(source_dir,f)) ]

print 'Clean data ... at', source_dir

def remove_duplicate(raw):
    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    res = nltk.sent_tokenize(raw)
    res = [e.strip() for e in res]
    res = f7(res)
    return ' '.join(res)

def contain_digit(string):
    for e in string:
        if e.isdigit():
            return True
    return False

import gensim
for true_filename in file_list:
    file_path = join(source_dir, true_filename)

    f_reader = open(file_path, "r")
    raw = f_reader.read()
    raw = NormalizationText.normalize_to_unicode(raw).strip()
    raw = raw.lower()
    # raw = raw.decode('utf-8').strip()
    # raw = remove_duplicate(raw)
    for e in REPLACE_WORDS:
        raw.replace(e, '')

    words = list(gensim.utils.tokenize(raw, lowercase=True))
    words = [e for e in words if e.lower() not in STOP_WORDS_origin() and len(e) < 15 and not contain_digit(e)]
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    words = [lmtzr.lemmatize(e) for e in words]
    content = ' '.join(words)

    output_file_path = join(output_dir, true_filename)
    f_writer = open(output_file_path,"w")
    f_writer.write(content)
    f_writer.close()
