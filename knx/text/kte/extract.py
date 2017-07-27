#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib
matplotlib.use('Agg')
import logging
import argparse
import cPickle as pickle
import gc
from itertools import combinations
import json
import math
# import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import re
import scipy.optimize as optimize
from scipy.sparse import coo_matrix, csr_matrix
import scipy
import sys
import threading
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler, StaticFileHandler, asynchronous
import time
from pulp import *
from pymongo import MongoClient

from BS.knx.text.chunker import MaxentNPChunker
from BS.knx.text import KnorexNERTagger
from BS.knx.text import DocToFeature, Lemmatizer, tf_to_tfidf, tf_to_okapi, tf_to_midf
from BS.knx.text.feature_to_arff import FeatureToArff
from knx.util.logging import Timing, Unbuffered
from knx.version import VERSION

logging.basicConfig(level=logging.INFO)
DEBUG = False
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)
###################
# Helper function #
###################
def loadarff(filename):
    """Load ARFF file which contains feature vectors with the document name as the last index

    **Parameters**

    filename : string
        The ARFF file to be loaded

    **Returns**

    tf_score : coo_matrix
        The feature vectors

    concepts : list
        The list of concept names (as document names)

    **Notes**

    This method assumes that the ARFF file is structured such that the last column will be a string attribute
    """
    with open(filename, 'r') as arff:
        lines = arff.readlines()
    num_attr = 0
    for idx, line in enumerate(lines):
        if line.startswith('@attribute'):
            num_attr += 1
        if line.startswith('@data'):
            dataIdx = idx + 1
            break
    num_attr -= 1
    lines = lines[dataIdx:]
    is_sparse = (lines[0][0] == '{')

    if is_sparse:
        data = []
        indices = []
        indptr = [0]
        concepts = []
        for row, line in enumerate(lines):
            # Read sparse
            is_sparse = True
            line = line.strip('{} \r\n\t')
            tmp = line.rsplit(' ', 1)
            items = map(int, re.split('[, ]', tmp[0]))
            concepts.append(re.sub(r'(?<!\\)_', ' ', tmp[1][1:-1]).replace('\\\'', '\'').replace(r'\_', '_'))
            data.extend(items[1::2])
            indices.extend(items[0:-1:2])
            indptr.append(indptr[-1] + ((len(items) - 1) / 2))
        data = np.array(data, dtype=float)
        indices = np.array(indices, dtype=np.intc)
        indptr = np.array(indptr, dtype=np.intc)
        return (csr_matrix((data, indices, indptr), shape=(len(concepts), num_attr)), concepts)
    else:
        data = []
        concepts = []
        for row, line in enumerate(lines):
            # Read dense
            line = line.strip(' \r\n\t')
            values = line.split(',')
            data.append(values[:len(values) - 1])
            concepts.append(values[-1])
        return (coo_matrix(data, dtype=float), concepts)

def update_seid(t_phrase, reduce_id, sid, eid):
    if sid == -1 or eid == -1:
        return sid, eid
    #Update start index (sid)
    reduce_id = sorted(reduce_id)
    prev, bias = 0, False
    for idx in reduce_id:
        if idx == prev:
            prev += 1
            sid += len(t_phrase[idx][0]) + 1
            if t_phrase[idx][0] == "%":
                bias = True
        else:
            break
    if bias or (prev + 1 < len(t_phrase) and t_phrase[prev + 1][0] == "%"):
        sid -= 1

    #Update end index (eid)
    reduce_id = sorted(reduce_id, reverse=True)
    prev = len(t_phrase) - 1
    for idx in reduce_id:
        if idx == prev:
            prev -= 1
            eid -= len(t_phrase[idx][0]) + 1
        else:
            break
    return sid, eid

def filter_noun_phrases(phrases, lemmatizer=None):
    """Filter phrases and produce phrases suitable as key phrases

    **Parameters**

    phrases : list of phrases
        The list of phrases where each phrase is in (word, pos) format, where pos is the POS tag for the word

    lemmatizer : the lemmatizer to be used (optional)
        The lemmatizer should implement the method "lemmatize" that accepts a word to be lemmatized and an optional POS
    """
    if lemmatizer is None:
        lemmatizer = Lemmatizer()

    def lemmatize_nouns(word_pos):
        word, pos = word_pos
        # Initially we want to remove if pos in {'NN', 'NNS'}, but that caused "Bras Basah" in start of sentence
        # to be lemmatized as "Bra Basah"
        # Need to couple this with NER, but until that happens, let's not make such silly lemmatization
        # by considering only lowercase words

        if word.islower() and pos.startswith('NN'):
            lemmatized = lemmatizer.lemmatize(word.lower(), 'n')
            word = lemmatized
        return (word, pos)
    
    def reduce_phrase(phrase):
        # Remove all words with those tags
        DISALLOWED = {'DT', 'EX', 'LS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'WDT',
                        'WP', 'WP$', 'WRB', "''", '``'} #, 'SYM', '$', 'CD'
        t_phrase, sid, eid = phrase
        res, reduce_id = list(), list()
        for i, word_pos in enumerate(t_phrase):
            if word_pos[1] in DISALLOWED:
                reduce_id.append(i)
                continue
            res.append(word_pos)

        sid, eid = update_seid(t_phrase, reduce_id, sid, eid)
        return (res, sid, eid)

    def lemmatize_phrase(phrase):
        t_phrase, sid, eid = phrase
        res = []
        prev = tuple()
        for word_pos in t_phrase:
            if len(prev) == 0 or prev[1] != 'CD':
                res.append(lemmatize_nouns(word_pos))
                prev = word_pos
            else:
                res.append(word_pos)
        return (res, sid, eid)
    
    def contains_noun(phrase):
        return any(pos.startswith('NN') for word, pos in phrase[0])
    
    def remove_cd(phrase):
        t_phrase, sid, eid = phrase
        pos_str = ' '.join(zip(*t_phrase)[1])
        if pos_str in {'CD NNS', 'CD JJ NNS', 'CD NN NNS'}:
            sid += len(t_phrase[0][0]) + 1
            del t_phrase[0:1]
            # return False
        elif pos_str in {'CD CD NNS', 'CD CD JJ NNS', 'CD CD NN NNS'}:
            sid += len(t_phrase[0][0]) + 1 + len(t_phrase[1][0]) + 1
            del t_phrase[0:2]
            # return False
        return (t_phrase, sid, eid)
        # return True

    def remove_boundary(phrase):
        def traverse(phrase, rev=False, pos=0):
            res = []
            tmp = (list(reversed(phrase)) if rev else phrase[:])
            n = len(phrase)
            for i, word_pos in enumerate(tmp):
                if i == pos:
                    if word_pos[1] in NON_BOUNDARY_POS or (word_pos in NON_BEGIN_WORDS and not rev) or\
                        (word_pos in NON_END_WORDS and rev):
                        if rev:
                            res.append(n - i - 1)
                        else:
                            res.append(i)
                        pos += 1
                else:
                    break
            return res
        NON_BOUNDARY_POS = {'CC', 'IN', ',', '.', '-LSB-', '-RSB-', '-LRB-', '-RRB-',  'CD', 'SYM', '$'} #,
        NON_BEGIN_WORDS = {('least','JJS')}
        NON_END_WORDS = {('one','NN'), ('ones','NNS'), ('thing', 'NN'), ('things', 'NNS')}
        t_phrase, sid, eid = phrase
        reduce_id = traverse(t_phrase)
        s = (max(reduce_id) + 1 if reduce_id else 0)

        r1 = traverse(t_phrase, rev=True)
        e = (min(r1) if r1 else len(t_phrase))
        reduce_id.extend(r1)
        sid, eid = update_seid(t_phrase, reduce_id, sid, eid)

        # res = [word_pos for i, word_pos in enumerate(t_phrase) 
        #         if word_pos[1] not in NON_BOUNDARY or 0 < i < len(t_phrase) - 1]
        return (t_phrase[s:e], sid, eid)

    def is_not_null(phrase):
        t_phrase, sid, eid = phrase
        if len(t_phrase) == 0:
            return False
        return True

    phrases = [reduce_phrase(phrase) for phrase in phrases]
    phrases = filter(is_not_null, phrases)
    phrases = [remove_cd(phrase) for phrase in phrases]
    # phrases = [lemmatize_phrase(phrase) for phrase in phrases]
    phrases = [remove_boundary(phrase) for phrase in phrases]
    phrases = filter(contains_noun, phrases)
    return phrases


###########################
# Functions for debugging #
###########################
def arctanh_param(a, b, c, d, x):
    b1 = np.tanh(b) / max(x)
    c1 = np.tanh(c)
    x1 = b1 * x ** 2 + c1
    x1[x1 > 1] = 1
    x1[x1 < -1] = -1
    return a * np.arctanh(x1) + d


def residuals(p, y, x):
    a, b, c, d = p
    return y - arctanh_param(a, b, c, d, x)
    # return x * (y - arctanh_param(a, b, c, d, x))  # For fitting more closely to the higher weights


def fit(x, y):
    """Fit the data (x, y) with the parameterized arctanh function: a * arctanh(bx + c) + d
    """
    p0 = np.array([1, 0, 0.1, 0], dtype=np.float64)
    plsq, cov_x, infodict, mesg, ier = optimize.leastsq(residuals, p0, args=(y, x), maxfev=5000, full_output=True)
    a, b, c, d = plsq
    b1 = np.tanh(b) / (len(x) - 1)
    c1 = (1 - np.abs(b1)) * np.tanh(c)
    print 'Fitted function:'
    print 'y = %.4E * arctanh(%.4Ex + %.4E) + %.4E' % (a, b1, c1, d)
    print ier, mesg
    print infodict['nfev']
    return arctanh_param(a, b, c, d, x)


def draw_vector(vect):
    vect_arr = np.array(sorted(vect.toarray()[0]), dtype=np.float64)
    vect_arr = vect_arr[-1000:]
    vect_arr = vect_arr[vect_arr > 0]
    if len(vect_arr) == 0:
        return
    x = np.arange(len(vect_arr), dtype=np.float64)
    y = np.array(vect_arr)
    z = fit(x, y)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth="2", label='data', alpha=0.5)
    plt.plot(x, z, linewidth="2", linestyle='--', color='k', label='arctanh least-square fit')
    plt.legend(loc='upper center')
    plt.savefig('vector.png')

####################
# Arguments parser #
####################


class FileAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values[0])
        if len(values) >= 2:
            setattr(namespace, self.dest + '_arff', values[1])
        if len(values) >= 3:
            setattr(namespace, self.dest + '_vocab', values[2])


class ArffAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values[0])
        if len(values) > 1:
            setattr(namespace, self.dest + '_vocab', values[1])


class StartAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == '--start_stdin':
            setattr(namespace, self.dest, 2)
        elif option_string == '--nostart':
            setattr(namespace, self.dest, 0)
        else:
            setattr(namespace, self.dest, 1)

scorers = {
    'tf': 'scorer_tf',
    'tfidf': 'scorer_tfidf',
    'midf': 'scorer_midf',
    'okapi': 'scorer_okapi'
}


def parse_arguments():
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(description=('Starts the concept extractor, or test the concept extractor if '
                                     'either --testarff or --testdir is specififed'),
                                     epilog=('Due to the structure of the optional arguments, please provide the term '
                                             'weighting scheme immediately after the program name'))
    parser.add_argument('scorer_name', choices=scorers.keys(),
                        help='The term weighting scheme used to process the raw TF counts')
    parser.add_argument('-p', '--port', dest='port', default=8205, type=int,
                        help='Specify the port in which the server should start (default to 8205)')
    parser.add_argument('--boost_method', dest='boost_method', metavar='[012]', default=0, type=int,
                        help=('Specify the boosting method that will be used. 0 means no boosting, 1 means concept '
                              'boosting, 2 means concept boosting with amplification (defaults to 0)'))
    parser.add_argument('--word_normalization', dest='word_normalization', metavar='{stem|lemmatize|none}',
                        default='stem',
                        help=('Specify how words should be normalized. Options: stem, lemmatize, none\n'
                              '(defaults to stem)'))
    parser.add_argument('--n_jobs', dest='n_jobs', action='store', default=1, type=int,
                        help='The number of processes to run')

    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument('--start_stdin', nargs=0, dest='start', default=1, action=StartAction,
                             help='Flag to start receiving input from standard input. Default is to run as server.')
    start_group.add_argument('--nostart', nargs=0, dest='start', default=1, action=StartAction,
                             help='Flag to not start anything. Default is to run as server')

    lowercase_group = parser.add_mutually_exclusive_group()
    lowercase_group.add_argument('--lowercase', dest='lowercase', default=True, action='store_true',
                                 help='Enable lowercasing on each word')
    lowercase_group.add_argument('--nolowercase', dest='lowercase', default=True, action='store_false',
                                 help='Disable lowercasing on each input word (default)')

    keep_nnp_group = parser.add_mutually_exclusive_group()
    keep_nnp_group.add_argument('--keep_nnp', dest='keep_nnp', default=False, action='store_true',
                                help=('Enable keeping words with POS NNP or NNPS intact, without stemming or '
                                      'lowercasing'))
    keep_nnp_group.add_argument('--nokeep_nnp', dest='keep_nnp', default=False, action='store_false',
                                help=('Disable keeping words with POS NNP or NNPS intact, without stemming or '
                                      'lowercasing (default)'))

    transliteration_group = parser.add_mutually_exclusive_group()
    transliteration_group.add_argument('--transliteration', dest='transliteration', default=True, action='store_true',
                                       help=('Enable transliteration on Unicode input (e.g., é into e, “ to ") '
                                             '(default)'))
    transliteration_group.add_argument('--notransliteration', dest='transliteration', default=True,
                                       action='store_false',
                                       help='Disable transliteration on Unicode input (e.g., é into e, “ to ")')

    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument('--logging', dest='logging', action='store_true', default=True,
                               help='Enable printing timing messages for each component (default)')
    logging_group.add_argument('--nologging', dest='logging', action='store_false', default=True,
                               help='Disable printing timing messages for each component')

    train_source = parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument('--traindir', dest='traindir', action=FileAction, nargs='+',
                              metavar=('traindir', 'arff_output [vocab_output]'),
                              help=('Training data comes from texts in a directory. If arff_output is provided, the '
                                    'raw TF counts is written to arff_output. If vocab_output is also provided, the '
                                    'vocabulary is dumped into vocab_output'))
    train_source.add_argument('--trainarff', dest='trainarff', action=ArffAction, nargs=2,
                              metavar=('train.arff', 'vocabulary_file'),
                              help='Training data comes from ARFF file, with the specified vocabulary')
    train_source.add_argument('--trainpickle', dest='trainpickle', action=ArffAction, nargs=2,
                              metavar=('train.pickle', 'vocabulary_file'),
                              help='Training data comes from pickled csr_matrix file, with the specified vocabulary')

    test_source = parser.add_mutually_exclusive_group()
    test_source.add_argument('--testdir', dest='testdir', action=FileAction, nargs='+',
                             metavar=('testdir', 'arff_output'),
                             help='Test from a folder. If arff_output is provided the raw TF counts is written '
                                  'to arff_output.')

    return parser.parse_args(sys.argv[1:])


class KeyTermsExtractor(object):

    """Extract key concepts and key words from a given input text

    This class is instantiated with a path to a training directory or training arff

    **Parameters**

    scorer_name : string, 'tfidf' by default
        The scorers to be used to compute the term weight matrix.
        Available options are:

            * 'tf' : Plain term frequency.
            * 'tfidf' : Standard TF-IDF term weighting scheme.
            * 'midf' : Modified TF-IDF term weighting scheme.
            * 'okapi' : Okapi BM25 term weighting scheme.

    traindir : string, None by default
        The directory containing training files.

        This directory is expected to contain files which names represent the concept they contain.

        See also traindir_arff and traindir_vocab for more information.

        Either this option, trainarff, or trainpickle must be specified.

    traindir_arff : string, optional, None by default
        The filename to dump the ARFF file containing term frequency generated during training from traindir.

    traindir_vocab : string, optional, None by default
        The filename to dump the vocab file containing the vocabulary mapping generated during training from traindir.

    trainarff : string, None by default
        The ARFF file containing term frequency as the training data.

        The ARFF file is expected to be compatible with `loadarff` method, that is, the last column is expected to be a
        string attribute.

        This option requires trainarff_vocab to be specified also.

        See also trainarff_vocab.

        Either this option, traindir, or trainpickle must be specified.

    trainarff_vocab : string, required if trainarff is used, None by default
        The vocab file containing pickled dictionary of vocabulary mapping.

    trainpickle : string, None by default
        The pickle file containing term frequency as the training data.

        This option requires trainpickle_vocab to be specified also.

        Either this option, traindir, or trainarff must be specified.

    trainpickle_vocab : string, required if trainpickle is used, None by default
        The vocab file containing pickled dictionary of vocabulary mapping.

    testdir : string, optional, None
        The directory containing files from which top concepts and words will be extracted.

        If this option of testarff is specified, the result will be printed to console.

        See also testdir_arff for more information.

    testdir_arff : string, optional, None by default
        The filename to dump the ARFF file containing term frequency generated from files in testdir

    testarff : string, optional, None by default
        The ARFF file containing term frequency as the test data.

        The ARFF file is expected to be compatible with `loadarff` method, that is, the last column is expected to be a
        string attribute.

    testarff_vocab : string, optional, None by default
        The vocab file containing pickled dictionary of vocabulary mapping.

        This is optional since in normal cases, the vocabulary from training phase is the one to be used. The use of
        this option will trigger synchronization of this vocabulary with the one generated during training.
        This option is available in the case the test ARFF file is heavy to generate.
        But for best result, please use testdir option.

    lowercase : boolean, True by default
        Whether to convert the input text into lowercase

    keep_nnp : boolean, False by default
        Whether to keep words with POS NNP and NNPS intact, that is, without converting it into lowercase or
        stemming it.

    transliteration : boolean, True by default
        Whether to do transliteration on Unicode input. For example, it will convert LEFT DOUBLE QUOTATION MARK into
        ASCII version double quotes

    boost_method : int, 1 by default
        The boosting method to be used:

            * 0 : No boosting (i.e., core algorithm)
            * 1 : Core + concept boosting
            * 2 : Core + concept boosting with amplification

        Read the documentation in https://wiki.knorex.asia/x/rAMYBQ

    port : int, 8205 by default
        The port number that will be used by `start_server` method to serve the extraction API

    logging : boolean, True by default
        Whether to print timing messages for each component in the code

    **Attributes**

    `forward_index_` : csr_matrix
        The terms (rows) to concept definitions (columns) matrix

    `inverted_index_` : csr_matrix
        The transpose of `forward_index_`

    `term_concept_index_` : csr_matrix
        The terms (rows) to concept name (columns) matrix

    `vocabulary_` : dict
        The mapping from terms to indices

    `mapping_` : dict
        The inverse mapping of vocabulary

    `concepts_` : list
        The list of concept names
    """
    ##################
    # Class constant #
    ##################
    COTH1 = 1 / np.tanh(1)

    def __init__(self, scorer_name='tfidf', traindir=None, traindir_arff=None, traindir_vocab=None,
                 trainarff=None, trainarff_vocab=None, testdir=None, testdir_arff=None,
                 trainpickle=None, trainpickle_vocab=None,
                 testarff=None, testarff_vocab=None, lowercase=True, keep_nnp=False, transliteration=True,
                 word_normalization='stem', boost_method=1, port=8205, logging=True):
        for karg, value in locals().items():
            setattr(self, karg, value)
        if self.traindir is None and self.trainarff is None and self.trainpickle is None:
            module_path = os.path.dirname(__file__)
            self.trainarff = os.path.join(module_path, 'training_data/wikipedia_larger.arff')
            self.trainarff_vocab = os.path.join(module_path, 'training_data/wikipedia_larger.vocab')
            self.lowercase = True
            self.keep_pos = False
            self.transliteration = True
        self.forward_index_ = None
        self.inverted_index_ = None
        self.term_concept_index_ = None
        self.concepts_ = None
        self.vocabulary_ = None
        self.mapping_ = None

    @property
    def boost_method(self):
        return self._boost_method

    @boost_method.setter
    def boost_method(self, value):
        self._boost_method = value
        if value == 1:
            self.boost_concept = True
            self.boost_lower = False
        elif value == 2:
            self.boost_concept = True
            self.boost_lower = True
        else:
            self.boost_concept = False
            self.boost_lower = False

    @staticmethod
    def gk_rank_similarity(input_rank, concept_rank):
        """Compare two rankings based on Goodman and Kruskal's rank correlation
        """
        word_to_rank_input = dict((word, rank) for rank, word in enumerate(input_rank))
        word_to_rank_concept = dict((word, rank) for rank, word in enumerate(concept_rank))
        N_s = 0.0
        N_d = 0.0
        for word1, word2 in combinations(input_rank, 2):
            order_input = np.sign(word_to_rank_input[word1] - word_to_rank_input[word2])
            order_concept = np.sign(word_to_rank_concept[word1] - word_to_rank_concept[word2])
            if order_input == order_concept:
                N_s += 1.0
            else:
                N_d += 1.0
        return (1 + ((N_s - N_d) / (N_s + N_d))) / 2

    @staticmethod
    def spearman_rank_similarity(input_rank, concept_rank):
        """Compare two rankings based on Spearman's rank correlation
        """
        word_to_rank_input = dict((word, rank) for rank, word in enumerate(input_rank))
        rank_diff = float(sum((rank - word_to_rank_input[word]) ** 2 for rank, word in enumerate(concept_rank)))
        size = len(input_rank)
        return 1 - (3 * rank_diff) / (size ** 3 - size)

    #########################################
    # Term weighting scheme helper function #
    #########################################
    def scorer_tf(self, doc_term_freq, concepts=None, use_existing_data=False):
        return doc_term_freq

    def scorer_tfidf(self, doc_term_freq, concepts=None, use_existing_data=False, norm='l2'):
        if use_existing_data:
            (doc_term, _) = tf_to_tfidf(doc_term_freq, idf_diag=self.idf_diag, sublinear_tf=True, smooth_idf=True,
                                        norm=norm)
        else:
            (doc_term, idf_diag) = tf_to_tfidf(doc_term_freq, sublinear_tf=True, smooth_idf=True, norm=norm)
            self.idf_diag = idf_diag
        return doc_term

    def scorer_okapi(self, doc_term_freq, concepts=None, use_existing_data=False, norm='l2'):
        if use_existing_data:
            (doc_term, _, _) = tf_to_okapi(doc_term_freq, idfs=self.idfs, avg_doc_len=self.avg_doc_len)
        else:
            (doc_term, idfs, avg_doc_len) = tf_to_okapi(doc_term_freq)
            self.idfs = idfs
            self.avg_doc_len = avg_doc_len
        return doc_term

    def scorer_midf(self, doc_term_freq, concepts=None, use_existing_data=False, norm='l2'):
        (doc_term_freq_idf, ) = tf_to_midf(doc_term_freq)
        return doc_term_freq_idf

    ##########################
    # Initialization methods #
    ##########################
    def _forward_index_from_directory(self):
        """Build forward index from a directory"""
        scorer_name = self.scorer_name
        traindir = self.traindir
        arff_output = self.traindir_arff
        vocab_output = self.traindir_vocab

        with Timing('Processing training files in the folder %s...' % traindir, self.logging):
            dtf = DocToFeature(lowercase=self.lowercase,
                               keep_nnp=self.keep_nnp,
                               transliteration=self.transliteration,
                               word_normalization=self.word_normalization)
            train_doc_term_freq = dtf.doc_to_tf(traindir)
            train_file_list = dtf.filelist
            vocabulary = dtf.vocabulary
            mapping = dtf.mapping
            concepts = [filename[filename.rfind('/') + 1:].replace('_', ' ').replace('.txt', '')
                        for filename in train_file_list]
        if arff_output is not None:
            with Timing('Dumping TF counts to %s...' % arff_output, self.logging):
                docs_arff = FeatureToArff(train_doc_term_freq, relation='TF.IDF')
                docs_arff.add_column(concepts, name='concept', type_='string')
                docs_arff.dump(arff_output, sparse=True)
            pickle_output = '%s.pickle' % arff_output[:arff_output.rfind('.')]
            with Timing('Pickling TF counts to %s...' % pickle_output, self.logging):
                def task(item, _pickle_output):
                    with open(_pickle_output, 'wb') as outfile:
                        pickle.dump(item, outfile, protocol=2)
                process = mp.Process(target=task, args=((train_doc_term_freq, concepts), pickle_output))
                process.start()
                process.join()
            train_list_output = '%s.list' % arff_output[:arff_output.rfind('.')]
            with Timing('Writing file names of %s into %s...' % (traindir, train_list_output), self.logging):
                with(open(train_list_output, 'w')) as filename_output:
                    for filename in train_file_list:
                        filename_output.write(filename + '\n')
        if vocab_output is not None:
            with Timing('Dumping vocabulary to %s...' % vocab_output, self.logging):
                with open(vocab_output, 'w') as vocab_output_file:
                    pickle.dump(vocabulary, vocab_output_file, protocol=2)
        with Timing('Calculating feature scores using scorer %s...' % scorer_name, self.logging):
            forward_index = self.get_scorer(scorer_name)(train_doc_term_freq)

        self.forward_index_ = forward_index
        self.num_concepts_, self.num_features_ = forward_index.shape
        self.concepts_ = concepts
        self.vocabulary_ = vocabulary
        self.mapping_ = mapping

    def _forward_index_from_arff(self):
        """Build forward index from ARFF file"""
        scorer_name = self.scorer_name
        arff_file = self.trainarff
        vocab_file = self.trainarff_vocab

        with Timing('Loading and processing training data from %s using scorer %s...' % (arff_file, scorer_name),
                    self.logging):
            (train_doc_term_freq, concepts) = loadarff(arff_file)
            pickle_output = '%s.pickle' % arff_file[:arff_file.rfind('.')]
            if not os.path.exists(pickle_output):
                def task(item, _pickle_output):
                    with open(_pickle_output, 'wb') as outfile:
                        pickle.dump(item, outfile, protocol=2)
                process = mp.Process(target=task, args=((train_doc_term_freq, concepts), pickle_output))
                process.start()
                process.join()
            forward_index = self.get_scorer(scorer_name)(train_doc_term_freq)
        with Timing('Loading vocabulary from %s...' % vocab_file, self.logging):
            with open(vocab_file, 'rb') as infile:
                vocabulary = pickle.load(infile)
            mapping = {}
            for word, idx in vocabulary.iteritems():
                mapping[idx] = word

        self.forward_index_ = forward_index
        self.num_concepts_, self.num_features_ = forward_index.shape
        self.concepts_ = concepts
        self.vocabulary_ = vocabulary
        self.mapping_ = mapping

    def _forward_index_from_pickle(self):
        """Build forward index from pickled csr_matrix"""
        scorer_name = self.scorer_name
        pickle_file = self.trainpickle
        vocab_file = self.trainpickle_vocab

        with Timing('Loading and processing training data from %s using scorer %s...' % (pickle_file, scorer_name),
                    self.logging):
            with open(pickle_file, 'rb') as infile:
                (train_doc_term_freq, concepts) = pickle.load(infile)
            forward_index = self.get_scorer(scorer_name)(train_doc_term_freq)
        with Timing('Loading vocabulary from %s...' % vocab_file, self.logging):
            with open(vocab_file, 'rb') as vocab_file:
                vocabulary = pickle.load(vocab_file)
            mapping = {}
            for word, idx in vocabulary.iteritems():
                mapping[idx] = word

        self.forward_index_ = forward_index
        self.num_concepts_, self.num_features_ = forward_index.shape
        self.concepts_ = concepts
        self.vocabulary_ = vocabulary
        self.mapping_ = mapping

    def _invert_index(self):
        """Invert the forward index"""
        forward_index = self.forward_index_
        with Timing('Inverting index... ', self.logging):
            inverted_index = forward_index.transpose(copy=True).tocsr()
            # Remove insignificant term-concept association
            #inverted_index.data[inverted_index.data<=1e-3] = 0
        self.inverted_index_ = inverted_index
        # Word informativeness based on:
        # http://www.ica.stc.sh.cn/picture/article/176/b8/e2/b5e4932249ec8284bb8a86866ec3/3b0d0bff-0e05-4d26-ba6f-85d15924594f.pdf
        # With corrected formula based on the description
        with Timing('Getting IDF scores...', self.logging):
            df = np.diff(inverted_index.indptr)
            idf = np.log(float(self.num_concepts_) / df)
            mul = 1.1
            exp_df = 0.25*np.sqrt(self.num_concepts_)
            fw = mul*abs(np.log(exp_df/df))
            word_info = idf - fw
            word_info = word_info-min(word_info)
            self.word_info_ = word_info/max(word_info)

    def _generate_term_concept_index(self):
        """Generate term concept index"""
        concepts = self.concepts_
        vocabulary = self.vocabulary_
        with Timing('Creating term-concept index...', self.logging):
            dtf = DocToFeature(lowercase=self.lowercase,
                               keep_nnp=self.keep_nnp,
                               transliteration=self.transliteration,
                               word_normalization=self.word_normalization)
            # concept_tf is the term count for each concept name, where each concept name is treated like one document
            concept_tf = dtf.str_to_tf(concepts, vocabulary=vocabulary)
            # concept_term_index is the normalized count
            concept_term_index = self.get_scorer(self.scorer_name)(concept_tf, use_existing_data=True, norm='l2')
            # term_concept_index is the transposed matrix from concept_term_index
            term_concept_index = concept_term_index.transpose(copy=False).tocsr()
        self.term_concept_index_ = term_concept_index

    def initialize(self):
        """Initialize the extractor

        **Notes**

        When initializing from directory (i.e., with traindir specified), please be informed that the training might
        take a very long time, depending on the amount of training data.

        In Knorex working environment, CountVectorizer in scikit-learn has been modified to support multiprocessing,
        and so the initialization process can be faster. It's on the branch "parallel_vectorizer"

        Because of that, whenever the scikit-learn is updated, we need to make sure that the "knx_patch_mpcv"
        branch is still working.
        """
        with Timing('Initializing text processing components...', self.logging):
            self.dtf = DocToFeature(lowercase=self.lowercase,
                                    keep_nnp=self.keep_nnp,
                                    transliteration=self.transliteration,
                                    word_normalization=self.word_normalization)
        with Timing('Initializing ner-tagger component...', self.logging):
            self.ner_tagger = KnorexNERTagger()

        with Timing('Initializing np chunker component...', self.logging):
            self.np_chunker = MaxentNPChunker()

        with Timing('Connect to Wikipedia database...', self.logging):
            self.client = MongoClient('localhost', 27017)
            self.db = self.client['wikipedia']
            self.coll = self.db['TittleId']

        if self.traindir:
            self._forward_index_from_directory()
        elif self.trainarff:
            self._forward_index_from_arff()
        elif self.trainpickle:
            self._forward_index_from_pickle()
        else:
            raise Exception('No training directory or ARFF or pickle file has been specified')

        self._invert_index()
        self._generate_term_concept_index()
        gc.collect()

    def check_initialized(self):
        if self.inverted_index_ is None:
            raise Exception('Inverted index has not been built! Run initialize() first')

    ############################
    # Batch extraction process #
    ############################
    def extract_from_directory(self, dirname, n=10, with_score=False, extraction_output=None):
        """Extract top concepts and top words from the given directory

        **Parameters**

        dirname : string
            The directory containing files that are to be extracted

        n : int, optional, 10 by default
            The number of concepts and words to be extracted

        with_score : boolean, optional, False by default
            Whether to return the score associated with each concept and word

        extraction_output : string, optional, None by default
            The file name to which the extraction output will be printed as JSON dump.

        **Returns**

        extraction_output : list
            The extraction output will always be returned in a list of tuple, where each tuple contains:

                top_concepts : list
                    This will be a list of strings if with_score=False is used,
                    otherwise it will be a list of (concept, score) tuple

                top_phrases : list
                    This will be a list of strings if with_score=False is used,
                    otherwise it will be a list of (phrase, score) tuple
        """
        if extraction_output is None:
            if dirname.find('/') >= 0:
                extraction_output = '%s.out' % dirname[dirname.rfind('/') + 1:]
            else:
                extraction_output = '%s.out' % dirname
        with Timing('Processing test files in the folder %s...' % dirname, self.logging):
            results = []
            for filename in sorted(os.listdir(dirname), key=lambda x: x.lower()):
                if filename == '.DS_Store':
                    continue
                with open(os.path.join(dirname, filename), 'r') as infile:
                    text = infile.read()
                title = filename[:(filename.rfind('.') + len(filename)) % len(filename)]
                result = self.extract(text, title=title, n=n, with_score=with_score)
                results.append((filename, result))
        with Timing('Writing output to %s...' % extraction_output, self.logging):
            with open(extraction_output, 'w') as outfile:
                json.dump(results, outfile)
        return results

    ######################
    # Extraction methods #
    ######################
    def _interpret(self, test_doc_term, test_doc_tf=None, boost_concept=None, boost_lower=None):
        """Convert a term weight matrix into interpretation matrix

        The test_doc_tf is used for concept boosting
        """
        inverted_index = self.inverted_index_
        term_concept_index = self.term_concept_index_
        mapping = self.mapping_
        concepts = self.concepts_
        if boost_concept is None:
            boost_concept = self.boost_concept
        if boost_lower is None:
            boost_lower = self.boost_lower

        with Timing('Calculating interpretation vector...', self.logging):
            interpretation_vector = test_doc_term * inverted_index
        if boost_concept:
            if None in [term_concept_index, test_doc_tf]:
                LOGGER.warn('Concept boosting requested but either term_concept_index or test_docs_tf is not '
                            'available!')
            else:
                # docs_term_index is test_doc_tf being l2-normalized
                with Timing('Calculating term weight scores...', self.logging):
                    docs_term_index = self.get_scorer(self.scorer_name)(test_doc_tf, use_existing_data=True, norm='l2')

                with Timing('Calculating concept multiplier...', self.logging):
                    # Perform concept multiplier calculation for each concept:
                    #     multiplier = 2^(1/sum (w_i.c'_i))
                    # with c'_i = tanh(1/(1-log(c_i)))/tanh(1) as the modified count
                    # where w_i is the weight of word i in concept matrix
                    # and c_i is the normalized count of word i in the document
                    concept_multiplier = docs_term_index * term_concept_index
                    if boost_lower:
                        concept_multiplier.data = self.COTH1 * np.tanh(1 / (1 - np.log(concept_multiplier.data ** 2)))
                    # The -1 is because this multiplier works as an addition to original matrix
                    # So later the multiplication can be done efficiently by using (or the equivalent):
                    #     interpretation_vector += interpretation_vector.multiply(concept_multiplier)
                    concept_multiplier.data = np.exp2(concept_multiplier.data) - 1

                if DEBUG:  # Debug process: print top 10 multipliers
                    docs_term_index_lil = docs_term_index.tolil()
                    top_concept_multiplier_indices = np.argsort(concept_multiplier.getrow(0).toarray()[0])[:-11:-1]
                    concept_multiplier_lil = concept_multiplier.tolil()
                    method_name = 'Core'
                    if boost_concept:
                        method_name = 'Core + Concept boost'
                    if boost_lower:
                        method_name = 'Core + Concept boost with amplification'
                    print 'Multipliers for %s:' % method_name
                    for idx in top_concept_multiplier_indices:
                        concept = concepts[idx]
                        nonzero_indices = term_concept_index.getcol(idx).nonzero()[0]
                        print '%s (%f): ' % (concept, concept_multiplier_lil[0, idx]),
                        for word_idx in nonzero_indices:
                            print '(%s, %f)' % (mapping[word_idx], docs_term_index_lil[0, word_idx]),
                        print
                    print

                with Timing('Multiplying coefficients...', self.logging):
                    interpretation_vector = interpretation_vector + interpretation_vector.multiply(concept_multiplier)
        return interpretation_vector

    def _take_top_phrases(self, interpretation_vector, test_doc_term, candidate_phrases, named_entities=[], n=10,
                          with_score=False, k=25, n_ranked=25, rank_sim='spearman_rank_similarity', text=None,
                          boost_ne = 0.15, max_phrase=0):
        """Return the top n concepts, word, and phrases

        interpretation_vector is expected to be a row vector as csr_matrix
        """
        concepts = self.concepts_
        forward_index = self.forward_index_
        vocabulary = self.vocabulary_
        mapping = self.mapping_
        #num_features = self.num_features_
        #num_concepts = self.num_concepts_
        word_info = self.word_info_
        tokenizer, postprocessor = self.dtf.get_tokenizer(), self.dtf.get_postprocessor()
        tokenized = list(tokenizer(text))

        # Error checking to make sure that we pass the correct variable types
        if not isinstance(interpretation_vector, csr_matrix):
            raise TypeError('Expecting csr_matrix for interpretation_vector, got %s' % type(interpretation_vector))
        if interpretation_vector.shape[0] != 1:
            raise ValueError('Expecting a row vector, found a matrix with %d rows' % interpretation_vector.shape[0])
        if not isinstance(test_doc_term, csr_matrix):
            raise TypeError('Expecting csr_matrix for test_doc_term, got %s' % type(test_doc_term))
        if not isinstance(forward_index, csr_matrix):
            raise TypeError('Expecting csr_matrix for forward_index, got %s' % type(forward_index))

        with Timing('Sorting concepts...', self.logging):
            doc_score = interpretation_vector.toarray()[0]
            sorted_concept_indices = np.argsort(doc_score)
            top_concept_indices = sorted_concept_indices[:-n - 1:-1]
            top_k_indices = sorted_concept_indices[:-k - 1:-1]

        if n_ranked > 0:
            rank_sim_func = getattr(KeyTermsExtractor, rank_sim)
            with Timing('Reranking top concepts...', self.logging):
                top_2k_indices = sorted_concept_indices[:-2 * k - 1:-1]
                # Find the top n_ranked terms in the input document
                word_indices_input = test_doc_term.indices[np.argsort(test_doc_term.data)[:-n_ranked - 1:-1]]
                word_indices_set_input = set(word_indices_input)
                test_doc_term = test_doc_term.tolil()
                concept_to_words = []
                #min_overlap = 2000
                #max_overlap = 0
                #sum_overlap = 0
                for concept_idx in top_2k_indices:
                    concept_vector = forward_index.getrow(concept_idx).tolil()
                    concept_vector_col_idx = np.array(concept_vector.rows[0])
                    concept_vector_data = concept_vector.data[0]
                    # Find the top n_ranked terms in the concept
                    word_indices_concept = concept_vector_col_idx[np.argsort(concept_vector_data)[:-n_ranked - 1:-1]]
                    word_indices_set_concept = set(word_indices_concept)
                    # Combine the top terms in input and in concept
                    word_indices_union = np.array(list(word_indices_set_input | word_indices_set_concept))
                    # Gather overlap statistics for analysis purpose (non-essential)
                    #overlap = len(word_indices_set_concept)+len(word_indices_set_input)-len(word_indices_union)
                    #min_overlap = min(min_overlap, overlap)
                    #max_overlap = max(max_overlap, overlap)
                    #sum_overlap += overlap
                    # Take the scores for each term in the combined list
                    filtered_word_scores_input = test_doc_term[:, word_indices_union].toarray()[0]
                    filtered_word_scores_concept = concept_vector[:, word_indices_union].toarray()[0]
                    # The next four lines to get sorted list of term indices (i.e. the ranking)
                    ranked_word_union_indices_input = np.argsort(filtered_word_scores_input)
                    ranked_word_union_indices_concept = np.argsort(filtered_word_scores_concept)
                    ranked_word_indices_input = word_indices_union[ranked_word_union_indices_input]
                    ranked_word_indices_concept = word_indices_union[ranked_word_union_indices_concept]
                    # The sorted list of term indices are then compared
                    rank_similarity_score = rank_sim_func(ranked_word_indices_input, ranked_word_indices_concept)
                    doc_score[concept_idx] *= rank_similarity_score
                    if DEBUG:
                        words_input = [mapping[idx] for idx in ranked_word_indices_input]
                        words_concept = [mapping[idx] for idx in ranked_word_indices_concept]
                        concept_to_words.append([concept_idx, concepts[concept_idx], words_input, words_concept,
                                                 rank_similarity_score, doc_score[concept_idx],
                                                 doc_score[concept_idx] * rank_similarity_score])
                if DEBUG:
                    from pprint import pprint
                    pprint(concept_to_words)
                sorted_concept_indices = top_2k_indices[np.argsort(doc_score[top_2k_indices])]
                top_concept_indices = list(sorted_concept_indices[:-n - 1:-1])
                k = len(sorted_concept_indices >= 1)
                top_k_indices = list(sorted_concept_indices[:-k - 1:-1])
            #LOGGER.debug('Min overlaps: %d\nMax overlaps: %d\nAvg overlaps: %.2f' %
            #              (min_overlap, max_overlap, sum_overlap/(2.0*k)))

        with Timing('Sorting terms...', self.logging):
            # This part is explained in https://wiki.knorex.asia/x/rAMYBQ the "Key Words Extraction Part" section
            # Take top k concepts score from the interpretation vector (shape: 1 x k)
            top_k_concepts = csr_matrix(doc_score[top_k_indices])
            # Multiply each term in the top k concept vectors by term weight in the input text (shape: k x |V|)
            concept_word_matrix = forward_index[top_k_indices, :].multiply(scipy.sparse.vstack([test_doc_term] * k))
            # Find the maximum term score in each concept (shape: 1 x k)
            padded_data = np.pad(concept_word_matrix.data, (0, 1), 'constant', constant_values=0)
            scale = csr_matrix(np.maximum.reduceat(padded_data, concept_word_matrix.indptr[:-1]))
            # Find the normalizing constant for the top k concepts of the interpretation vector, multiply it to scale
            # Now scale contains the normalizing constant multiplied by the maximum term score in each concept
            scale = scale * top_k_concepts.sum(axis=1)[0, 0]
            # Invert the scale so that later division is just a multiplication with this scale
            scale.data = 1 / scale.data
            # Normalize the interpretation vector as well as divide each with the maximum term score of each concept
            # This completes step 3 (normalizing interpretation vector top_k_concepts) and prepare for step 2
            scale = scale.multiply(top_k_concepts)
            # When scale is multiplied (matrix multiplication) with the top k concept vectors, we are doing
            # step 2 and 4 simultaneously, resulting in a 1 x |V| vector containing the desired term score
            word_affinity_vector = scale * concept_word_matrix
            word_affinity_vector = word_affinity_vector.toarray()[0]
            top_terms_indices = [i for i in np.argsort(word_affinity_vector)[:-n - 1:-1]
                                 if word_affinity_vector[i] > 0]
            # top_terms_indices = []

        def WordLength(tokens):
            punct = '.,()&[]\'\"-/\\\n'
            res = 0
            for x in tokens:
                if x not in punct:
                    res += 1
            return res

        def dist(word, tokenized):
            pos = tokenized.index(word)
            return 0.5 - 1.0/(len(tokenized) - pos + 1)

        def calcMaxPhrases(lDoc):
            nPhrases = len(candidate_phrases)
            if nPhrases < 3:
                return nPhrases
            if nPhrases < 19:
                return nPhrases/3 + 2
            return int(round(nPhrases/math.log(nPhrases))) + 1
            # tokset = set()
            # for phrase in candidate_phrases:
            #     for x in tokenizer(phrase):
            #         tokset.add(postprocessor(x))
            # nToks = len(tokset)
            # return (nPhrases + nToks)/10, (nPhrases + nToks + lDoc)/20

        shortThres = 250


        #Update word_affinity_vector for short text
        lDoc = WordLength(tokenized)
        if lDoc < shortThres:
            tokenized = map(postprocessor, tokenized)
            tokenized = [token for token in tokenized if token in vocabulary]
            for word in set(tokenized):
                wid = vocabulary[word]
                word_affinity_vector[wid] = word_info[wid]*dist(word, tokenized)

        def ILPSolver(named_entities=[], regularizer=0.00, max_phrase=15, min_len=0.0, TOL=0.00001, w_ne=2.0, 
                      postprocessor=postprocessor, tokenizer=tokenizer):
            def build_word_list(token_phrases):
                res = []
                for x in token_phrases:
                    res.extend(x)
                return list(set(res))

            def build_substr(token_phrases):
                def cal_prefix_score(l1, l2):
                    len_l1, len_l2, res = len(l1), len(l2), 0.0
                    for i, x in enumerate(l1):
                        if i == len_l2:
                            break
                        if x == l2[i]:
                            res += 1.0
                        else:
                            break
                    return res/len_l1

                def cal_suffix_score(t1, t2):
                    l1, l2 = list(reversed(t1)), list(reversed(t2))
                    len_l1, len_l2, res = len(l1), len(l2), 0.0
                    for i, x in enumerate(l1):
                        if i == len_l2:
                            break
                        if x == l2[i]:
                            res += 1.0
                        else:
                            break
                    return res/len_l1

                res = []
                for x1, ls1 in enumerate(token_phrases):
                    count = 0.0
                    s1 = ' '.join(ls1)
                    for x2, ls2 in enumerate(token_phrases):
                        if x1 != x2:
                            s2 = ' '.join(ls2)
                            if s2.find(s1) != -1 and len(s2) != 0:
                                count += float(len(ls1))/len(ls2)
                            elif s1.find(s2) == -1 and len(s1)!= 0:
                                count += cal_suffix_score(ls1, ls2)
                                count += cal_prefix_score(ls1, ls2)
                    res.append(count)
                return res

            def build_ne_reg(phrases_list, named_entities):
                res = []
                for phrase in phrases_list:
                    tmp = 0.0
                    for ne in named_entities:
                        if phrase.find(ne) != -1:
                            tmp = w_ne
                            break
                    res.append(tmp)
                return res

            def build_occ_termphrase(TERMS, PHRASES):
                res = dict()
                for id_terms in TERMS:
                    tmp = []
                    for id_phrase in PHRASES:
                        if occ(id_terms, id_phrase) == 1:
                            tmp.append(id_phrase)
                    res[id_terms] = tmp
                return res

            def occ(id_term, id_phrase):
                # term, phrase = mapping[id_term], token_phrases[id_phrase]
                term, phrase = word_map(id_term), token_phrases[id_phrase]
                if term in phrase:
                    return 1
                return 0

            def length_phrase(id_phrase):
                tokens = token_phrases[id_phrase]
                return len(tokens)

            def cal_phrase_score(id_phrase):
                score = 0.00
                for word in token_phrases[id_phrase]:
                    wid = word_index(word)
                    if wid == -1:
                        continue
                    if term_vars[wid].varValue > TOL:
                        score += word_score(wid)
                score /= length_phrase(id_phrase)
                return abs(score - regularizer*(length_phrase(j) - min_len)/(1.0 + substr[j] + ne_reg[j]))

            def phrase_tokenize(phrase, tokenizer=None):
                if tokenizer:
                    res = [x.strip('.,()&[]\'\"-/\\\n ') for x in tokenizer(phrase)]
                else:
                    res = [x.strip('.,()&[]\'\"-/\\\n ') for x in phrase.split()]
                res = [x.replace(u'\n', u' ') for x in res if len(x) > 0]
                return [postprocessor(x) for x in res]

            def word_score(wordIdx):
                if wordIdx >= 0:
                    return word_affinity_vector[wordIdx]
                else:
                    return ne_score[wordIdx]
            
            def word_index(word):
                if word in vocabulary:
                    return vocabulary[word]
                if word in ne_vocab:
                    return ne_vocab[word]
                return -1

            def word_map(wid):
                if wid >= 0:
                    return mapping[wid]
                if wid < -1:
                    return ne_mapping[wid]

            def build_ne_word_score(named_entities, tokenizer=None):
                neVocab, neMap, neScore = {}, {}, {}
                idx = -2
                for named_entity in named_entities:
                    tokens = phrase_tokenize(named_entity, tokenizer)
                    boostScore = boost_ne*1.0/len(tokens)
                    for token in tokens:
                        if token not in vocabulary:
                            if token not in neVocab:
                                neVocab[token] = idx
                                neMap[idx] = token
                                neScore[idx] = boostScore
                                idx -= 1
                            elif neScore[neVocab[token]] < boostScore:
                                neScore[neVocab[token]] = boostScore
                        # elif word_affinity_vector[vocabulary[token]] < TOL:
                        #     word_affinity_vector[vocabulary[token]] = boostScore
                return neVocab, neMap, neScore

            phrases_list = list(candidate_phrases)
            token_phrases = [phrase_tokenize(x, tokenizer) for x in phrases_list]
            word_list = build_word_list(token_phrases)
            substr = build_substr(token_phrases)            
            ne_reg = build_ne_reg(phrases_list, named_entities)

            ne_vocab, ne_mapping, ne_score = build_ne_word_score(named_entities, tokenizer)
            TERMS = [word_index(word) for word in word_list if word_index(word) != -1]
            # TERMS = [vocabulary[word] for word in word_list if word in vocabulary] #word_id_list
            PHRASES = range(len(phrases_list))

            prob = LpProblem("TermPhrase", LpMaximize)
            term_vars = LpVariable.dicts("UseTerm", TERMS, 0, 1, LpBinary)
            phrase_vars = LpVariable.dicts("UsePhrase", PHRASES, 0, 1, LpBinary)
            # prob += lpSum(term_vars[i]*word_affinity_vector[i] for i in TERMS) \
            prob += lpSum(term_vars[i]*word_score(i) for i in TERMS) \
                    - regularizer*lpSum(phrase_vars[j]*(length_phrase(j) - min_len)/(1.0 + substr[j] + ne_reg[j]) 
                                            for j in PHRASES)
            prob += lpSum(phrase_vars[j] for j in PHRASES) <= max_phrase
            for j in PHRASES:
                for i in TERMS:
                    if occ(i,j) == 1:
                        prob += phrase_vars[j] <= term_vars[i]
            occ_termphrase = build_occ_termphrase(TERMS, PHRASES)
            for i in TERMS:
                prob += lpSum(phrase_vars[j] for j in occ_termphrase[i]) >= term_vars[i]
            prob.solve()
            
            # top_terms = [(mapping[i], word_affinity_vector[i]) for i in TERMS if term_vars[i].varValue > TOL] 
            top_terms = [(word_map(i), word_score(i)) for i in TERMS if term_vars[i].varValue > TOL] 
            top_phrases = [(phrases_list[j], cal_phrase_score(j)) for j in PHRASES if phrase_vars[j].varValue > TOL] 
            return (sorted(top_terms, key=lambda x: x[1], reverse=True), 
                        sorted(top_phrases, key=lambda x: x[1], reverse=True))

        if max_phrase == 0:
            max_phrase = calcMaxPhrases(lDoc)

        with Timing('Solving ILP problem...', self.logging):
            top_terms, top_phrases = ILPSolver(named_entities=named_entities, regularizer=0.01, max_phrase=max_phrase)
        
        if with_score:
            top_concepts = [(concepts[i], doc_score[i]) for i in top_concept_indices]
        else:
            top_concepts = [concepts[i] for i in top_concept_indices]
            top_terms = [x[0] for x in top_terms]
            top_phrases = [x[0] for x in top_phrases]

        if len(top_terms) > n:
            top_terms = top_terms[:n]    
        return (top_concepts, top_terms, top_phrases)

    def extract_candidate_phrases(self, preprocessed_data, np_chunker=None, lemmatizer=None, named_entities=[]):
        """Returns list of phrases suitable as key words from the text.

        **Parameters**

        preprocessed_data : str
            The text which has been preprocessed (e.g., through get_preprocessor())

        np_chunker : knx.text.chunker.NPChunker
            The noun phrase chunker that will be used to find noun phrases in the text

        lemmatizer : knx.text.doc_to_feature.Lemmatizer
            The lemmatizer that will be used to determine whether two noun phrases are the same

        **Returns**

        phrases : list of tuple: phrase, start index, end index in the orginal text
            The return value will just be list of noun phrases as string
        """
        def adjust_postags(chunked_phrases):
            def alnum(word):
                return not word.isalpha() and not word.isdigit() and word.isalnum()

            for phrase in chunked_phrases:
                for idx, word_pos in enumerate(phrase):
                    word, pos = word_pos
                    if pos == 'CD' and alnum(word):
                        phrase[idx] = (word, 'JJ')
            return chunked_phrases

        def join_phrase(tokens):
            res, prev = str(), str()
            for token in tokens:
                if token == '$' and len(prev) == 2 and prev.isupper():
                    res += token
                elif prev == '$' and len(token) > 0 and token[0].isdigit():
                    res += token
                elif token == '%' and len(prev) > 0 and prev[0].isdigit():
                    res += token
                elif token == ',':
                    res += token
                elif token == '&' and len(prev) <= 2:
                    res += token
                elif prev == '&' and len(token) <= 2:
                    res += token
                else:
                    res += (' ' + token if len(res) > 0 else token)
                prev = token
            return res

        def find_phrases(chunked_phrases):
            """
                Find the position of the phrases in the orginal text (preprocessed_data)
            """
            res = []
            prev = 0
            for chunked_phrase in chunked_phrases:
                phrase = join_phrase(zip(*chunked_phrase)[0])
                # phrase = ' '.join(zip(*chunked_phrase)[0])
                sid = preprocessed_data.find(phrase, prev)
                if sid != -1:
                    prev = eid = sid + len(phrase)
                else:
                    eid, prev = -1, prev + 1
                tmp = (chunked_phrase, sid, eid)
                res.append(tmp)
            return res
        
        def find_ne(position, ne_type=str()):
            for se, ee, named_entity, ne_type1 in named_entities:
                if se <= position < ee and (ne_type == ne_type1 or not ne_type):
                    return (se, ee, named_entity, ne_type1)
            return tuple()

        def contain_CC(chunked_phrase):
            for i, word_pos in enumerate(chunked_phrase):
                if word_pos[1] == 'CC' and word_pos[0] != '&':
                    return i
            return -1

        def countAND(chunked_phrase):
            res = []
            for idx, word_pos in enumerate(chunked_phrase):
                if word_pos[0] == 'and':
                    res.append(idx)
            return res

        def checkDT(phrase):
            chunked_phrase, sp, ep = phrase
            DTWords = {u'the', u'that', u'those', u'these', u'this'}
            if len(chunked_phrase) != 2 or chunked_phrase[0][1] != 'DT':
                return True 
            if chunked_phrase[0][0].lower() not in DTWords or chunked_phrase[1][1] not in {'NNS', 'NN'}:
                return True
            return False
            
        if np_chunker is None:
            np_chunker = self.np_chunker
        if lemmatizer is None:
            lemmatizer = self.dtf.lemmatizer
        chunked_phrases = np_chunker.chunk(preprocessed_data, sent_tokenized=False, output_tags=True, split_words=True)
        chunked_phrases = adjust_postags(chunked_phrases)
        chunked_phrases = find_phrases(chunked_phrases)


        def NPString(np, with_pos_tag=True):
            if with_pos_tag:
                return ' '.join([x[0] + u'/' + x[1] for x in np])
            else:
                return ' '.join(zip(*np)[0])


        #Remove meaningless words if they are not included in any named entities
        MEANINGLESS_WORDS = {u'other', u'others', u'such', u'many', u'any', u'etc', u'e.g', u'much',u'someone',
                             u'anyone', u'someelse', u'anything', u'something', u'nothing', u'everyone', u'everything'}
        new_chunked_phrases = []
        for chunked_phrase, sp, ep in chunked_phrases:
            if sp == -1: 
                continue
            tmp = chunked_phrase[:]
            chunked_phrase[:] = []
            reduce_id = []
            for idx, word_pos in enumerate(tmp):
                word, pos = word_pos
                if word.lower() not in MEANINGLESS_WORDS or find_ne(sp):
                    chunked_phrase.append((word,pos))
                else:
                    reduce_id.append(idx)
            sp, ep = update_seid(tmp, reduce_id, sp, ep)
            if chunked_phrase:
                new_chunked_phrases.append((chunked_phrase, sp, ep))
        chunked_phrases = new_chunked_phrases

        # Eliminate some phrases with pattern: "the/that/those/these/this + NN/NNS"
        chunked_phrases = filter(checkDT,chunked_phrases)

        # Adjust phrases ended with &/CC
        tmp = chunked_phrases
        chunked_phrases = []
        lphrases, flag = len(tmp), True
        for i, phrase in enumerate(tmp):
            chunked_phrase, sp, ep = phrase
            if not flag or not chunked_phrase:
                flag = True
                continue
            if chunked_phrase[-1] == ('&', 'CC') and i + 1 < lphrases:
                nchunked_phrase, nsp, nep = tmp[i + 1]
                if nchunked_phrase[0][1] == 'NNP' and ep == nsp:
                    chunked_phrase.extend(nchunked_phrase)
                    ep = nep
                    flag = False
            chunked_phrases.append((chunked_phrase, sp, ep))

        # Seperate phrases containing two or more conjunctions
        tmp = chunked_phrases
        chunked_phrases = []
        for chunked_phrase, sp, ep in tmp:
            if len(countAND(chunked_phrase)) > 0:
                prev, nsp, nep = 0, sp, sp
                for i, word_pos in enumerate(chunked_phrase):
                    word, pos = word_pos
                    if word == 'AND' and not find_ne(nep) and i - prev > 0:
                        chunked_phrases.append((chunked_phrase[prev:i], nsp, nep - 1))
                        nsp = nep + len(word) + 1
                        prev = i + 1
                    nep += len(word) + 1
            else:
                chunked_phrases.append((chunked_phrase, sp, ep))

        # Filter and modify the noun phrases for suitable KTE outputs
        chunked_phrases = filter_noun_phrases(chunked_phrases, lemmatizer)

        # Remove percentage named entities
        tmp = chunked_phrases
        chunked_phrases = []
        for chunked_phrase, sp, ep in tmp:
            ne = find_ne(sp, "Percentage")
            if ne and ne[0] <= sp < ne[1]:
                flag = False
                for idx, word_pos in enumerate(chunked_phrase):
                    sp += len(word_pos[0]) + 1
                    if sp >= ne[1]:
                        flag = True
                        break
                if flag:
                    if idx != len(chunked_phrase) - 1:
                        chunked_phrases.append((chunked_phrase[idx + 1:], sp, ep))
                else:
                    chunked_phrases.append((chunked_phrase, sp, ep))
            else:
                chunked_phrases.append((chunked_phrase, sp, ep))

        # Seperate some phrases containing conjunction
        tmp = chunked_phrases
        chunked_phrases = []
        for chunked_phrase, sp, ep in tmp:
            if sp == -1:
                continue
            idx = contain_CC(chunked_phrase)
            flag = True
            if idx != -1 and idx != len(chunked_phrase):
                if not find_ne(sp):
                    pos_str = ' '.join(zip(*chunked_phrase)[1])
                    if pos_str in {'NNP CC NNP', 'NN CC NN', 'NNS CC NNS', 'NNPS CC NNPS'}:
                        flag = False
                        chunked_phrases.append(([chunked_phrase[0]], sp, sp + len(chunked_phrase[0][0])))
                        chunked_phrases.append(([chunked_phrase[2]], ep - len(chunked_phrase[2][0]), ep))
                else:
                    sp2 = sp
                    for i in range(idx + 1):
                        sp2 += len(chunked_phrase[i][0]) + 1
                    spcc = sp2 - len(chunked_phrase[idx][0]) - 1
                    if find_ne(sp2) and find_ne(ep - 1) and not find_ne(spcc):
                        flag = False
                        chunked_phrases.append((chunked_phrase[0:idx], sp, spcc - 1))
                        chunked_phrases.append((chunked_phrase[idx+1:], sp2, ep))
            if flag:
                chunked_phrases.append((chunked_phrase, sp, ep))        

        phrases = []
        for chunked_phrase, sp, ep in chunked_phrases:
            if not chunked_phrase:
                continue
            phrase = join_phrase(zip(*chunked_phrase)[0])
            # phrase = ' '.join(zip(*chunked_phrase)[0])
            phrases.append((phrase, sp, ep))
        return phrases

    def get_first_paragraph(self, preprocessed_data):
        """Returns the (approximately) first paragraph from the preprocessed data based on newline and period
        """
        first_idx = preprocessed_data.find('\n', 25)
        if first_idx == -1 or first_idx > 500:
            first_idx = preprocessed_data.find('. ', 75) + 1
        if first_idx == -1:
            return ''
        if '.' not in preprocessed_data[first_idx - 4:first_idx]:
            tmp = preprocessed_data.find('.', first_idx)
            if not (tmp == -1):
                first_idx = tmp
        return preprocessed_data[:first_idx]

    def extract(self, data, title='', n=10, with_score=False, k=25, n_ranked=25, boost_concept=None, boost_lower=None,
                rank_sim='spearman_rank_similarity', return_values=['concepts', 'terms', 'phrases'],
                ner_tagger=None, np_chunker=None, lemmatizer=None, boost_ne=0.15):
        """Extract top concepts and phrases from a document string

        **Parameters**

        data : string
            The document from which concepts and phrases should be extracted

        title : string
            The title of the document. This is used to extract important information often found in titles

        n : int, optional, 10 by default
            The number of top concepts and phrases to be extracted

        with_score : boolean, optional, False by default
            Whether to include the scores for each concept and phrase

        k : int, optional, 25 by default
            The parameter that controls the number of concepts should affect the phrase scoring

        n_ranked : int, optional, 25 by default
            The parameter that controls how many top words in each concept will be considered when reranking

        boost_concept : boolean, optional
            Whether to boost concept scores.
            Will use the value according to the boost_method property if not provided

        boost_lower : boolean, optional
            Whether to amplify lower scores, only applicable if boost_concept is True
            Will use the value according to the boost_method property if not provided

        rank_sim : str, one of {'gk_rank_similarity', 'spearman_rank_similarity'}, default to 'spearman_rank_similarity'
            The rank similarity measure to rerank the concepts

        return_values : collection of str in {'concepts', 'terms', 'phrases'}
            The values that will be returned.

                concepts : list of top concepts
                terms    : list of top terms
                phrases  : list of top phrases

        ner_tagger : a ner_tagger object, optional
            An object with the method `processDocument` that accepts a string and produce an XML

        np_chunker : an NPChunker object, optional
            knx.text.chunker.NPChunker object to do the noun phrase chunking

        lemmatizer : a Lemmatizer object, optional
            An object with the method `lemmatize` that accepts a word and an optional POS tag,
            and produces the lemma of the specified word

        **Returns**

        top_values : dict
            The returned value will be a dictionary containing any combination of these mappings, depending on the
            `return_values` argument:

                'concepts' -> top_concepts : list
                    This will be a list of concepts if with_score=False is used,
                    otherwise it will be a list of (concept, score) tuple

                'terms' -> top_terms : list
                    This will be a list of terms if with_score=False is used,
                    otherwise it will be a list of (term, score) tuple

                'phrases' -> top_phrases : list
                    This will be a list of phrases if with_score=False is used,
                    otherwise it will be a list of (phrase, score) tuple
        """
        def match_ne(phrase, named_entity):
            if ((phrase in named_entity or named_entity in phrase)
                                and abs(len(phrase) - len(named_entity)) < 3):
                return True
            else:
                return False
        
        if ner_tagger is None:
            ner_tagger = self.ner_tagger
        if np_chunker is None:
            np_chunker = self.np_chunker
        if lemmatizer is None:
            lemmatizer = self.dtf.lemmatizer

        if boost_concept is None:
            boost_concept = self.boost_concept
        if boost_lower is None:
            boost_lower = self.boost_lower
        
        with Timing('Converting input into matrix...', self.logging):
            # vector_tf = self.dtf.str_to_tf(data.replace('-',' '), self.vocabulary_)
            vector_tf = self.dtf.str_to_tf(data, self.vocabulary_)
            # LOGGER.info('%s',vector_tf)

        with Timing('Preprocessing the input text...', self.logging):
            preprocessed = self.dtf.get_preprocessor()(data)
            # LOGGER.info('%s',preprocessed)

        with Timing('Boosting title and first paragraph...', self.logging):
            # first_para = self.get_first_paragraph(preprocessed)
            # summary = title + ' . ' + first_para
            summary = title + ' . ' + preprocessed
            for word in self.dtf.get_analyzer()(title):
                word_idx = np.argwhere(vector_tf.indices == self.vocabulary_.get(word, None))
                if word_idx:
                    vector_tf.data[word_idx] = vector_tf.data[word_idx] * 2 + 1

        with Timing('Tagging named entities...', self.logging):
            banned_type = {u'Currency',u'Percentage', u'Date', u'Time', u'Position', u'Malaysian_title'}
            if ner_tagger is not None:
                bias = len(title + ' . ')
                nes, tmpdupnes = ner_tagger.tag(summary)
                full_named_entities = [(x[0] - bias, x[1] - bias, x[2], x[3]) for x in nes]
                dupnes = []
                for y in tmpdupnes:
                    tmp = [(x[0] - bias, x[1] - bias, x[2], x[3]) for x in y]
                    dupnes.append(tmp)
                named_entities = [x[2] for x in full_named_entities if x[3] not in banned_type]
                named_entities = list(set(named_entities))
                
                banned_named_entities = [x[2] for x in full_named_entities if x[3] in banned_type]
                banned_named_entities = list(set(banned_named_entities))
            else:
                named_entities = banned_named_entities = full_named_entities = []
                if self.logging:
                    LOGGER.warn('Not running ner tagger')

        with Timing('Extracting candidate phrases...', self.logging):
            candidate_phrases = self.extract_candidate_phrases(preprocessed, np_chunker, lemmatizer,
                                                               full_named_entities)
            # LOGGER.info('%s',candidate_phrases)

        with Timing('Processing noun phrases with named entities...', self.logging):
            #Correct noun phrases containing incompleted named entities
            tmp = candidate_phrases[:]
            candidate_phrases = []
            for phrase, sp, ep in tmp:
                found = False
                for se, ee, named_entity, ne_type in full_named_entities:
                    if se < sp < ee < ep:
                        found = True
                        new_phrase = preprocessed[se:sp] + phrase 
                        candidate_phrases.append((new_phrase, se, ep))
                        break
                    if sp < se < ep < ee:
                        found = True
                        new_phrase = phrase + preprocessed[ep:ee]
                        candidate_phrases.append((new_phrase, sp, ee))
                        break
                    if se <= sp < ep <= ee:
                        found = True
                        candidate_phrases.append((named_entity, se, ee))
                        break
                if not found:
                    candidate_phrases.append((phrase, sp, ep))
            #Eliminating some noun phrases (Percentage, Currency ...) standing alone
            tmp = candidate_phrases[:]
            candidate_phrases = []
            for phrase in tmp:
                found = False
                for named_entity in banned_named_entities:
                    if match_ne(phrase[0], named_entity):
                        found = True
                        break
                if not found:
                    candidate_phrases.append(phrase)
            #Adding some named entities not belong to any phrases
            for named_entity in named_entities:
                found = True
                for phrase, sp, ep in candidate_phrases:
                    if named_entity in phrase:
                        found = True
                        break
                if not found:
                    candidate_phrases.add((named_entity, se, ee))

        with Timing('Eliminate nearly duplicated noun phrases based on morphological features...', self.logging):
            def compare(phrase1, phrase2):
                def compare1(token1, token2):
                    t1, t2 = token1.lower(), token2.lower()
                    if t1[-1] in 's.,' and t1[0:-1] == t2:  return 1
                    if t1[-2:] == 'es':
                        if t1[:-2] == t2:  return 1
                        if t1[:-3] == t2[:-1] and t1[-3] == 'i' and t2[-1] == 'y':  return 1
                    
                    if t2[-1] in 's.,' and t2[0:-1] == t1:  return -1
                    if t2[-2:] == 'es':
                        if t2[:-2] == t1:  return -1
                        if t2[:-3] == t1[:-1] and t2[-3] == 'i' and t1[-1] == 'y':  return -1

                    if t1 != t2:
                        return -2
                    return 0
                
                def compare2(token1, token2):
                    if token1 == token2:
                        return 0
                    if token1[0].isupper() and token2[0].islower():
                        return 1
                    if token2[0].isupper() and token1[0].islower():
                        return -1
                    if token1.isupper() and (token2.islower() or token2.istitle()):
                        return 1
                    if token2.isupper() and (token1.islower() or token1.istitle()):
                        return -1
                    return 0
                
                lphrase1, lphrase2 = phrase1.split(), phrase2.split()
                if len(lphrase1) != len(lphrase2):
                    return 0
                s1 = s2 = 0 
                for i in range(len(lphrase1)):
                    cmp1 = compare1(lphrase1[i], lphrase2[i])
                    if cmp1 == -2:
                        return 0
                    cmp2 = compare2(lphrase1[i], lphrase2[i])
                    s1 += cmp1
                    s2 += cmp2
                if s1 < 0:
                    return -1
                elif s1 > 0:
                    return 1
                if s2 < 0:
                    return -1
                elif s2 > 0:
                    return 1
                return 0
            
            flag = [True]*len(candidate_phrases)
            for id1, phrase1 in enumerate(candidate_phrases):
                if not flag[id1]:
                    continue
                for id2, phrase2 in enumerate(candidate_phrases):
                    if id1 >= id2:
                        continue
                    t_cmp = compare(phrase1[0], phrase2[0])
                    if t_cmp == -1:
                        flag[id1] = False
                        break
                    elif t_cmp == 1:
                        flag[id2] = False
            candidate_phrases = [x for i, x in enumerate(candidate_phrases) if flag[i]]

        with Timing('Eliminate duplicated named enitites...', self.logging):            
            def get_ne_id(phrases):
                def find_ne(sid, named_entities):
                    for idx, ne in enumerate(named_entities):
                        se, ee, named_entity, ne_type = ne
                        if se <= sid < ee:
                            return idx
                    return -1
                
                res = dict()
                for phrase, sid, eid in phrases:
                    flag = True
                    for lidx, named_entities in enumerate(dupnes):
                        idx = find_ne(sid, named_entities)
                        if idx != -1:
                            res[phrase, sid, eid] = (lidx, idx)
                            flag = False
                            break
                    if flag:
                        res[phrase, sid, eid] = (-1, -1)
                return res

            def compare_ne(phrase1, phrase2):
                p1, sid1, eid1 = phrase1
                p2, sid2, eid2 = phrase2
                if sid1 == -1 or sid2 == -1:
                    return 0
                neidx1, neidx2 = ne_list_id[phrase1], ne_list_id[phrase2]
                if neidx1[0] != -1:
                    ne1 = dupnes[neidx1[0]][neidx1[1]]
                    if neidx2[0] == -1:
                        return 0
                    ne2 = dupnes[neidx2[0]][neidx2[1]]
                    if ne1[2] == ne2[2]:
                        return 0
                    sidne1, eidne1 = ne1[0] - sid1, ne1[1] - sid1
                    sidne2, eidne2 = ne2[0] - sid2, ne2[1] - sid2
                    if p1[0:sidne1] == p2[0:sidne2] and p1[eidne1:] == p2[eidne2:]:
                        if len(p1) > len(p2):
                            return 1
                        elif len(p1) < len(p2):
                            return -1
                return 0
            
            ne_list_id = get_ne_id(candidate_phrases)
            flag = [True]*len(candidate_phrases)
            for id1, phrase1 in enumerate(candidate_phrases):
                if not flag[id1]:
                    continue
                for id2, phrase2 in enumerate(candidate_phrases):
                    if id1 >= id2:
                        continue
                    cmp_ne = compare_ne(phrase1, phrase2)
                    if cmp_ne == -1:
                        flag[id1] = False
                        break
                    elif cmp_ne == 1:
                        flag[id2] = False
            candidate_phrases = [x for i, x in enumerate(candidate_phrases) if flag[i]]

        with Timing('Eliminate based on Wikipedia direct machanism...', self.logging):
            #Eliminating duplicated noun phrases by creating a set of phrases (set of strings)
            tmp = set()
            for phrase, sid, eid in candidate_phrases:
                tmp.add(phrase)

            def get_wiki_id(candidate_phrases):
                res = dict()
                for s in candidate_phrases:
                    s1 = s.upper().replace(u' ',u'_')
                    f = self.coll.find_one({'title': s1}, {'_id': False, 'title': False})
                    if f == None:
                        res[s] = unicode()
                    else:
                        res[s] = f[u'id']
                return res

            def compare_wiki_direct(phrase1, phrase2):
                wiki_id1, wiki_id2 = wiki_id[phrase1], wiki_id[phrase2]
                if wiki_id1 == str() or wiki_id2 == str():
                    return 0
                if wiki_id1 == wiki_id2:
                    if len(phrase1) > len(phrase2):
                        return 1
                    elif len(phrase1) < len(phrase2):
                        return -1
                return 0
            
            wiki_id = get_wiki_id(tmp)        
            flag = [True]*len(tmp)
            for id1, phrase1 in enumerate(tmp):
                if not flag[id1]:
                    continue
                for id2, phrase2 in enumerate(tmp):
                    if id1 >= id2:
                        continue
                    cmp_wiki = compare_wiki_direct(phrase1, phrase2)
                    if  cmp_wiki == -1:
                        flag[id1] = False
                        break
                    elif cmp_wiki == 1:
                        flag[id2] = False
            candidate_phrases = [x for i, x in enumerate(tmp) if flag[i]]
            candidate_phrases = [x for x in candidate_phrases if len(x.split()) <= 9]

        with Timing('Scoring each term in the matrix and calculating interpretation vector...', self.logging):
            vector = self.get_scorer(self.scorer_name)(vector_tf, use_existing_data=True)
            interpretation_vect = self._interpret(vector, test_doc_tf=vector_tf,
                                                  boost_concept=boost_concept, boost_lower=boost_lower)
        
        if DEBUG:
            with Timing('Drawing data graph...', self.logging):
                draw_vector(interpretation_vect)

        top_concepts, top_terms, top_phrases = self._take_top_phrases(interpretation_vect,
                                                                     test_doc_term=vector,
                                                                     candidate_phrases=candidate_phrases,
                                                                     named_entities=named_entities,
                                                                     n=n, with_score=with_score, 
                                                                     k=k, n_ranked=n_ranked,
                                                                     rank_sim=rank_sim, text=preprocessed,
                                                                     boost_ne=boost_ne)
        result = dict()
        if 'concepts' in return_values:
            result['concepts'] = top_concepts
        if 'terms' in return_values:
            result['terms'] = top_terms
        if 'phrases' in return_values:
            result['phrases'] = top_phrases
        return result

    def extract_batch(self, doc_list, title='', n=10, with_score=False, k=25, n_ranked=25,
                      return_values=['concepts', 'terms', 'phrases'],
                      boost_concept=None, boost_lower=None, rank_sim='spearman_rank_similarity'):
        """Extract top concepts, terms, or phrases from a list of documents

        **Parameters**

        doc_list : list
            The list of strings from which concepts and phrases should be extracted

        title : string
            The title of the document. This is used to extract important information often found in titles

        n : int, optional, 10 by default
            The number of top concepts and phrases to be extracted

        with_score : boolean, optional, False by default
            Whether to include the scores for each concept and phrase

        k : int, optional, 25 by default
            The parameter that controls the number of concepts should affect the phrase scoring

        n_ranked : int, optional, 25 by default
            The parameter that controls how many top words in each concept will be considered when reranking

        return_values : collection of str in {'concepts', 'terms', 'phrases'}
            The values that will be returned.

                concepts : list of top concepts
                terms    : list of top terms
                phrases  : list of top phrases

        boost_concept : boolean, optional
            Whether to boost concept scores.
            Will use the value according to the boost_method property if not provided

        boost_lower : boolean, optional
            Whether to amplify lower scores, only applicable if boost_concept is True
            Will use the value according to the boost_method property if not provided

        **Returns**

        result : list of dict
            The returned value will be a list of dictionaries, where each tuple is the extraction result from one
            document, in the same order as given.
            Each dictionary will contain any combination of these mappings, depending on the `return_values` argument:

                'concepts' -> top_concepts : list
                    This will be a list of concepts if with_score=False is used,
                    otherwise it will be a list of (concept, score) tuple

                'terms' -> top_terms : list
                    This will be a list of terms if with_score=False is used,
                    otherwise it will be a list of (term, score) tuple

                'phrases' -> top_phrases : list
                    This will be a list of phrases if with_score=False is used,
                    otherwise it will be a list of (phrase, score) tuple
        """
        old_logging = self.logging
        self.logging = False
        with Timing('Extracting top concepts and phrases from list of texts...', old_logging):
            result = [self.extract(data, title=title, n=n, with_score=with_score, k=k, n_ranked=n_ranked,
                                   return_values=return_values, boost_concept=boost_concept, boost_lower=boost_lower,
                                   rank_sim=rank_sim) for data in doc_list]
        self.logging = old_logging
        return result

    #####################
    # General utilities #
    #####################
    def start_server(self, basedir=os.path.dirname(__file__), index_file='static/extract.html', n_jobs=1):
        """Start a web server serving the extraction API at /extract

        **Parameters**

        basedir : string, optional, defaults to current directory
            The base directory to serve static files.
            If None, no static files will be served.

        index_file : string, optional, defaults to "static/extract.html"
            The page to be served as interface.

        n_jobs : int, optional, defaults to 1
            The number of processes to run. This determines the number of concurrent requests that can be handled
            simultaneously.
        """
        self.check_initialized()

        def _extract_process(extractor, pipe, ner_tagger, np_chunker):
            while True:
                params = pipe.recv()
                try:
                    result = extractor.extract(ner_tagger=ner_tagger, np_chunker=np_chunker, **params)
                except Exception:
                    LOGGER.error('Title: %s\nText: %s' % (params['title'], params['data']), exc_info=True)
                    result = {}
                pipe.send(result)

        if basedir is not None:
            parent_pipes = []
            workers = []
            if n_jobs < 1:
                n_jobs += mp.cpu_count()
            for i in xrange(n_jobs):
                parent_pipe, child_pipe = mp.Pipe()
                semaphore = threading.Semaphore()
                if self.ner_tagger is not None:
                    ner_tagger = KnorexNERTagger()
                    # ner_gateway = JavaGateway(GatewayClient(port=48100), auto_convert=True, auto_field=True)
                    # ner_tagger = ner_gateway.entry_point
                else:
                    ner_tagger = None
                np_chunker = MaxentNPChunker()
                worker = mp.Process(target=_extract_process, args=(self, child_pipe, ner_tagger, np_chunker))
                worker.start()
                workers.append(worker)
                parent_pipes.append((parent_pipe, semaphore))
            application = Application([
                (r'/', MainHandler, {'index_file': index_file}),
                (r'/version', VersionHandler),
                (r'/extract', ConceptExtractorHandler, {'pipes': parent_pipes}),
                (r'/static/(.*)', StaticFileHandler, {'path': os.path.join(basedir, 'static')}),
            ])
        else:
            application = Application([
                (r'/extract', ConceptExtractorHandler, {'extractor': self}),
            ])
        application.listen(self.port)
        print 'Server started at port %d' % self.port
        try:
            IOLoop.instance().start()
        except KeyboardInterrupt:
            for worker in workers:
                worker.join(timeout=1)
                worker.terminate()
                worker.join()

    def get_scorer(self, scorer_name):
        global scorers
        return getattr(self, scorers[scorer_name])


############################
# Web application handlers #
############################
class ConceptExtractorHandler(RequestHandler):
    """Web application handler
    """

    def initialize(self, pipes):
        self.pipes = pipes

    @asynchronous
    def get(self):
        text = self.get_argument('data', default=None)
        title = self.get_argument('title', default='')
        boost_method = self.get_argument('boost_method', default='1')
        n = self.get_argument('n', default='10')
        k = self.get_argument('k', default='25')
        n_ranked = self.get_argument('n_ranked', default='25')
        rank_sim = self.get_argument('rank_sim', default='spearman_rank_similarity')
        return_values_str = self.get_argument('return_values', default='concepts,terms,phrases')
        return_values = [value.strip() for value in return_values_str.split(',')]
        callback = self.get_argument('callback', default=None)
        try:
            boost_method = int(boost_method)
        except ValueError:
            boost_method = 1
        if boost_method == 1:
            boost_concept = True
            boost_lower = False
        elif boost_method == 2:
            boost_concept = True
            boost_lower = True
        else:
            boost_concept = False
            boost_lower = False
        try:
            n = int(n)
        except ValueError:
            n = 10
        if n < 1:
            n = 1
        try:
            k = int(k)
        except ValueError:
            k = 25
        if k < 1:
            k = 1
        try:
            n_ranked = int(n_ranked)
        except ValueError:
            n_ranked = 25
        if n_ranked < 0:
            n_ranked = 0
        if rank_sim not in {'gk_rank_similarity', 'spearman_rank_similarity'}:
            rank_sim = 'spearman_rank_similarity'
        params = {'data': text,
                  'title': title,
                  'with_score': True,
                  'n': n,
                  'k': k,
                  'n_ranked': n_ranked,
                  'return_values': return_values,
                  'boost_concept': boost_concept,
                  'boost_lower': boost_lower,
                  'rank_sim': rank_sim
                  }
        threading.Thread(target=self._get_thread, args=(params, callback)).start()

    def _on_finish(self, result, callback):
        """Finisher function for asynchronous get method
        """
        self.set_header("Content-Type", "application/json")
        if callback:
            self.write('%s(%s)' % (callback, result))
        else:
            self.write(result)
        self.flush()
        self.finish()

    def _get_thread(self, params, callback):
        result = ''
        try:
            if params['data'] is not None:
                available_pipe = None
                semaphore = None
                while not available_pipe:
                    for pipe, sem in self.pipes:
                        if sem.acquire(False):
                            available_pipe = pipe
                            semaphore = sem
                            break
                    if not available_pipe:
                        time.sleep(2.0 / len(self.pipes))
                start_time = time.time()
                try:
                    available_pipe.send(params)
                    extraction_output = available_pipe.recv()
                finally:
                    LOGGER.info('Extraction done in %.3fs' % (time.time() - start_time))
                    semaphore.release()
                result = json.dumps(extraction_output)
        finally:
            IOLoop.instance().add_callback(lambda: self._on_finish(result, callback))

    @asynchronous
    def post(self):
        return self.get()


class VersionHandler(RequestHandler):

    def get(self):
        self.write(VERSION)

    def post(self):
        self.get()


class MainHandler(RequestHandler):

    def initialize(self, index_file):
        self.index_file = index_file

    def get(self):
        self.redirect(self.index_file, permanent=True)

def main():
    sys.stdout = Unbuffered(sys.stdout)
    parsed = parse_arguments()
    parsed = dict(**vars(parsed))
    start = parsed['start']
    del(parsed['start'])
    n_jobs = parsed['n_jobs']
    del(parsed['n_jobs'])
    extractor = KeyTermsExtractor(**parsed)
    extractor.initialize()
    if extractor.testdir:
        extractor.extract_from_directory(parsed.testdir, with_score=False, arff_output=parsed.testdir_arff)
    elif start == 1:
        extractor.start_server(n_jobs=n_jobs)
    elif start == 2:
        from pprint import pprint
        while True:
            text = raw_input('Text: ')
            pprint(extractor.extract(text, with_score=True, n=20))

if __name__ == '__main__':
    main()
