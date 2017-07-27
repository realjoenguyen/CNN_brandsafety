#!/usr/bin/python
import logging
import os
import re
from itertools import chain, ifilter

import nltk
import numpy as np
from knx.text.postagger import default_tagger as postagger
from knx.text.preprocess_text import NormalizationText as NT
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from unidecode import unidecode

from BS.knx.text.tokenizer import default_tokenizer as tokenizer

DEBUG = False
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)
if DEBUG:
    LOGGER.setLevel(logging.DEBUG)


def count_word(word, pos=None):
    return sum(lemma.count()
               for synset in wn.synsets(word, pos)
               for lemma in synset.lemmas()
               if word in lemma.name().split('.'))


class Lemmatizer(nltk.wordnet.WordNetLemmatizer):
    all_cached = False

    @staticmethod
    def cache_all():
        """Cache all the Synset so that this class becomes thread-safe"""
        if not Lemmatizer.all_cached:
            for form in wn._lemma_pos_offset_map:
                for pos in wn._lemma_pos_offset_map[form]:
                    for offset in wn._lemma_pos_offset_map[form][pos]:
                        wn._synset_from_pos_and_offset(pos, offset)
            Lemmatizer.all_cached = True

    def lemmatize(self, word, pos=None):
        try:
            if pos is not None:
                return max(wn._morphy(word, pos), key=lambda x: count_word(x, pos))
            else:
                return max(chain.from_iterable(wn._morphy(word, pos) for pos in {'n', 'v', 'a', 'r'}),
                           key=lambda x: count_word(x))
        except ValueError:
            return word


class DocToFeature:
    pos_tagger = postagger
    stemmer = nltk.stem.porter.PorterStemmer()
    lemmatizer = Lemmatizer()
    word_normalization_opts = {'stem', 'lemmatize', 'none'}

    def __init__(self, word_normalization='stem', lowercase=True, keep_nnp=False, transliteration=True):
        self.filelist = None
        self.vocabulary = None
        self.mapping = None
        if word_normalization not in DocToFeature.word_normalization_opts:
            raise ValueError('Invalid value for word_normalization: %s\n'
                             'Valid values are: %s' % (word_normalization, str(DocToFeature.word_normalization_opts)))
        self.word_normalization = word_normalization
        self.lowercase = lowercase
        self.keep_nnp = keep_nnp
        self.transliteration = transliteration

        # Objects
        self.count_vectorizer = None
        self.stop_words = set(stopwords.words('english')).union(stop_words.ENGLISH_STOP_WORDS)

        # Functions
        self.preprocessor = self._build_preprocessor()
        self.tokenizer = self._build_tokenizer()
        self.postagger = self._build_postagger()
        self.postprocessor = self._build_postprocessor()
        self.analyzer = self._build_analyzer(stop_words=stop_words)

        # Constants
        self.has_multicore_support = False
        self.PARALLEL_THRESHOLD = 200

    def _build_preprocessor(self):
        def preprocess(string):
            string = NT.normalize_to_unicode(string)
            if self.transliteration:
                string = unidecode(string)
            return string
        self.preprocessor = preprocess

    def _build_tokenizer(self):
        def tokenize(string):
            sentences = nltk.sent_tokenize(string)
            #tokens = []
            flag = False
            if type(string) == str:
                flag = True
            for sentence in sentences:
                for token in tokenizer.tokenize(sentence):
                    if flag:
                        yield intern(token)
                    else:
                        yield token
            #    tokens.extend((intern(token) for token in tokenizer.tokenize(sentence)))
            #return tokens
        self.tokenizer = tokenize

    def _build_postagger(self):
        def postag(string):
            return DocToFeature.pos_tagger.tag(string)
        postag('They refuse to refuse the produce I produce.')  # Initialization
        self.postagger = postag

    def _build_postprocessor(self):
        def postprocess(word, pos=None):
            if self.word_normalization == 'stem':
                word = DocToFeature.stemmer.stem(word)
            elif self.word_normalization == 'lemmatize':
                lemmatizer = DocToFeature.lemmatizer
                if pos is not None:
                    word = lemmatizer.lemmatize(word, pos)
                else:
                    word = lemmatizer.lemmatize(word)
            if not re.match('.*[A-Za-z].*', word):
                return ''
            if self.lowercase:
                return word.lower()
            else:
                return word
        postprocess('species')  # Initialization
        self.postprocessor = postprocess

    def _build_analyzer(self, input_type='string', stop_words=None):
        if stop_words is None:
            stop_words = self.stop_words
        if input_type == 'file':
            def analyze(string):
                with open(string, 'r') as infile:
                    string = infile.read()
                string = self.get_preprocessor()(string)
                tokens = self.get_tokenizer()(string)
                postprocessor = self.get_postprocessor()
                result = ifilter(None, (postprocessor(word) for word in tokens if word not in stop_words))
                return result
        else:
            def analyze(string):
                string = self.get_preprocessor()(string)
                tokens = self.get_tokenizer()(string)
                postprocessor = self.get_postprocessor()
                result = ifilter(None, (postprocessor(word) for word in tokens if word not in stop_words))
                return result
        self.analyzer = analyze

    def _build_count_vectorizer(self, vocabulary=None, input_type='string', min_df=3, max_df=0.5, stop_words=None):
        if vocabulary is None:
            vocabulary = self.vocabulary
        if stop_words is None:
            stop_words = self.stop_words
        analyzer = self.get_analyzer(input_type)
        # Explanation of parameters
        # min_df=3 -> Include only terms that appear in >= 3 documents
        # max_df=0.5 -> Include only terms that appear in <= 50% of the documents
        # analyzer=analyzer -> Use the analyzer defined above to do the processing
        # stop_words='english' -> Use the default english stop words list to filter out stop words
        # vocabulary -> Use the terms in this vocabulary, or deduce from data if None
        self.count_vectorizer = CountVectorizer(min_df=min_df,
                                                max_df=max_df,
                                                analyzer=analyzer,
                                                stop_words=stop_words,  # In case vocabulary is None
                                                vocabulary=vocabulary)
        _func_code = self.count_vectorizer.fit.func_code
        if 'n_jobs' in _func_code.co_varnames[:_func_code.co_argcount]:
            self.has_multicore_support = True

    def get_preprocessor(self, force=False):
        if self.preprocessor is None or force:
            self._build_preprocessor()
        return self.preprocessor

    def get_tokenizer(self, force=False):
        if self.tokenizer is None or force:
            self._build_tokenizer()
        return self.tokenizer

    def get_postagger(self, force=False):
        if self.postagger is None or force:
            self._build_postagger()
        return self.postagger

    def get_postprocessor(self, force=False):
        if self.postprocessor is None or force:
            self._build_postprocessor()
        return self.postprocessor

    def get_analyzer(self, input_type='string', stop_words=None, force=False):
        if self.analyzer is None or force:
            self._build_analyzer(input_type, stop_words)
        globals()['analyze'] = self.analyzer
        return self.analyzer

    def get_count_vectorizer(self, vocabulary=None, input_type='string', force=False,
                             min_df=3, max_df=0.5, stop_words=None):
        if self.count_vectorizer is None or force:
            self._build_count_vectorizer(vocabulary=vocabulary, input_type=input_type,
                                         min_df=min_df, max_df=max_df, stop_words=None)
        return self.count_vectorizer

    def str_to_tf(self, text, vocabulary=None, fit_vocabulary=False, n_jobs=0):
        """Read a string or a list of strings and return the TF matrix in coo_matrix format

        Parameters
        ----------
        text : str or list
            Input documents

        vocabulary : dict
            The mapping from words to column indices in the resulting TF matrix

        fit_vocabulary : boolean
            If False, existing vocabulary will be used, of if there is no existing vocabulary, the provided vocabulary
                will be used
            If True, any existing or provided vocabulary will be ignored, and the vectorizer will be fitted into the
                input text.

        n_jobs : int, optional
            The number of processes that should be spawned to do the counting.
            If this is 0, this will be assigned with a value determined based on some heuristics depending on the
            input text.

        Returns
        -------
        docs_tf : scipy.sparse.coo_matrix
            The TF matrix in sparse matrix

        Notes
        -----
        The vocabulary argument only influence the first call to this method. Subsequent calls will vectorize the input
        according to the vocabulary passed to the first call to this method, unless fit_vocabulary=True
        Call self._build_count_vectorizer(vocabulary, min_df=0, max_df=1, stop_words={}) to rebuild the vectorizer.
        """
        if fit_vocabulary:
            count_vectorizer = self.get_count_vectorizer(min_df=0.0, max_df=1.0)
        else:
            count_vectorizer = self.get_count_vectorizer(vocabulary)

        if n_jobs == 0:
            if (type(text) in {str, unicode} or
                    len(text) < self.PARALLEL_THRESHOLD or
                    len(text) < 10 * self.PARALLEL_THRESHOLD):
                n_jobs = 1
            else:
                n_jobs = -1

        if n_jobs != 1:
            if self.word_normalization == 'lemmatize' and not Lemmatizer.all_cached:
                Lemmatizer.cache_all()

        if type(text) in {str, unicode}:
            text = [text]
        if self.has_multicore_support:
            if fit_vocabulary:
                return count_vectorizer.fit_transform(text, n_jobs=n_jobs)
            else:
                return count_vectorizer.transform(text, n_jobs=n_jobs)
        else:
            # LOGGER.warn('Using normal scikit-learn, no multiprocessing is used')
            if fit_vocabulary:
                return count_vectorizer.fit_transform(text)
            else:
                return count_vectorizer.transform(text)

    # def doc_to_tf(self, collection, vocabulary=None, n_jobs=0):
    #     """Read a collection of files and return the TF matrix in coo_matrix format and a list containing the filenames
    #
    #     Parameters
    #     ----------
    #     collection : str or list
    #         The input documents, this can be a list of file names, or it can be a string storing the path to a folder
    #         containing the files
    #
    #     vocabulary : dict, optional
    #         The mapping from words to column indices in the resulting TF matrix.
    #         If this is None (as the default), the vocabulary will be derived from the input documents.
    #
    #     n_jobs : int, optional
    #         The number of processes that should be spawned to do the counting.
    #         If this is 0, this will be assigned with a value determined based on some heuristics depending on the
    #         input text.
    #
    #     Returns
    #     -------
    #     docs_tf : scipy.sparse.coo_matrix
    #         The TF matrix in sparse matrix
    #
    #     Notes
    #     -----
    #     This method will initialize some data members of this DocToFeature instance:
    #         filelist <list> the list of filenames used to produce the docs_tf
    #         vocabulary <dict> the list of words found in form of mapping from word to its feature index
    #         mapping <dict> the mapping from feature index to word
    #     """
    #     if type(collection) is not str:
    #         self.filelist = collection
    #     else:
    #         base_dir = collection
    #         if not base_dir.endswith('/'):
    #             base_dir = base_dir + '/'
    #         self.filelist = [base_dir + filename
    #                          for filename in sorted(os.listdir(collection), key=lambda x: x.lower())
    #                          if filename != '.DS_Store']
    #
    #     count_vectorizer = self.get_count_vectorizer(vocabulary=vocabulary, input_type='file')
    #     if not self.has_multicore_support or len(self.filelist) < self.PARALLEL_THRESHOLD:
    #         docs_tf = csr_matrix(count_vectorizer.fit_transform(self.filelist), copy=False)
    #     else:
    #         if self.word_normalization == 'lemmatize' and not Lemmatizer.all_cached:
    #             Lemmatizer.cache_all()
    #         docs_tf = csr_matrix(count_vectorizer.fit_transform(self.filelist, n_jobs=-1), copy=False)
    #
    #     self.vocabulary = count_vectorizer.vocabulary_
    #     self.mapping = count_vectorizer.get_feature_names()
    #     del(count_vectorizer)
    #
    #     if DEBUG:
    #         (row, col) = docs_tf.shape
    #         with Timing('Writing log files...'):
    #             logfile = open(base_dir + '_tf.txt', 'w')
    #             for i in range(row):
    #                 logfile.write(
    #                     ' '.join(map(str, docs_tf.getrow(i).toarray()[0])) + '\n')
    #             logfile.close()
    #
    #     return docs_tf

    def doc_to_tf(self, collection, n_gram = 2, vocabulary=None, n_jobs=0):
        if type(collection) is not str:
            self.filelist = collection
        else:
            base_dir = collection
            if not base_dir.endswith('/'):
                base_dir = base_dir + '/'
            self.filelist = [base_dir + filename
                             for filename in sorted(os.listdir(collection), key=lambda x: x.lower())
                             if filename != '.DS_Store']
        corpus = []
        print 'DEBUG'
        for path_filename in self.filelist:
            content = open(path_filename).read()
            corpus.append(content)

        vectorizer = CountVectorizer(min_df=3, max_df=0.5, ngram_range=(1, n_gram), vocabulary=vocabulary)
        X = vectorizer.fit_transform(corpus)
        self.vocabulary = vectorizer.vocabulary_
        self.mapping = vectorizer.get_feature_names()
        return X

def tf_to_tfidf(docs_tf, sublinear_tf=False, smooth_idf=False, use_idf=True, norm='l2', idf_diag=None):
    """Transform TF matrix into TFIDF matrix

    Return value:
        A tuple which contains:
            docsTFIDF - The TFIDF scores in coo_matrix format
            idf_diag - The idf information produced by TfidfTransformer

    Use the function from sklearn.feature_extraction library
    """
    tfidf_transformer = TfidfTransformer(
        sublinear_tf=sublinear_tf, smooth_idf=smooth_idf, use_idf=use_idf, norm=norm)
    if idf_diag is not None:
        tfidf_transformer._idf_diag = idf_diag
        docsTFIDF = tfidf_transformer.transform(docs_tf)
    else:
        tfidf_transformer._idf_diag = None
        docsTFIDF = tfidf_transformer.fit_transform(docs_tf)
    return (docsTFIDF, tfidf_transformer._idf_diag)

def tf_to_okapi(docs_tf, k1=20, b=1, idfs=None, avg_doc_len=None):
    """Transform TF matrix into BM25.TFIDF matrix

    Return value:
        A tuple which contains:
            docsScore - The Okapi scores in coo_matrix format
            idfs - A list containing IDF values for each term
            avg_doc_len - A floating point number representing the average document length

    Parameters:
    k1: saturation parameter (the tf.idf score converges to k1+1 as tf goes to infinity)
    b: document length factor (1 refers to full inclusion of document length, 0 refers to ignoring document length)

    Arguments:
    idfs: A list representing the value of IDF for each term in order
    avg_doc_len: A floating point number representing the average document length

    The arguments idfs and avg_doc_len should be provided whenever the statistics from docs_tf is not sufficient.
    For example when docs_tf represents only one or a few documents (usually used in real time classification)

    Implemented based on paper by John S. Whissell and Charles L. A. Clarke. 2011.
    Improving document clustering using Okapi BM25 feature weighting
    """

    (num_docs, num_terms) = docs_tf.shape  # Each row represents a document
    if avg_doc_len is None:
        avg_doc_len = float(docs_tf.sum()) / num_docs

    # Learn about csr_matrix (and other sparse matrices) from http://docs.scipy.org/doc/scipy/reference/sparse.html
    if idfs is None:
        # The number of documents that contain a term (which is represented in
        # a column) is the difference between adjacent indptr
        dfs = np.diff(docs_tf.tocsc().indptr)
        # IDF formulation: idf = log(numDoc / df)
        idfs = np.log(float(num_docs) / dfs)
    # Converted to diagonal matrix for later multiplication
    idfMat = spdiags(idfs, 0, num_terms, num_terms)

    docs = docs_tf.tocsr()
    numer = docs.copy()
    numer = numer * (k1 + 1)

    denom = docs.copy()
    # Original: k1 * ( (1-b) + b*docLen/avg_doc_len )
    val = k1 * (1 - b) + docs.sum(axis=1) * (k1 * (b / avg_doc_len))
    denom.data = denom.data + np.repeat(val.tolist(), np.diff(denom.indptr))  # tf := tf + val

    docs.data = numer.data / denom.data
    docs = docs * idfMat
    normalize(docs, copy=False)
    return (docs, idfs, avg_doc_len)


def tf_to_midf(docs_tf, sublinear_tf=True, norm='l2'):
    """Transform TF matrix into TF.MIDF matrix

    Return value:
        A tuple which contains:
            docs - The MIDF scores in coo_matrix format

    sublinear_tf : boolean, optional, True by default
        Uses (1+log(tf+exp(-1))) if true, tf otherwise

    norm : string, {'l1','l2',None}, 'l2' by default
        Uses arithmetic mean normalization if 'l1', cosine normalization if 'l2', no normalization otherwise

    Implemented based on paper by Deisy. C. et al. 2010. A novel term weighting scheme MIDF for text categorization
    """
    (num_docs, num_terms) = docs_tf.shape  # Each row represents a document

    docs = docs_tf.tocsr()
    # Get docLen for every document (row) by summing over x-axis
    docLens = np.array(docs.sum(axis=1).T, dtype=float)
    # Get number of unique terms per document
    num_uniq_terms = np.diff(docs.indptr)
    midf = docLens / num_uniq_terms  # MIDF formulation: midf = DFR / DF

    if sublinear_tf:
        docs.data = np.log(docs.data + np.exp(-1)) + 1.0

    # Multiply each entry with its document MIDF score
    docs.data = docs.data * np.repeat(midf.tolist(), np.diff(docs.indptr))

    if norm is not None:
        normalize(docs, norm=norm, copy=False)

    return (docs,)


def tf_to_rf(docs_tf, labels=None, rf_vectors=None):
    """Transform TF matrix into N TF.RF matrices, where N is the number of classes

    Parameters:
        docs_tf <coo_matrix> - The raw TF counts
        labels <list> - The list of labels (in the same order as docs_tf)
        rf_vectors <dict<list>> - Mapping from class names to RF vectors

    Return value:
        A tuple which contains:
            docsScores <dict<coo_matrix>> - A dict from class names to TF.RF matrices
            rf_vectors <dict<list>> - Mapping from class names to RF vectors
    """
    result = {}
    (num_docs, num_terms) = docs_tf.shape
    if labels is not None:
        rf_vectors = {}
        classes = sorted(list(set(labels)))
    else:
        classes = sorted(rf_vectors.keys())
    for class_ in classes:
        docs = docs_tf.tocsc()
        docs.astype(float)
        if labels is None:
            rf_vector = rf_vectors[class_]
        else:
            pos_mask = map(lambda x: 1 if x == class_ else 0, labels)
            neg_mask = map(lambda x: 1 - x, pos_mask)
            mask = csr_matrix([pos_mask, neg_mask])
            # Split the documents into positive and negative class
            ac_matrix = mask * docs
            # The TF counts in positive class
            aVector = ac_matrix.getrow(0).toarray()[0]
            # The TF counts in negative class, 0 is converted to 1
            cVector = np.array(
                map(lambda x: x if x > 0 else 1, ac_matrix.getrow(1).toarray()[0]))
            rf_vector = map(
                lambda (a, c): np.log2(2 + (a / c)), zip(aVector, cVector))
            rf_vectors[class_] = rf_vector
        docs.data = docs.data * np.repeat(rf_vector, np.diff(docs.indptr))
        result[class_] = docs.tocoo()
    return (result, rf_vectors)


def main():
    docs_tf = coo_matrix([[1, 0, 1, 0, 0], [1, 0, 0, 1, 1], [1, 1, 0, 1, 0]])
    labels = ['yes', 'no', 'X']
    print tf_to_rf(docs_tf, labels)

if __name__ == '__main__':
    main()
