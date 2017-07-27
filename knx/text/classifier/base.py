#!/usr/bin/python
import argparse
import json
import os
import pickle
import sys
from functools import partial
from multiprocessing import Pipe, Process, cpu_count

import numpy
from knx.metrics.confusionmatrix import ConfusionMatrix
from knx.text.doc_to_feature import DocToFeature, tf_to_tfidf, tf_to_okapi, tf_to_midf, tf_to_rf
from knx.text.preprocess_text import NormalizationText
from knx.util.logging import Timing, Unbuffered
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix
from sklearn import svm, linear_model, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler, StaticFileHandler

from BS.knx.text.feature_to_arff import FeatureToArff

NormalizationText = NormalizationText()
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gen-py'))
# Thrift Server
from category import Category
from thrift.transport import TSocket
from thrift.protocol import TBinaryProtocol
from thrift.server.TNonblockingServer import TNonblockingServer
import logging

LOGGER = logging.getLogger(__name__)

#################################
# Classifier and scorer options #
#################################
classifiers = {
    'NaiveBayes': lambda: MultinomialNB(),
    'LinearSVC': lambda: OneVsRestClassifier(svm.LinearSVC(C=5, class_weight='auto')),
    'SVC': lambda: svm.SVC(C=50, gamma=0.05, cache_size=1000),
    'SGDClassifier': lambda: linear_model.SGDClassifier()
}

scorers = {
    'tf': 'scorer_tf',
    'tfidf': 'scorer_tfidf',
    'okapi': 'scorer_okapi',
    'midf': 'scorer_midf',
    'rf': 'scorer_rf'
}

##############################
# Classifier helper function #
##############################


def classifier_svc():
    """Used for finding the best parameter configuration for SVC

    The result is that varying C from the set {10,50,100,125,150} does not change the accuracy much,
    while the best result is obtained by setting gamma to 0.025, 0.05, or, 0.075
    """
    parameters = {'C': [10, 50, 100, 125, 150],
                  'gamma': [0.1, 0.01, 0.025, 0.05, 0.075],
                  'kernel': ['rbf'],
                  'class_weight': ['auto', None]}
    classifier = GridSearchCV(svm.SVC(), parameters, cv=5, scoring='accuracy')
    return classifier


def fit(classifier_name, doc_term, labels):
    """Fit a classifier into a feature vector matrix or a set of matrices in case of one-vs-all

    **Parameters**

    classifier_name : str
        Classifier name

    doc_term : coo_matrix or dict<coo_matrix>
        Feature vector scores or a mapping from class names into feature vector scores

    labels : list
        The class labels (in the same order as what is given in the feature vector scores)
    """
    global classifiers
    if type(doc_term) is dict:
        fitted_classifiers = {}
        for class_ in doc_term.keys():
            classifier = classifiers[classifier_name]()
            classifier.fit(doc_term[class_], map(lambda x: x if x == class_ else '**NO**', labels))
            fitted_classifiers[class_] = classifier
        return fitted_classifiers
    elif type(doc_term) in [coo_matrix, csr_matrix, csc_matrix]:
        classifier = classifiers[classifier_name]()
        classifier.fit(doc_term, labels)
        return classifier


def _scores_to_labels(scores, classes):
    """Convert a list of scores into a sorted list of (label,score) for which the score is positive

    If no score is positive, return the largest.

    The returned list is sorted in decreasing order, so the first entry is the class label to be assigned
    in non-multilabel case
    """
    if max(scores) <= 0:
        # return [(classes[numpy.argmax(scores)], max(scores))]  # Return the highest class
        return [("others", 1.0)]  # Return "others" when there is no positive score
    else:
        result = sorted([(classes[idx], value) for idx, value in enumerate(scores) if value > 0],
                        key=lambda x: x[1], reverse=True)  # Return only positive scores
        # result = sorted([(classes[idx], value) for idx, value in enumerate(scores)],
        #                key=lambda x: x[1], reverse=True)  # Return all scores
        return result


def predict(fitted_classifier, doc_term, pos_label=None):
    """Predict the class labels of a doc_term matrix using the fitted classifier (or classifiers in one-vs-all case)

    **Parameters**

    fitted_classifier : classifier or dict<classifier>
        If a single classifier is provided, that will be used directly to predict the class labels using the
        predict method of the classifier using doc_term as the input

    doc_term : coo_matrix
        Documents-terms score matrix
    """
    if type(fitted_classifier) is dict:
        scores = []
        classes = []
        for class_ in sorted(fitted_classifier.keys()):
            classifier = fitted_classifier[class_]
            try:
                scores.append(classifier.decision_function(doc_term[class_]).tolist())
            except:
                LOGGER.error('RF is called with a non LinearSVC classifier')
                raise Exception('ERROR: Currently only LinearSVC is supported to work with RF!')
            classes.append(class_)
        if len(classes) == 2:
            if classes[0] == pos_label:
                scores = [(x, -x) for x in scores[0]]
            else:
                scores = [(-x, x) for x in scores[1]]
        else:
            scores = zip(*scores)
    else:
        classes = fitted_classifier.classes_
        try:  # Whether this classifier contains decision_function method
            scores = fitted_classifier.decision_function(doc_term)
            if len(classes) == 2:
                scores = [(-x, x) for x in scores]
        except:
            try:  # Whether this classifier contains predict_log_proba method
                scores = fitted_classifier.predict_log_proba(doc_term)
            except:
                labels = fitted_classifier.predict(doc_term)
                labels = map(lambda x: [(x, 1)], labels)
                return labels
    labels = map(partial(_scores_to_labels, classes=classes), scores)
    return labels


def take_best_label(scores):
    return numpy.array(map(lambda x: x[0][0], scores))


def _cross_validate(classifier_name, doc_term, labels, cv=10, result_file=None, pos_label=None):
    """Do a cross validation on the doc_term using the specified classifier

    **DEPRECATED**
    Use the multi-core version (below) instead, which is faster

    **Returns**

    result : tuple
        A tuple consisting of three elements:
            - accuracy : numpy.array
                accuracy for each fold

            - F1-score : numpy.array
                F1-score for each fold

            - confMat : numpy.array
                confusion matrix for each fold
    """
    indices = StratifiedKFold(labels, n_folds=cv, indices=True)
    accuracy = []
    f1score = []
    conf_mat = []
    for train_index, test_index in indices:
        if type(doc_term) is dict:
            train_doc_term = {}
            test_doc_term = {}
            for className in doc_term:
                train_doc_term[className] = doc_term[className].tocsr()[train_index, :]
                test_doc_term[className] = doc_term[className].tocsr()[test_index, :]
        else:
            train_doc_term = doc_term.tocsr()[train_index, :]
            test_doc_term = doc_term.tocsr()[test_index, :]
        train_labels = [labels[i] for i in train_index]
        test_labels = numpy.array([labels[i] for i in test_index])
        fitted_classifiers = fit(classifier_name, train_doc_term, train_labels)
        prediction = take_best_label(predict(fitted_classifiers, test_doc_term, pos_label))

        accuracy.append(metrics.accuracy_score(test_labels, prediction))
        f1score.append(metrics.f1_score(test_labels, prediction, average='macro'))
        conf_mat.append(numpy.array(metrics.confusion_matrix(test_labels, prediction)))
    return (numpy.array(accuracy), numpy.array(f1score), conf_mat, test_labels, prediction)


def cross_validate(classifier_name, doc_term, labels, cv=10, result_file=None, pos_label=None):
    """Do a cross validation on the doc_term using the specified classifier

    This is the multi-core version of the cross_validate function.
    This runs 3-5x faster than the single process version in dev-1 with 16 cores

    **Returns**

    result : tuple
        A tuple consisting of three elements:
            - accuracy : numpy.array
                accuracy for each fold

            - F1-score : numpy.array
                F1-score for each fold

            - confMat : numpy.array
                confusion matrix for each fold
    """
    indices = StratifiedKFold(labels, n_folds=cv, indices=True)
    validators = []
    result_pipes = []
    results = []
    for index in indices:
        (parent, child) = Pipe()
        result_pipes.append(parent)
        validator = Process(target=validate_one_fold, args=(index,
                                                            classifier_name,
                                                            doc_term,
                                                            labels,
                                                            child,
                                                            pos_label))
        validator.start()
        validators.append(validator)
    for validator, result_pipe in zip(validators, result_pipes):
        results.append(result_pipe.recv())
        validator.join()
    results = map(numpy.array, zip(*results))
    return results


def validate_one_fold(indices, classifier_name, doc_term, labels, result_pipe=None, pos_label=None):
    (train_index, test_index) = indices
    if type(doc_term) is dict:
        train_doc_term = {}
        test_doc_term = {}
        for className in doc_term:
            train_doc_term[className] = doc_term[className].tocsr()[train_index, :]
            test_doc_term[className] = doc_term[className].tocsr()[test_index, :]
    else:
        train_doc_term = doc_term.tocsr()[train_index, :]
        test_doc_term = doc_term.tocsr()[test_index, :]
    train_labels = [labels[i] for i in train_index]
    test_labels = numpy.array([labels[i] for i in test_index])
    fitted_classifiers = fit(classifier_name, train_doc_term, train_labels)
    label_score_prediction = predict(fitted_classifiers, test_doc_term, pos_label)
    prediction = take_best_label(label_score_prediction)

    # print "Prediction:", prediction

    accuracy = metrics.accuracy_score(test_labels, prediction)
    if len(set(labels)) == 2:
        f1score = metrics.f1_score(test_labels, prediction, pos_label=pos_label)
        precision_score = metrics.precision_score(test_labels, prediction, pos_label=pos_label)
        recall_score = metrics.recall_score(test_labels, prediction, pos_label=pos_label)
    else:
        f1score = metrics.f1_score(test_labels, prediction, average='weighted')
        precision_score = metrics.precision_score(test_labels, prediction, average='weighted')
        recall_score = metrics.recall_score(test_labels, prediction, average='weighted')
    # confMat = numpy.array(metrics.confusion_matrix(testLabels, prediction))
    # Prepare data in a suitable data structure to calculate metrics for multi labels
    test_labels_for_multi_label = []
    for i in range(len(test_labels)):
        item = test_labels[i]
        sub_item = [item]
        test_labels_for_multi_label.append(sub_item)

    prediction_for_multi_label = []
    for i in range(len(label_score_prediction)):
        item = label_score_prediction[i]
        sub_item = []
        for label, score in item:
            sub_item.append(label)
        prediction_for_multi_label.append(sub_item)

    # Compuare scores for multi labels
    accuracy_for_multi_label = metrics.accuracy_score(test_labels_for_multi_label, prediction_for_multi_label)
    f1score_for_multi_label = metrics.f1_score(test_labels_for_multi_label,
                                               prediction_for_multi_label,
                                               average='weighted')
    precision_score_for_multi_label = metrics.precision_score(test_labels_for_multi_label,
                                                              prediction_for_multi_label,
                                                              average='weighted')
    recall_score_for_multi_label = metrics.recall_score(test_labels_for_multi_label,
                                                        prediction_for_multi_label,
                                                        average='weighted')

    single_label_score = [accuracy, precision_score, recall_score, f1score]
    multi_label_score = [accuracy_for_multi_label, precision_score_for_multi_label,
                         recall_score_for_multi_label, f1score_for_multi_label]

    if result_pipe is None:
        return (single_label_score, test_labels, prediction, multi_label_score,
                test_labels_for_multi_label, prediction_for_multi_label)
    else:
        result_pipe.send((single_label_score, test_labels, prediction, multi_label_score,
                          test_labels_for_multi_label, prediction_for_multi_label))


###################
# Helper function #
###################


def loadarff(filename):
    """Load ARFF file which contains feature vectors with the class label as the last index

    **Parameters**

    filename : string
        The ARFF file name to be loaded

    **Returns**

    tfScore : coo_matrix
        The feature vectors

    labels : list
        The list of labels

    classes : str
        The classes available for the label, represented as is from ARFF file

    **Notes**

    This method assumes that the ARFF file only contains floating points and nominal, which
    do not contain any commas other than delimiting commas
    """
    with open(filename, 'r') as arff:
        lines = arff.readlines()
    classes = None
    num_attr = 0
    for idx, line in enumerate(lines):
        if line.find('LABEL') != -1:
            classes = line.split(' ', 2)[2].strip()
        if line.startswith('@attribute'):
            num_attr += 1
        if line.startswith('@data'):
            dataIdx = idx + 1
            break
    num_attr -= 1
    lines = lines[dataIdx:]
    is_sparse = (lines[0][0] == '{')

    data = []
    labels = []
    if is_sparse:
        row_idx = []
        col_idx = []
        for row, line in enumerate(lines):
            # Read sparse
            is_sparse = True
            values = []
            indices = []
            line = line.strip('{} \r\n\t')
            tmp = line.rsplit(' ', 1)
            items = tmp[0].split(',')
            items[-1] += ' ' + tmp[1]
            # (indices, values) = zip(*(item.split(' ',1) for item in items)) # Wow, this is slower!
            for item in items:
                (index, value) = item.split(' ', 1)
                indices.append(index)
                values.append(value)
            data.extend(values[:len(values) - 1])
            row_idx.extend([row] * (len(values) - 1))
            col_idx.extend(indices[:len(values) - 1])
            labels.append(values[-1])
        return (coo_matrix((data, (row_idx, col_idx)), shape=(len(labels), num_attr), dtype=float), labels, classes)
    else:
        for row, line in enumerate(lines):
            # Read dense
            line = line.strip(' \r\n\t')
            values = line.split(',')
            data.append(values[:len(values) - 1])
            labels.append(values[-1])
        return (coo_matrix(data, dtype=float), labels, classes)


def print_confusion_matrix(confusion_matrix, classes):
    for i in range(len(classes)):
        print '    %s' % chr(i + 97),
    print '| total'
    print '-' + ('------' * (len(classes) + 1))
    for i in range(len(classes)):
        for j in range(len(classes)):
            print '%5d' % confusion_matrix[i][j],
        print '| %5d' % sum(confusion_matrix[i]),
        print '%s=%s' % (chr(i + 97), classes[i])


####################
# Arguments parser #
####################


class FileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, [])
        getattr(namespace, self.dest).append(values[0])
        if len(values) >= 2:
            setattr(namespace, self.dest + '_arff', values[1])
        if len(values) >= 3:
            setattr(namespace, self.dest + '_vocab', values[2])


class ArffAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values[0])
        if len(values) > 1:
            setattr(namespace, self.dest + '_vocab', values[1])


def parse_arguments():
    parser = argparse.ArgumentParser(description=('Starts the classifier, or test the classifier if either of '
                                                  '--testcv, --testarff, or --testdir is specififed'),
                                     epilog=('Due to the structure of the optional arguments, please provide the '
                                             'classifier and term weighting scheme immediately after the program name')
                                     )
    parser.add_argument('classifier_name', choices=classifiers.keys(),
                        help='The name of classifier used to classify the instances')
    parser.add_argument('scorer_name', choices=scorers.keys(),
                        help='The term weighting scheme used to process the raw TF counts')
    parser.add_argument('-p', '--port', dest='port', default=8091, type=int,
                        help='Specify the port in which the server should start (default to 8091)')

    parser.add_argument("-P", "--thrift_port", dest="thrift_port", default=9090, type=int,
                        help="Specify the port in which Thrift server should start (default 9090)")

    parser.add_argument('--poslabel', dest='pos_label', default=None,
                        help='Specify the positive label in case of binary classification')
    parser.add_argument('--keywords', dest='keywords', nargs='+', default=[],
                        help='Provide a list of keywords which weight should be boosted')
    parser.add_argument('--nonkeywords', dest='nonkeywords', nargs='+', default=[],
                        help='Provide a list of non keywords which weight should be dampened')
    parser.add_argument('--badkeywords', dest='badkeywords', nargs='+', default=[],
                        help='Provide a list of bad keywords which weight should be negated')
    parser.add_argument('--reducefeatures', dest='reduce_features', action='store_true', default=False,
                        help='Whether feature dimensionality reduction should be performed')
    parser.add_argument('--topfeatures', dest='num_top_features', nargs='?', metavar='n', type=int, default=0,
                        help='Write the top-n features for each document to output file. Default to 8 top features')
    parser.add_argument('--numworkers', dest='num_workers', metavar='n', type=int, default=0,
                        help=('The number of workers that should be spawned if multiprocessing is required. '
                              'Default to the number of CPUs'))
    parser.add_argument('--lowercase', dest='lowercase', action='store_true', default=False,
                        help=('Whether the words should be converted to lower case. '
                              'This is not applied if ARFF is used'))
    parser.add_argument('--printkeywords', dest='print_keywords', action='store_true', default=True,
                        help='Whether to print the top words in each category to standard output')

    parser.add_argument('--word_normalization', dest='word_normalization', nargs="?", default="stem",
                        help="The use of lemmatization in preprocessing ")

    train_source = parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument('--traindir', dest='traindir', action=FileAction, nargs='+',
                              metavar=('traindir', 'arff_output [vocab_output]'),
                              help=('Training data comes from texts in a directory. This option can be specified '
                                    'multiple times. If arff_output is provided, the raw TF counts is written to '
                                    'arff_output. If vocab_output is also provided, the vocabulary is dumped into '
                                    'vocab_output. NOTE: If this option is specified multiple times, only the last '
                                    'arff_output and vocab_output will be used'))
    train_source.add_argument('--trainarff', dest='trainarff', action=ArffAction, nargs=2,
                              metavar=('trainarff', 'vocabulary_file'),
                              help='Training data comes from ARFF file, with the specified vocabulary')
    train_source.add_argument('--trainpickle', dest='trainpickle', action=ArffAction, nargs=2,
                              metavar=('train.pickle', 'vocabulary_file'),
                              help='Training data comes from pickled csr_matrix file, with the specified vocabulary')

    test_source = parser.add_mutually_exclusive_group()
    test_source.add_argument('--testcv', dest='testcv', nargs='?', action='append', type=int, metavar='n',
                             help='Perform an n-fold cross-validation on training data. Default to 10-fold.')
    test_source.add_argument('--testdir', dest='testdir', action=FileAction, nargs='+',
                             metavar=('testdir', 'arff_output'),
                             help=('Test from a folder. This option can be specified multiple times to include more '
                                   'directories. If arff_output is provided the raw TF counts is written to '
                                   'arff_output. NOTE: If this option is specified multiple times, only the last '
                                   'arff_output will be used'))
    test_source.add_argument('--testarff', dest='testarff', action=ArffAction, nargs='+',
                             metavar=('test.arff', 'vocabulary_file'),
                             help=('Test from ARFF file. This must have used the same vocabulary as the training data. '
                                   'If not, a vocabulary file of the original test data might be supplied, in which '
                                   'case the program will try to synchronize the two vocabularies. '
                                   'Note that while this synchronization is of high quality, is not 100%% accurate.'))

    return parser.parse_args(sys.argv[1:])


class DocumentClassifier(object):
    #########################################
    # Term weighting scheme helper function #
    #########################################

    def scorer_tf(self, doc_term_freq, labels=None, use_existing_data=False):
        return doc_term_freq

    def scorer_tfidf(self, doc_term_freq, labels=None, use_existing_data=False):
        if use_existing_data:
            (doc_term, _) = tf_to_tfidf(doc_term_freq, sublinear_tf=True, idf_diag=self.idf_diag)
        else:
            (doc_term, idf_diag) = tf_to_tfidf(doc_term_freq, sublinear_tf=True)
            self.idf_diag = idf_diag
        return doc_term

    def scorer_okapi(self, doc_term_freq, labels=None, use_existing_data=False):
        if use_existing_data:
            (doc_term, _, _) = tf_to_okapi(
                doc_term_freq, idfs=self.idfs, avg_doc_len=self.avg_doc_len)
        else:
            (doc_term, idfs, avg_doc_len) = tf_to_okapi(doc_term_freq)
            self.idfs = idfs
            self.avg_doc_len = avg_doc_len
        return doc_term

    def scorer_midf(self, doc_term_freq, labels=None, use_existing_data=False):
        (doc_term_freq_idf,) = tf_to_midf(doc_term_freq)
        return doc_term_freq_idf

    def scorer_rf(self, doc_term_freq, labels=None, use_existing_data=False):
        if use_existing_data:
            (doc_term, _) = tf_to_rf(doc_term_freq, rf_vectors=self.rf_vectors)
        elif labels is not None:
            (doc_term, rf_vectors) = tf_to_rf(doc_term_freq, labels=labels)
            self.rf_vectors = rf_vectors
        else:
            LOGGER.error('No label for each document or no existing data was given to RF')
            raise Exception(
                'DEV_ERROR:Label for each document or existing data is required to compute RF!')
        return doc_term

    ##################################
    # Feature Manipulation Utilities #
    ##################################
    def boost_keywords(self, doc_term):
        """Multiply each keyword score by 3
        """
        vocabulary = self.vocabulary
        keywords = self.keywords
        num_features = doc_term.shape[1]
        multiplier_matrix = lil_matrix((num_features, num_features), dtype=float)
        for i in range(num_features):
            multiplier_matrix[i, i] = 1.0
        dtf = self.dtf
        for keyword in keywords:
            keyword = dtf.get_postprocessor()(keyword)
            keyword_idx = vocabulary.get(keyword, None)
            if keyword_idx:
                multiplier_matrix[keyword_idx, keyword_idx] *= 3
        return doc_term * multiplier_matrix

    def dampen_nonkeywords(self, doc_term):
        """Multiply each keyword score by 1/3
        """
        vocabulary = self.vocabulary
        nonkeywords = self.nonkeywords
        num_features = doc_term.shape[1]
        multiplier_matrix = lil_matrix((num_features, num_features), dtype=float)
        for i in range(num_features):
            multiplier_matrix[i, i] = 1.0
        dtf = self.dtf
        for nonkeyword in nonkeywords:
            nonkeyword = dtf.get_postprocessor()(nonkeyword)
            nonkeyword_idx = vocabulary.get(nonkeyword, None)
            if nonkeyword_idx:
                multiplier_matrix[nonkeyword_idx, nonkeyword_idx] /= 3
        return doc_term * multiplier_matrix

    def negate_badkeywords(self, doc_term, labels):
        """Multiply the effect of specified keywords for articles with negative labels
        """
        vocabulary = self.vocabulary
        badkeywords = self.badkeywords
        pos_label = self.pos_label
        num_docs = doc_term.shape[0]
        num_features = doc_term.shape[1]
        multiplier_matrix = lil_matrix((1, num_features), dtype=float)
        for i in range(num_features):
            multiplier_matrix[0, i] = 1.0
        dtf = self.dtf
        for badkeyword in badkeywords:
            badkeyword = dtf.get_postprocessor()(badkeyword)
            badkeyword_idx = vocabulary.get(badkeyword, None)
            if badkeyword_idx:
                multiplier_matrix[0, badkeyword_idx] *= 3
        for row in range(num_docs):
            if labels[row] == pos_label:
                continue
            start = doc_term.indptr[row]
            end = doc_term.indptr[row + 1]
            doc_term.data[start:end] = doc_term.getrow(row).multiply(multiplier_matrix).data
        return doc_term

    def _get_class_to_feature_weights(self):
        '''Returns the mapping from each class label to feature weights
        '''
        fitted_classifier = self.fitted_classifier
        vocabulary = self.vocabulary
        if not (isinstance(fitted_classifier, svm.LinearSVC) or isinstance(fitted_classifier, OneVsRestClassifier)):
            raise Exception(
                'Only LinearSVC or OneVsRestClassifier is currently supported for top features extraction')
        if not hasattr(self, 'mapping') or self.mapping is None:
            self.mapping = {}
            for word, idx in vocabulary.items():
                self.mapping[idx] = word
        feature_weights = fitted_classifier.coef_
        class_to_feature_weights = {}
        classes = fitted_classifier.classes_
        if len(classes) == 2:
            pos_label = classes[1]
            neg_label = classes[0]
            class_to_feature_weights[pos_label] = coo_matrix(feature_weights,
                                                             shape=(1, len(feature_weights)))
            class_to_feature_weights[neg_label] = coo_matrix(-feature_weights,
                                                             shape=(1, len(feature_weights)))
        else:
            for idx, class_name in enumerate(classes):
                class_to_feature_weights[class_name] = coo_matrix(feature_weights[idx],
                                                                  shape=(1, len(feature_weights[idx])))
        return class_to_feature_weights

    def take_top_features_per_category(self, n=8):
        """Returns the top words in each category that likely have the most impact towards classification result
        """
        class_to_feature_weights = self._get_class_to_feature_weights()
        result = {}
        for class_name in class_to_feature_weights.keys():
            feature_weights = csr_matrix(class_to_feature_weights[class_name])
            top_feature_indices = zip(*sorted(zip(feature_weights.data, feature_weights.indices),
                                              key=lambda x: -x[0]))[1]
            result[class_name] = ['%s::%f' % (self.mapping[idx], feature_weights[0, idx])
                                  for idx in top_feature_indices[:n]]
        return result

    def take_top_features(self, doc_term, labels, n=8):
        """Returns the top words in each article that contribute the most to the classification result
        """
        class_to_feature_weights = self._get_class_to_feature_weights()
        num_workers = self.num_workers
        result = []
        while num_workers <= 0:
            num_workers += cpu_count()
        num_rows = len(labels) / num_workers
        workers = []
        result_pipes = []
        for i in range(num_workers):
            (parent, child) = Pipe()
            result_pipes.append(parent)
            worker = Process(target=self.process_top_feature, args=(class_to_feature_weights,
                                                                    self.mapping,
                                                                    doc_term,
                                                                    labels,
                                                                    i * num_rows, (i + 1) * num_rows, n, child))
            worker.start()
            workers.append(worker)
        for worker, result_pipe in zip(workers, result_pipes):
            result.extend(result_pipe.recv())
            worker.join()
        return result

    def process_top_feature(self, class_to_feature_weights, mapping, doc_term, labels, start, end, n=5,
                            result_pipe=None):
        """Worker function to produce top features
        """
        result = []
        for idx in range(start, end):
            label = labels[idx]
            feature_weights = doc_term.getrow(idx).multiply(class_to_feature_weights[label])
            top_feature_indices = zip(*sorted(zip(feature_weights.data, feature_weights.indices),
                                              key=lambda x: -x[0])
                                      )[1]
            result.append(['%s:%f' % (mapping[idx], feature_weights[0, idx]) for idx in top_feature_indices[:n]])
        if result_pipe is None:
            return result
        else:
            result_pipe.send(result)

    def reduce_features(self, doc_term, labels):
        LOGGER.warn('reduce_features is a stub. No feature dimensionality reduction has been performed')
        return doc_term

    ############################
    # Training data processing #
    ############################
    def train_on_directory(self):
        scorer_name = self.scorer_name
        traindir = self.traindir
        arff_output = self.traindir_arff
        vocab_output = self.traindir_vocab
        keywords = self.keywords
        nonkeywords = self.nonkeywords
        badkeywords = self.badkeywords
        pos_label = self.pos_label
        lowercase = self.lowercase

        with Timing('Processing training files in the folder %s...' % traindir, self.logging):
            dtf = DocToFeature(lowercase=lowercase, word_normalization=self.word_normalization)
            if type(traindir) == list:
                file_list = []
                for dirname in traindir:
                    if not dirname.endswith('/'):
                        dirname += '/'
                    file_list.extend((dirname + filename for filename in os.listdir(dirname)
                                      if filename != '.DS_Store'))
                traindir = file_list
            train_doc_term_freq = dtf.doc_to_tf(traindir)
            train_file_list = dtf.filelist
            vocabulary = dtf.vocabulary
            mapping = dtf.mapping
            labels = []
            train_classes = []
            for filename in train_file_list:
                true_filename = filename[filename.rfind('/') + 1:]
                label = true_filename[0:true_filename.find('_')]
                labels.append(label)
                if label not in train_classes:
                    train_classes.append(label)
            train_classes.sort()

        if arff_output is not None:
            with Timing('Dumping TF counts to %s...' % arff_output, self.logging):
                docs_arff = FeatureToArff(train_doc_term_freq, relation='TF.IDF')
                docs_arff.add_column(labels, name='LABEL', type_=train_classes)
                docs_arff.dump(arff_output, sparse=True)

            pickle_output = '%s.pickle' % arff_output[:arff_output.rfind('.')]
            with Timing('Pickling TF counts to %s...' % pickle_output, self.logging):
                def task(item, _pickle_output):
                    with open(_pickle_output, 'wb') as outfile:
                        pickle.dump(item, outfile, protocol=2)

                process = Process(target=task, args=((train_doc_term_freq, labels, train_classes), pickle_output))
                process.start()
                process.join()

            train_list_output = '%s.list' % arff_output[:arff_output.rfind('.')]
            with Timing('Writing file names into %s...' % (train_list_output), self.logging):
                with(open(train_list_output, 'w')) as filename_output:
                    for filename in train_file_list:
                        filename_output.write(filename + '\n')

        if vocab_output is not None:
            with Timing('Dumping vocabulary to %s...' % vocab_output, self.logging):
                with open(vocab_output, 'w') as vocab_output_file:
                    pickle.dump(vocabulary, vocab_output_file)

        if keywords:
            with Timing('Boosting weights of keywords %s...' % keywords, self.logging):
                train_doc_term_freq = self.boost_keywords(train_doc_term_freq)

        if nonkeywords:
            with Timing('Dampening weights of nonkeywords %s...' % nonkeywords, self.logging):
                train_doc_term_freq = self.dampen_nonkeywords(train_doc_term_freq)

        if badkeywords:
            if not pos_label:
                LOGGER.warn('badkeywords requires a positive class (as defined by --poslabel)')
            else:
                with Timing('Negating weights of badkeywords %s...' % badkeywords, self.logging):
                    train_doc_term_freq = self.negate_badkeywords(train_doc_term_freq, labels)

        with Timing('Calculating feature scores using scorer %s...' % scorer_name, self.logging):
            train_doc_term = self.get_scorer(scorer_name)(train_doc_term_freq, labels=labels)

        self.train_doc_term = train_doc_term
        self.train_labels = labels
        self.train_classes = train_classes
        self.vocabulary = vocabulary
        self.mapping = mapping

    def train_on_arff(self):
        scorer_name = self.scorer_name
        arff_file = self.trainarff
        vocab_file = self.trainarff_vocab
        keywords = self.keywords
        nonkeywords = self.nonkeywords
        badkeywords = self.badkeywords
        pos_label = self.pos_label

        with Timing('Loading training data from %s...' % (arff_file,), self.logging):
            (train_doc_term_freq, labels, train_classes) = loadarff(arff_file)
            self.train_classes = train_classes

        with Timing('Loading vocabulary from %s...' % vocab_file, self.logging):
            with open(vocab_file, 'rb') as vocab_file:
                self.vocabulary = pickle.load(vocab_file)

        if keywords:
            with Timing('Boosting weights of keywords %s...' % keywords, self.logging):
                train_doc_term_freq = self.boost_keywords(train_doc_term_freq)

        if nonkeywords:
            with Timing('Dampening weights of nonkeywords %s...' % nonkeywords, self.logging):
                train_doc_term_freq = self.dampen_nonkeywords(train_doc_term_freq)

        if badkeywords:
            if not pos_label:
                LOGGER.warn('badkeywords requires a positive class (as defined by --poslabel)')
            else:
                with Timing('Negating weights of badkeywords %s...' % badkeywords, self.logging):
                    train_doc_term_freq = self.negate_badkeywords(train_doc_term_freq, labels)

        with Timing('Calculating feature scores using scorer %s...' % scorer_name, self.logging):
            train_doc_term = self.get_scorer(scorer_name)(train_doc_term_freq, labels=labels)

        self.train_doc_term = train_doc_term
        self.train_labels = labels

    def train_on_pickle(self):
        scorer_name = self.scorer_name
        pickle_file = self.trainpickle
        vocab_file = self.trainpickle_vocab
        keywords = self.keywords
        nonkeywords = self.nonkeywords
        badkeywords = self.badkeywords
        pos_label = self.pos_label

        with Timing('Loading and processing training data from %s using scorer %s...' % (pickle_file, scorer_name),
                    self.logging):
            with open(pickle_file, 'rb') as infile:
                (train_doc_term_freq, labels, train_classes) = pickle.load(infile)
            self.train_classes = train_classes
        with Timing('Loading vocabulary from %s...' % vocab_file, self.logging):
            with open(vocab_file, 'rb') as vocab_file:
                self.vocabulary = pickle.load(vocab_file)

        if keywords:
            with Timing('Boosting weights of keywords %s...' % keywords, self.logging):
                train_doc_term_freq = self.boost_keywords(train_doc_term_freq)

        if nonkeywords:
            with Timing('Dampening weights of nonkeywords %s...' % nonkeywords, self.logging):
                train_doc_term_freq = self.dampen_nonkeywords(train_doc_term_freq)

        if badkeywords:
            if not pos_label:
                LOGGER.warn('badkeywords requires a positive class (as defined by --poslabel)')
            else:
                with Timing('Negating weights of badkeywords %s...' % badkeywords, self.logging):
                    train_doc_term_freq = self.negate_badkeywords(train_doc_term_freq, labels)

        with Timing('Calculating feature scores using scorer %s...' % scorer_name, self.logging):
            train_doc_term = self.get_scorer(scorer_name)(train_doc_term_freq, labels=labels)

        self.train_doc_term = train_doc_term
        self.train_labels = labels

    ####################
    # Cross validation #
    ####################
    def test_cv(self, cv=10):
        classifier_name = self.classifier_name
        train_doc_term = self.train_doc_term
        labels = self.train_labels
        pos_label = self.pos_label

        with Timing('Performing %d-fold cross validation...' % cv, self.logging):
            (single_label_scores, test_labels_list, predictions_list,
             multi_label_score, test_labels_for_multi_label,
             prediction_for_muli_label) = cross_validate(classifier_name, train_doc_term,
                                                         labels, cv=cv, pos_label=pos_label)

        accuracy = single_label_scores[:, 0]  # 0 is for acccuracy
        precision_scores = single_label_scores[:, 1]  # 1 is for precision score
        recall_scores = single_label_scores[:, 2]  # 2 is for recall score
        f1scores = single_label_scores[:, 3]  # 3 is for f1 score

        accuracy_for_multi_label = multi_label_score[:, 0]  # 0 is for accuracy
        precision_score_for_multi_label = multi_label_score[:, 1]  # 1 is for precision score
        recall_score_for_multi_label = multi_label_score[:, 2]  # 2 is for recall score
        f1score_for_multi_label = multi_label_score[:, 3]  # 3 is for f1 score

        print ('(SINGLE LABEL) Average accuracy from %d-fold cross-validation is: %.3f%% (+- %.3f%%)'
               % (cv, 100 * accuracy.mean(), 100 * accuracy.std() * 2))
        print ('(SINGLE LABEL) Average weighted precision-score from %d-fold cross-validation is: %.3f%% (+- %.3f%%)'
               % (cv, 100 * precision_scores.mean(), 100 * precision_scores.std() * 2))
        print ('(SINGLE LABEL) Average weighted recall-score from %d-fold cross-validation is: %.3f%% (+- %.3f%%)'
               % (cv, 100 * recall_scores.mean(), 100 * recall_scores.std() * 2))
        print ('(SINGLE LABEL) Average weighted f1-score from %d-fold cross-validation is: %.3f%% (+- %.3f%%)'
               % (cv, 100 * f1scores.mean(), 100 * f1scores.std() * 2))

        aggregated_test_labels = None
        aggregated_prediction = None

        for prediction_item in predictions_list:
            if aggregated_prediction is None:
                aggregated_prediction = prediction_item.tolist()
            else:
                aggregated_prediction = aggregated_prediction + prediction_item.tolist()

        for test_label_item in test_labels_list:
            if aggregated_test_labels is None:
                aggregated_test_labels = test_label_item.tolist()
            else:
                aggregated_test_labels = aggregated_test_labels + test_label_item.tolist()

        # calculate new configuration mat from here
        avg_new_conf_mat = numpy.array(metrics.confusion_matrix(aggregated_test_labels, aggregated_prediction))
        avg_conf_mat = numpy.around(avg_new_conf_mat / float(len(predictions_list)))
        final_classes = aggregated_prediction + aggregated_test_labels
        new_classes = sorted(list(set(final_classes)))

        print '(SINGLE LABEL) Averaged confusion matrix from %d-fold cross-validation:' % (cv,)
        print_confusion_matrix(avg_conf_mat, new_classes)
        # NEW METRICS
        print ('(MULTI LABEL) Average accuracy from %d-fold cross-validation is: %.3f%% (+- %.3f%%)' % (
            cv, 100 * accuracy_for_multi_label.mean(), 100 * accuracy_for_multi_label.std() * 2))
        print ('(MULTI LABEL) Average weighted precision score from %d-fold cross-validation is: %.3f%% (+- %.3f%%)' % (
            cv, 100 * precision_score_for_multi_label.mean(), 100 * precision_score_for_multi_label.std() * 2))
        print ('(MULTI LABEL) Average weighted recall score from %d-fold cross-validation is: %.3f%% (+- %.3f%%)' % (
            cv, 100 * recall_score_for_multi_label.mean(), 100 * recall_score_for_multi_label.std() * 2))
        print ('(MULTI LABEL) Average weighted f1-score from %d-fold cross-validation is: %.3f%% (+- %.3f%%)' % (
            cv, 100 * f1score_for_multi_label.mean(), 100 * f1score_for_multi_label.std() * 2))

        # CONFUSION MATRIX (MULTI LABEL)
        aggregated_test_labels_multiLabel = [item for sublist in test_labels_for_multi_label for item in sublist]
        aggregated_prediction_multiLabel = [item for sublist in prediction_for_muli_label for item in sublist]
        cm = ConfusionMatrix()
        len_result = len(aggregated_test_labels_multiLabel)
        for i in range(len_result):
            cm.add(aggregated_test_labels_multiLabel[i], aggregated_prediction_multiLabel[i])

        print('(MULTI LABEL) Confusion matrix:')
        print cm

        # Scores
        print "(MULTI LABEL) Scores for each category:"
        print metrics.classification_report(aggregated_test_labels_multiLabel, aggregated_prediction_multiLabel)

    #####################################
    # Learning model from training data #
    #####################################
    def fit_model(self):
        with Timing('Fitting classifier %s to data...' % self.classifier_name, self.logging):
            fitted_classifier = fit(self.classifier_name, self.train_doc_term, self.train_labels)
        self.fitted_classifier = fitted_classifier

    ###################
    # Testing process #
    ###################
    def test_on_directory(self):
        scorer_name = self.scorer_name
        testdir = self.testdir
        arff_output = self.testdir_arff
        keywords = self.keywords
        nonkeywords = self.nonkeywords
        lowercase = self.lowercase
        vocabulary = self.vocabulary

        if arff_output is not None:
            if arff_output.find('.') >= 0:
                self.prediction_output = '%s.out' % arff_output[:arff_output.rfind('.')]
            else:
                self.prediction_output = '%s.out' % arff_output
        else:
            path = testdir
            if type(testdir) is list:
                path = testdir[0]
            if path.find('/') >= 0:
                self.prediction_output = '%s.out' % path[path.rfind('/') + 1:]
            else:
                self.prediction_output = '%s.out' % path

        with Timing('Processing test files in the folder %s...' % testdir, self.logging):
            dtf = DocToFeature(lowercase=lowercase)
            if type(testdir) == list:
                file_list = []
                for dirname in testdir:
                    if not dirname.endswith('/'):
                        dirname += '/'
                    file_list.extend((dirname + filename for filename in os.listdir(dirname)
                                      if filename != '.DS_Store'))
                testdir = file_list
            test_doc_term_freq = dtf.doc_to_tf(testdir, vocabulary=vocabulary)
            test_file_list = dtf.filelist
            labels = []
            test_classes = []
            for filename in test_file_list:
                true_filename = filename[filename.rfind('/') + 1:]
                label = true_filename[0:true_filename.find('_')]
                labels.append(label)
                if label not in test_classes:
                    test_classes.append(label)
            test_classes.sort()

        if arff_output is not None:
            with Timing('Dumping TF counts to %s...' % arff_output, self.logging):
                docs_arff = FeatureToArff(test_doc_term_freq, relation='TF.IDF')
                docs_arff.add_column(labels, name='LABEL', type_=test_classes)
                docs_arff.dump(arff_output, sparse=True)

            test_list_output = '%s.list' % arff_output[:arff_output.rfind('.')]
            self.prediction_output = '%s.out' % arff_output[:arff_output.rfind('.')]
            with Timing('Writing file names into %s...' % (test_list_output), self.logging):
                with(open(test_list_output, 'w')) as filename_output:
                    for filename in test_file_list:
                        filename_output.write(filename + '\n')

        if keywords:
            with Timing('Boosting weights of keywords %s...' % keywords, self.logging):
                test_doc_term_freq = self.boost_keywords(test_doc_term_freq)

        if nonkeywords:
            with Timing('Dampening weights of nonkeywords %s...' % nonkeywords, self.logging):
                test_doc_term_freq = self.dampen_nonkeywords(test_doc_term_freq)

        if scorer_name == 'midf':
            mesg = 'Calculating feature scores using scorer %s...' % (scorer_name)
        else:
            mesg = ('Calculating feature scores using scorer %s '
                    'using estimated collection-specific information...' % (scorer_name)),
        with Timing(mesg, self.logging):
            test_doc_term = self.get_scorer(scorer_name)(test_doc_term_freq, use_existing_data=True)

        return (test_doc_term, labels)

    def test_on_arff(self):
        scorer_name = self.scorer_name
        arff_file = self.testarff
        vocab_file = self.testarff_vocab
        keywords = self.keywords
        nonkeywords = self.nonkeywords
        vocabulary = self.vocabulary
        train_classes = self.train_classes
        self.prediction_output = '%s.out' % arff_file[:arff_file.rfind('.')]

        with Timing('Loading test data from %s...' % arff_file, self.logging):
            (test_doc_term_freq, labels, test_classes) = loadarff(arff_file)

        if vocab_file is not None:
            with Timing('Reading vocabulary test file %s...' % vocab_file, self.logging):
                with open(vocab_file, 'rb') as vocab_file:
                    vocabulary_test = pickle.load(vocab_file)
            LOGGER.warn('Vocabulary for test data is provided, will try to synchronize vocabulary of test'
                        'data\n'
                        'This might not be exactly the same as running using --testdir')
            with Timing('Synchronizing vocabulary of test data to training data...', self.logging):
                conversion_map = {}
                for term, idx in vocabulary.items():
                    conversion_map[vocabulary_test.get(term, -1)] = idx
                # Filter (data,row,col) into three arrays of data, row, and col,
                # where col != -1
                [new_data, new_rows, new_cols] = map(
                    list,
                    zip(*[
                        (data, row, col)
                        for data, row, col in
                        zip(test_doc_term_freq.data,
                            test_doc_term_freq.row,
                            map(lambda x: conversion_map.get(x, -1), test_doc_term_freq.col))
                        if col != -1
                    ])
                )
                shape = (test_doc_term_freq.shape[0], len(vocabulary))
                test_doc_term_freq = coo_matrix((new_data, (new_rows, new_cols)), shape)
            [test_row_indices, labels] = zip(*[(i, item) for i, item in enumerate(labels) if item in train_classes])
            labels = list(labels)
            if len(labels) != test_doc_term_freq.shape[0]:
                LOGGER.warn('Number of classes in test data is larger than the number of classes in training data, '
                            'no output file will be produced')
                self.prediction_output = None
                with Timing('Removing documents in test data which class label is not present in training data...',
                            self.logging):
                    test_row_indices = list(test_row_indices)
                    test_doc_term_freq = test_doc_term_freq.tocsr()[test_row_indices, :].tocoo()

        if keywords:
            with Timing('Boosting weights of keywords %s...' % keywords, self.logging):
                test_doc_term_freq = self.boost_keywords(test_doc_term_freq)

        if nonkeywords:
            with Timing('Dampening weights of nonkeywords %s...' % nonkeywords, self.logging):
                test_doc_term_freq = self.dampen_nonkeywords(test_doc_term_freq)

        if scorer_name == 'midf':
            mesg = 'Calculating feature scores using scorer %s...' % (scorer_name)
        else:
            mesg = ('Calculating feature scores using scorer %s '
                    'using estimated collection-specific information...' % (scorer_name)),
        with Timing(mesg, self.logging):
            test_doc_term = self.get_scorer(scorer_name)(test_doc_term_freq, use_existing_data=True)

        return (test_doc_term, labels)

    def do_test(self, test_doc_term, labels):
        fitted_classifier = self.fitted_classifier
        pos_label = self.pos_label
        num_top_features = self.num_top_features
        prediction_output = self.prediction_output

        with Timing('Testing classifier %s using scorer %s...' % (self.classifier_name, self.scorer_name),
                    self.logging):
            try:
                ###########################################################
                # Main process: predict the labels of the given test data #
                ###########################################################
                label_score_prediction = predict(fitted_classifier, test_doc_term, pos_label)
                prediction = take_best_label(label_score_prediction)
            except ValueError:
                LOGGER.error('If the vocabulary size is different, you can either do:\n'
                             '\t- Use --testdir <dir> instead of --testarff, or\n'
                             '\t- Use --testarff <arff> <vocab> (i.e., supply the vocabulary file)\n'
                             'The first method is more accurate than the second one', exc_info=True)
                exit(1)
            if prediction_output is not None:
                if num_top_features > 0:
                    top_features = self.take_top_features(test_doc_term,
                                                          labels,
                                                          n=num_top_features)
                    with(open(prediction_output, 'w')) as prediction_file:
                        for label, top_feature in zip(prediction, top_features):
                            if type(label) is numpy.string_:
                                prediction_file.write(
                                    label + ' [' + '] ['.join(top_feature).encode('utf-8') + ']\n')
                            else:
                                prediction_file.write(
                                    ','.join(label) + ' [' + '] ['.join(top_feature).encode('utf-8') + ']\n')
                else:
                    with(open(prediction_output, 'w')) as prediction_file:
                        for label in prediction:
                            if type(label) is numpy.string_:
                                prediction_file.write(label + '\n')
                            else:
                                prediction_file.write(','.join(label) + '\n')
            incorrect = (prediction != labels).sum()
            correct = len(prediction) - incorrect

        print '###################################'
        print '# Correct classification:   %5d #' % correct
        print '# Incorrect classification: %5d #' % incorrect
        print '# Accuracy:               %6.2f%% #' % (100.0 * (float(correct) / len(prediction)))
        print '###################################'
        print metrics.classification_report(labels, prediction)
        if len(set(labels)) <= 2:
            f1_score = metrics.f1_score(labels, prediction, pos_label=pos_label)
            precision_score = metrics.precision_score(labels, prediction, pos_label=pos_label)
            recall_score = metrics.recall_score(labels, prediction, pos_label=pos_label)
        else:
            f1_score = metrics.f1_score(labels, prediction, average='weighted')
            precision_score = metrics.precision_score(labels, prediction, average='weighted')
            recall_score = metrics.recall_score(labels, prediction, average='weighted')

        prediction_label_set = set(prediction)
        prediction_label_set_length = len(prediction_label_set)

        print '(SINGLE LABEL) Weighted-average of Precision-Score: %.2f%%' % (100 * precision_score)
        print '(SINGLE LABEL) Weighted-average of Recall-Score: %.2f%%' % (100 * recall_score)
        print '(SINGLE LABEL) Weighted-average of F1-Score: %.2f%%' % (100 * f1_score)

        confusion_matrix = metrics.confusion_matrix(labels, prediction)
        classes = sorted(list(set(prediction)))
        print_confusion_matrix(confusion_matrix, classes)

        # NEW METRICS
        # Prepare data for multi label classification
        labels_for_multi_label = []
        for i in range(len(labels)):
            item = labels[i]
            sub_item = [item]
            labels_for_multi_label.append(sub_item)

        # Convert label_score_prediction into multi-label-array-format
        prediction_for_multi_label = []

        for i in range(len(label_score_prediction)):
            item = label_score_prediction[i]
            sub_item = []
            for label, score in item:
                sub_item.append(label)
            prediction_for_multi_label.append(sub_item)

        # Compute new metrics
        accuracy_multi_label = metrics.accuracy_score(labels_for_multi_label, prediction_for_multi_label)
        print '(MULTI LABEL) ACCURACY  %.2f%%' % (100 * accuracy_multi_label)
        recall_multi_label = metrics.recall_score(labels_for_multi_label, prediction_for_multi_label,
                                                  average='weighted')
        print '(MULTI LABEL) weighted recall  %.2f%%' % (100 * recall_multi_label)
        f1score_multi_label = metrics.f1_score(labels_for_multi_label, prediction_for_multi_label, average='weighted')
        print '(MULTI LABEL) weighted F1-Score : %.2f%%' % (100 * f1score_multi_label)

        # Confusion Matrix Multi Label
        cm = ConfusionMatrix()
        len_result = len(labels_for_multi_label)
        for i in range(len_result):
            cm.add(labels_for_multi_label[i], prediction_for_multi_label[i])
        print '(MULTI LABEL) Confusion Matrix'
        print cm
        print "(MULTI LABEL) Scores for each category:"
        print metrics.classification_report(labels_for_multi_label, prediction_for_multi_label)

    def __init__(self, classifier_name='LinearSVC', scorer_name='midf', traindir=None, traindir_arff=None,
                 traindir_vocab=None, trainarff=None, trainarff_vocab=None,
                 trainpickle=None, trainpickle_vocab=None,
                 testcv=None, testdir=None, testdir_arff=None, testarff=None, testarff_vocab=None,
                 pos_label=None, keywords=None, nonkeywords=None, badkeywords=None, print_keywords=False,
                 reduce_features=False, lowercase=False, num_top_features=8, num_workers=0, port=8091,
                 word_normalization="stem", logging=True, thrift_port=9090):
        for karg, value in locals().items():
            setattr(self, karg, value)
        self.fitted_classifier = None
        self.train_doc_term = None
        self.train_labels = None
        self.train_classes = None
        self.word_normalization = word_normalization
        self.dtf = DocToFeature(word_normalization=word_normalization)

    def initialize(self):
        if self.traindir:
            self.train_on_directory()
        elif self.trainarff:
            self.train_on_arff()
        elif self.trainpickle:
            self.train_on_pickle()
        else:
            raise Exception('No training data provided! Provide either a directory to traindir, '
                            'or an ARFF file and vocab file to trainarff and trainarff_vocab')

        # if self.reduce_features:
        #    self.reduce_features()

        self.fit_model()

    def check_initialized(self):
        if self.fitted_classifier is None:
            raise Exception(
                'Classifier has not been fitted! Run initialize() first')

    def run_test(self):
        self.check_initialized()
        testcv = self.testcv
        testdir = self.testdir
        testarff = self.testarff
        try:
            if testcv is not None:
                if str(testcv).isdigit():
                    self.test_cv(cv=int(testcv))
                    return True
                else:
                    self.test_cv()
                    return True
            elif testdir:
                (test_doc_term, test_labels) = self.test_on_directory()
            elif testarff:
                (test_doc_term, test_labels) = self.test_on_arff()

            if testdir or testarff:
                self.do_test(test_doc_term, test_labels)
                return True
        finally:
            if self.print_keywords:
                category_to_keywords = self.take_top_features_per_category(self.num_top_features)
                for category in category_to_keywords.keys():
                    print 'Top features for %s: [%s]' % (category,
                                                         '] ['.join(category_to_keywords[category]).encode('utf-8'))
        return False

    def classify_one(self, data):
        self.check_initialized()

        vector = self.dtf.str_to_tf(data, self.vocabulary)
        vector = self.get_scorer(self.scorer_name)(vector, use_existing_data=True)
        return predict(self.fitted_classifier, vector, self.pos_label)[0]

    def start_thrift_server(self):
        thrift_port = self.thrift_port
        handler = ThriftCategoryHandler(self.get_scorer(self.scorer_name), self.fitted_classifier, self.vocabulary,
                                        self.lowercase, self.word_normalization, self.pos_label)
        processor = Category.Processor(handler)

        transport = TSocket.TServerSocket(port=thrift_port)
        # tfactory = TTransport.TBufferedTransportFactory()
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()

        # You could do one of these for a multithreaded server
        # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
        # server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
        # server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)

        server = TNonblockingServer(processor, transport, pfactory, threads=10)

        LOGGER.info('Starting the TNonBlockingServer at port %s' % (thrift_port))
        server.serve()

    def start_tornado(self, basedir=os.path.dirname(__file__)):
        self.check_initialized()
        if basedir is None:
            raise Exception('No base directory for static files is given!')

        application = Application([
            (r'/', MainHandler),
            (r'/classify', ClassificationHandler, {'scorer': self.get_scorer(self.scorer_name),
                                                   'fitted_classifier': self.fitted_classifier,
                                                   'vocabulary': self.vocabulary,
                                                   'pos_label': self.pos_label,
                                                   'lowercase': self.lowercase,
                                                   'word_normalization': self.word_normalization,
                                                   }),
            (r'/static/(.*)', StaticFileHandler, {'path': os.path.join(basedir, 'static')}),
            (r'/test', TestHandler)
        ])

        # http_server = httpserver.HTTPServer(application)
        # http_server.listen(self.port, address="127.0.0.1")

        application.listen(self.port)
        print 'Tornado server started at port: %s' % self.port
        IOLoop.instance().start()

    def get_scorer(self, scorer_name):
        global scorers
        return getattr(self, scorers[scorer_name])


############################
# Web application handlers #
############################

class TestHandler(RequestHandler):
    def get(self, *args, **kwargs):
        self.write("Hello world")

    def post(self, *args, **kwargs):
        self.write("Hello")


class ClassificationHandler(RequestHandler):
    def initialize(self, scorer, fitted_classifier, vocabulary, pos_label=None, lowercase=False):
        self.scorer = scorer
        self.fitted_classifier = fitted_classifier
        self.vocabulary = vocabulary
        self.pos_label = pos_label
        self.lowercase = lowercase
        self.dtf = DocToFeature(lowercase=lowercase)

    def get(self):
        text = self.get_argument('data', default=None)
        callback = self.get_argument('callback', default=None)
        result = ''
        if text is not None:
            text = NormalizationText.preprocess(text)
            vector = self.dtf.str_to_tf(text, self.vocabulary)
            vector = self.scorer(vector, use_existing_data=True)
            result = json.dumps(predict(self.fitted_classifier, vector, self.pos_label)[0])
        else:
            result = ''

        self.set_header("Content-Type", "application/json")
        if callback:
            self.write('%s(%s)' % (callback, result))
        else:
            self.write(result)


class MainHandler(RequestHandler):
    def get(self):
        self.redirect('static/classify.html', permanent=True)


class ThriftCategoryHandler():
    def __init__(self, scorer, fitted_classifier, vocabulary, lowercase, word_normalization, pos_label):
        self.scorer = scorer
        self.fitted_classifier = fitted_classifier
        self.vocabulary = vocabulary
        self.dtf = DocToFeature(lowercase=lowercase, word_normalization=word_normalization)
        self.pos_label = pos_label
        self.lowercase = lowercase

    def ping(self):
        print 'ping()'

    def get_category(self, data):
        if self.lowercase:
            data = data.lower()
        text = NormalizationText.preprocess(data)
        vector = self.dtf.str_to_tf(text, self.vocabulary)
        vector = self.scorer(vector, use_existing_data=True)
        category_result = predict(self.fitted_classifier, vector, self.pos_label)[0]
        return {"label": category_result[0][0], "score": str(category_result[0][1])}

    def get_multi_category(self, data):
        if self.lowercase:
            data = data.lower()
        text = NormalizationText.preprocess(data)
        LOGGER.debug("Data to be classified: ", text)
        vector = self.dtf.str_to_tf(text, self.vocabulary)
        vector = self.scorer(vector, use_existing_data=True)
        multi_labels = predict(self.fitted_classifier, vector, self.pos_label)[0]

        category_result = []
        for item in multi_labels:
            category_result.append({"label": item[0], "score": str(item[1])})
        return category_result


def main():
    sys.stdout = Unbuffered(sys.stdout)
    parsed = parse_arguments()
    classifier = DocumentClassifier(**vars(parsed))
    classifier.initialize()
    if not classifier.run_test():
        classifier.start_thrift_server()


if __name__ == '__main__':
    main()
