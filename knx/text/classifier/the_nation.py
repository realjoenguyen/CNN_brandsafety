import logging
import os

import nltk
from knx.text.preprocess_text import NormalizationText as NT

from BS.knx.text.classifier import DocumentClassifier, predict

LOGGER = logging.getLogger(__name__)


class TheNationClassifier(DocumentClassifier):

    """Classifier trained on articles from The Nation (nationmultimedia.com)
    """

    def __init__(self, **kwargs):
        module_path = os.path.dirname(__file__)
        super(TheNationClassifier, self).__init__(classifier_name='LinearSVC', scorer_name='midf',
                                                  trainarff=os.path.join(module_path,
                                                                         ('training_data/'
                                                                          'thenation_business_reduced.arff')),
                                                  trainarff_vocab=os.path.join(module_path,
                                                                               ('training_data/'
                                                                                'thenation_business_reduced.vocab')),
                                                  pos_label='property',
                                                  keywords=['condominiums', 'condos', 'housing', 'property',
                                                            'resident'],
                                                  nonkeywords=['commission', 'government'], **kwargs)

    def classify_one(self, data):
        """Classify a text into either property or business

        This is a hybrid classifier that combines keyword-based and SVM-based classification.

        Returns:
            List of tuples - Represents the possible classes for the input, decreasing in score.
                Each tuple is in the form of (<label>,<score>)
                For extracting just the best label, use output[0][0]
        """
        if self.fitted_classifier is None:
            LOGGER.error('Classifier has not been fitted! Warned user to run initialize() first')
            raise Exception('Classifier has not been fitted! Run initialize() first')
        data = NT.preprocess(data)
        tokenized = nltk.wordpunct_tokenize(data.lower())
        keywords = {'condominium', 'condominiums', 'condo', 'condos',
                    'housing', 'house', 'houses', 'property', 'residential', 'residence'}
        keywords_count = 0
        for word in tokenized:
            if word in keywords:
                keywords_count += 1
        if keywords_count / float(len(tokenized)) >= 0.006 and keywords_count >= 5:
            return [('property', 1.0)]
        vector = self.dtf.str_to_tf(data, self.vocabulary)
        vector = self.get_scorer(self.scorer_name)(vector, use_existing_data=True)
        return predict(self.fitted_classifier, vector, self.pos_label)[0]
