import os

from nose.tools import assert_list_equal

from BS.knx.text.kte.extract import KeyTermsExtractor


def test_extractor():
    base_dir = os.path.dirname(__file__)
    kte = KeyTermsExtractor(trainarff=base_dir + '/articles_T.arff',
                            trainarff_vocab=base_dir + '/articles_T.vocab',
                            boost_method=2)
    kte.initialize()
    result = kte.extract(title='Google goes online',
                         data=('Google Inc. yesterday announced that it will go online on Thursday, '
                               '29 May 2014, where it will finally reveal what are their plans.'))

    expected = ['Google', 'Google Inc.', 'online', 'Thursday', 'yesterday', 'plan']
    assert_list_equal(result['phrases'], expected)

    kte = KeyTermsExtractor(trainarff=base_dir + '/articles_T.arff',
                            trainarff_vocab=base_dir + '/articles_T.vocab',
                            boost_method=0,
                            word_normalization='lemmatize')
    kte.initialize()
    result = kte.extract(data=('Google Inc. yesterday announced that it will go online on Thursday, '
                               '29 May 2014, where it will finally reveal what are their plans.'),
                         with_score=True)
    expected = [('Thursday', 0.31106565535934372), ('yesterday', 0.26572847018938262),
                ('Google Inc.', 0.23199999999999998), ('plan', 0.2061552852498359), ('online', 0.0)]
    assert_list_equal(result['phrases'], expected)
