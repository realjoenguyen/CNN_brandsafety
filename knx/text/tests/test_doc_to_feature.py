from BS.knx.text import DocToFeature as DTF
from BS.knx.text import NormalizationText as NT

from nose.tools import assert_raises, assert_equal


def test_doc_to_tf_empty_vocabulary():
    dtf = DTF()
    assert_raises(ValueError, dtf.doc_to_tf, [])


def test_preprocess():
    test_str = u'\x91\x92\x93\x94\x96\x97\xa0\u2013\u2014\u2018\u2019\u201c\u201d\x80\xce\xff'
    expected_str = u'\'\'""-- --\'\'""'
    result_str = NT.normalize_to_unicode(test_str)
    assert_equal(result_str, expected_str)


def test_str_to_tf():
    dtf = DTF()
    dtf.str_to_tf('This is a string to be vectorized', fit_vocabulary=True)


def test_tokenize():
    pass  # TODO call _tokenize with various parameter combinations

if __name__ == '__main__':
    for s in dir():
        if s.startswith('test_'):
            globals()[s]()
