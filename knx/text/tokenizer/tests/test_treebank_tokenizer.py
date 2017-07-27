from nose.tools import assert_list_equal

from BS.knx.text.tokenizer import treebank_tokenizer


def test_treebank_tokenizer():
    sentence = "Who'd have thought that 'that' thing, that you gave me (without notice), contains my dogs' \"thing\"!"
    result = ['Who', "'d", 'have', 'thought', 'that', '`', 'that', "'", 'thing', ',', 'that', 'you', 'gave', 'me',
              '(', 'without', 'notice', ')', ',', 'contains', 'my', 'dogs', "'", '``', 'thing', "''", '!']
    assert_list_equal(result, treebank_tokenizer.tokenize(sentence))
