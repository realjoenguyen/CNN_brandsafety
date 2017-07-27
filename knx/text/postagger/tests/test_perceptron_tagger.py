from nose.tools import assert_list_equal

from BS.knx.text.postagger import perceptron_tagger


def test_tagger():
    sentence = "I 'know' that one + one = two is true in my 'natural language' paradigm."
    result = [('I', 'PRP'), ('`', '``'), ('know', 'VBP'), ("'", "''"), ('that', 'WDT'), ('one', 'CD'), ('+', 'SYM'),
              ('one', 'CD'), ('=', 'SYM'), ('two', 'CD'), ('is', 'VBZ'), ('true', 'JJ'), ('in', 'IN'), ('my', 'PRP$'),
              ('`', '``'), ('natural', 'JJ'), ('language', 'NN'), ("'", "''"), ('paradigm', 'NN'), ('.', '.')]
    assert_list_equal(perceptron_tagger.tag(sentence), result)

    sentence = "'The dogs' movements are good."
    result = [('`', '``'), ('The', 'DT'), ('dogs', 'NNS'), ("'", "''"), ('movements', 'NNS'), ('are', 'VBP'),
              ('good', 'JJ'), ('.', '.')]
    assert_list_equal(perceptron_tagger.tag(sentence), result)

    sentence = ['The', 'book', 'is', 'here']
    result = [('The', 'DT'), ('book', 'NN'), ('is', 'VBZ'), ('here', 'RB')]
    assert_list_equal(perceptron_tagger.tag(sentence), result)
