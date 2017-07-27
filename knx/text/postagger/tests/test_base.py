from BS.knx.text import map_paren, reverse_map_paren
from nose.tools import assert_equal


def test_map():
    words = ['(', '{', '[', ')', '}', ']']
    mapped_words = ['-LRB-', '-LCB-', '-LSB-', '-RRB-', '-RCB-', '-RSB-']
    for word, mapped_word in zip(words, mapped_words):
        assert_equal(map_paren(word), mapped_word)
        assert_equal(word, reverse_map_paren(mapped_word))
    non_parens = ['hello', '$', 'Google', '1']
    for word in non_parens:
        assert_equal(map_paren(word), word)
        assert_equal(reverse_map_paren(word), word)
