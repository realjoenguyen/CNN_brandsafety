import nltk
from itertools import chain
from BS.knx.text import map_paren, reverse_map_paren


def tag(text):
    """Returns the POS tags of the text using nltk POS tagger

    Parameters
    ----------
    text : str or iterable
        This is the text to be processed.
        If it's a str, it will be sentence tokenized and word tokenized using nltk
        If it's an iterable, it will be assumed to be a list of tokens

    Returns
    -------
    tags : list
        List of (word, pos) tuples
    """
    if type(text) in {str, unicode}:
        result = nltk.pos_tag(map(map_paren, chain.from_iterable(nltk.word_tokenize(sent)
                                                                 for sent in nltk.sent_tokenize(text))
                                  )
                              )
    else:
        result = nltk.pos_tag(map(map_paren, text))
    return map(lambda x: (reverse_map_paren(x[0]), x[1]), result)

if __name__ == '__main__':
    import time
    start_time = time.time()
    print tag('The horse raced past the barn fell (badly).')
    print 'Done tagging in %.3fs' % (time.time() - start_time)
    start_time = time.time()
    print tag(['The', 'horse', 'raced', 'past', 'the', 'barn', 'fell', '(', 'badly', ')', '.'])
    print 'Done tagging (tokenized) in %.3fs' % (time.time() - start_time)
    while True:
        sentence = raw_input('Enter a sentence: ')
        start_time = time.time()
        print tag(sentence)
        print 'Done in %.3fs' % (time.time() - start_time)
