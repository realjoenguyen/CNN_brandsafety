import logging
LOGGER = logging.getLogger(__name__)
try:
    # from .import stanford_tagger as postagger
    # LOGGER.debug('Use stanford_tagger')
    from . import perceptron_tagger as postagger
    LOGGER.debug('Use perceptron_tagger')
except:
    from . import nltk_tagger as postagger
    LOGGER.debug('Use nltk_tagger')


def tag(text):
    """Returns the POS tags of the text, using the default POS tagger (currently PerceptronTagger)

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
    return postagger.tag(text)

if __name__ == '__main__':
    import time
    start_time = time.time()
    print tag('The horse raced past the barn fell.')
    print 'Done tagging in %.3fs' % (time.time() - start_time)
    start_time = time.time()
    print tag(['The', 'horse', 'raced', 'past', 'the', 'barn', 'fell', '.'])
    print 'Done tagging (tokenized) in %.3fs' % (time.time() - start_time)
    while True:
        sentence = raw_input('Enter a sentence: ')
        start_time = time.time()
        print tag(sentence)
        print 'Done in %.3fs' % (time.time() - start_time)
