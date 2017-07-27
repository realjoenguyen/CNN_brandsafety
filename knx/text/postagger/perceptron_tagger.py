from functools import partial

import nltk
from knx.text.postagger.base import map_paren, reverse_map_paren

from BS.knx.text.tokenizer import default_tokenizer as tokenizer

try:
    from textblob_aptagger import PerceptronTagger
    perceptron_tagger = PerceptronTagger()

    SYMBOLS = {'@', '#', '%', '^', '*', '+', '=', '~'}
    # Replace the original tag method to support tokenized text

    def _tag(self, corpus, tokenize=True):
        """Tags a string `corpus`."""
        # Assume untokenized corpus has \n between sentences and ' ' between words
        s_split = nltk.sent_tokenize if tokenize else lambda text: [text]
        w_split = tokenizer.tokenize if tokenize else lambda sent: sent

        def split_sents(corpus):
            for s in s_split(corpus):
                yield map(map_paren, w_split(s))

        prev, prev2 = self.START
        has_open_left_single_quote = False
        tokens = []
        for words in split_sents(corpus):
            context = self.START + [self._normalize(w) for w in words] + self.END
            for i, word in enumerate(words):
                tag = self.tagdict.get(word)
                if not tag:
                    features = self._get_features(i, word, context, prev, prev2)
                    tag = self.model.predict(features)
                pos = tag
                if word in SYMBOLS:
                    pos = 'SYM'
                elif word == "'" and pos == 'POS' and has_open_left_single_quote:
                    pos = "''"
                    has_open_left_single_quote = False
                elif word == "'" and pos == "''":
                    has_open_left_single_quote = False
                elif word == '`' and pos == '``':
                    has_open_left_single_quote = True
                word = reverse_map_paren(word)
                tokens.append((word, pos))
                prev2 = prev
                prev = pos
        return tokens
    perceptron_tagger.tag = partial(_tag, perceptron_tagger)
except:  # pragma: no cover
    raise NotImplementedError('PerceptronTagger from textblob_aptagger does not exist!')


def tag(text):
    """Returns the POS tags of the text using PerceptronTagger

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
        return perceptron_tagger.tag(text, tokenize=True)
    else:
        return perceptron_tagger.tag(text, tokenize=False)

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
