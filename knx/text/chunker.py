import cPickle as pickle
import os

import nltk
from nltk.corpus import conll2000
from py4j.java_gateway import JavaGateway, GatewayClient

try:
    from redshift.parser import Parser as RedshiftParser
    from redshift.sentence import Input
    REDSHIFT_AVAILABLE = True
except Exception as e:
    REDSHIFT_AVAILABLE = False

from knx.util.logging import Timing
from BS.knx.text import postagger
from BS.knx.text import reverse_map_paren
import logging

DEBUG = False
LOGGER = logging.getLogger(__name__)
if DEBUG:
    LOGGER.setLevel(logging.DEBUG)

LEFT_PAREN = {'(', '{', '['}
RIGHT_PAREN = {')', '}', ']'}


class ConsecutiveNPChunkTagger(nltk.TaggerI):

    """Internal class used in MaxentNPChunker"""

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = _npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = _npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):

    """Internal class used in MaxentNPChunker"""

    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.util.conlltags2tree(conlltags)


def _npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i - 1]
    if i == len(sentence) - 1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i + 1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": _tags_since_dt(sentence, i)}


def _tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


class NPChunker(object):

    """Base class for noun phrase chunker objects

    Returns
    =======
    chunker : NPChunker
        NPChunker object, typically used by calling its chunk() method
    """

    def chunk(self, sentence, postagged=False, output_tags=False):
        """Chunk the given sentence, producing list of noun phrases

        Parameters
        ----------
        sentence : str or iterable
            The sentence to be parsed. If an iterable, it's assumed to be tokenized

        postagged : boolean, optional, default to False
            Whether the sentence is already POS tagged.
            If the input is a str, it's expected to be in the form of a sequence of "word/POS"
            If the input is an iterable, it's expected to be a list of (word, POS) tuples

        output_tags : boolean, optional, default to False
            Whether to output the POS tags along with the noun phrases

        Returns
        -------
        noun_phrases : list of str
            If output_tags is False, returns list of noun phrases as str
            If output_tags is True, returns list of noun phrases with the POS tag of each word in the format word/POS
        """
        return sentence


class MaxentNPChunker(NPChunker):

    """Train a maxent NP chunker based on CONLL dataset.

    Parameters
    ==========
    model_path : str
        Path to chunker.model, a pre-trained model.
        If not available this will train on CONLL corpus in NLTK package.
        Note that training on CONLL corpus might take quite a long time.

    Returns
    =======
    chunker : NPChunker
        NPChunker object, typically used by calling its chunk() method
    """

    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), 'models', 'chunker.model')):
        if os.path.exists(model_path):
            with Timing('Loading model from %s...' % model_path, DEBUG):
                with open(model_path, 'rb') as infile:
                    chunker = pickle.load(infile)
        else:
            with Timing('%s not found, training model...' % model_path, DEBUG):
                train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
                chunker = ConsecutiveNPChunker(train_sents)
            with Timing('Saving model to %s...' % model_path, DEBUG):
                with open(model_path, 'wb') as outfile:
                    pickle.dump(chunker, outfile)
        self.chunker = chunker

    def chunk(self, text, postagged=False, sent_tokenized=False, output_tags=False, split_words=False):
        if sent_tokenized:
            sentences = text
        else:
            sentences = nltk.sent_tokenize(text)
        result = []
        for sentence in sentences:
            if not postagged:
                # Not postagged, then do POS tagging, product a list of (word, pos) tuples
                sentence = postagger.tag(sentence)
            if isinstance(sentence, str) or isinstance(sentence, unicode):
                # The input must have been POS tagged (using word/POS format), for otherwise it would have been a list
                sentence = [word_pos.rsplit('/', 1) for word_pos in sentence.split(' ')]
            chunks = _take_np(self.chunker.parse(sentence), output_tags, split_words)
            if sent_tokenized:
                # Input is a list of sentences, output the chunks grouped by sentences
                result.append(chunks)
            else:
                # Input is a plain text, output the chunks as one long list
                result.extend(chunks)
        return result


class RegexNPChunker(NPChunker):

    """Train an NP chunker based on regex rules.

    Parameters
    ==========
    grammar : str
        Set of rules that matches NP chunks based on the POS tags.
        If None, will use the default grammar:

            'NP: {<CD>*((<RB.?|JJ.?|VBG><,>?)+<CC>)?<RB.?|JJ.?|VBG>*
                  (<NN|NNS>*<NNP|NNPS>+|<NN|NNS|VBG>*<NN|NNS><NN|NNS|VBG>*)}'

    Returns
    =======
    chunker : NPChunker
        NPChunker object, typically used by calling its chunk() method
    """

    def __init__(self, grammar='NP: {(<PDT>*<DT>)?<CD>*((<RB.?|JJ.?|VBG><,>?)+<CC>)?<RB.?|JJ.?|VBG>*'
                               '(<NN|NNS>*<NNP|NNPS>+|<NN|NNS|VBG>*<NN|NNS><NN|NNS|VBG>*)}'):
        self.chunker = nltk.chunk.RegexpParser(grammar)

    def chunk(self, text, postagged=False, sent_tokenized=False, output_tags=False, split_words=False):
        if sent_tokenized:
            sentences = text
        else:
            sentences = nltk.sent_tokenize(text)
        result = []
        for sentence in sentences:
            if not postagged:
                # Not postagged, then do POS tagging, product a list of (word, pos) tuples
                sentence = postagger.tag(sentence)
            if type(sentence) in {str, unicode}:
                # The input must have been POS tagged then, since otherwise it would have been a list
                sentence = [word_pos.rsplit('/', 1) for word_pos in sentence.split(' ')]
            chunks = _take_np(self.chunker.parse(sentence), output_tags, split_words)
            if sent_tokenized:
                # Input is a list of sentences, output the chunks grouped by sentences
                result.append(chunks)
            else:
                # Input is a plain text, output the chunks as one long list
                result.extend(chunks)
        return result


class OpennlpNPChunker(NPChunker):

    """Returns a chunker object connecting to sentence-parser service

    Parameters
    ==========
    port : int, optional
        The port used to connect to sentence-parser service

    Returns
    =======
    chunker : NPChunker
        NPChunker object, typically used by calling its chunk() method
    """

    def __init__(self, port=48110):
        try:
            gateway = JavaGateway(GatewayClient(port=port), auto_convert=True, auto_field=True)
        except:
            raise NotImplementedError(('sentence-parser is either not running or not running in port {}.\n'
                                       'Cannot use Opennlp chunker').format(port))
        self.chunker = gateway.entry_point
        result = self.chunker.initialize()
        if result not in (0, 1):
            raise Exception('Error during sentence-parser initialization. Check logs')

    def chunk(self, text, postagged=False, sent_tokenized=False, output_tags=False, split_words=False):
        if sent_tokenized:
            sentences = text
        else:
            sentences = nltk.sent_tokenize(text)
        result = []
        for sentence in sentences:
            if type(sentence) in {str, unicode}:
                # Not tokenized as list, but perhaps POS tagged in the format word/POS
                chunks = list(self.chunker.chunk(sentence,
                                                 False,  # Not tokenized
                                                 postagged,
                                                 output_tags))
            else:
                # Sentence is tokenized as list of tokens
                if type(sentence[0]) in {tuple, list}:
                    # Each token is a tuple, must be in the (word, POS) format
                    chunks = list(self.chunker.chunk(' '.join(word + '/' + pos for word, pos in sentence),
                                                     True,  # Tokenized
                                                     True,  # POS tagged
                                                     output_tags))
                else:
                    # Each token is a string, but perhaps POS tagged in the format word/POS
                    chunks = list(self.chunker.chunk(' '.join(sentence),
                                                     True,  # Tokenized
                                                     postagged,
                                                     output_tags))
            if split_words:
                if output_tags:
                    chunks = [[tuple(word_pos.rsplit('/')) for word_pos in chunk.split(' ')]
                              for chunk in chunks]
                else:
                    chunks = [[word for word in chunk.split(' ')] for chunk in chunks]
            if sent_tokenized:
                # Input is a list of sentences, output the chunks grouped by sentences
                result.append(chunks)
            else:
                # Input is a plain text, output the chunks as one long list
                result.extend(chunks)
        return result


class RedshiftNPChunker(NPChunker):

    """Train a beam-based NP chunker based on Treebank dataset

    Parameters
    ==========
    model_path : str
        Path to redshift_model folder, containing required files:
            - config.json
            - labels
            - model
            - pos
            - tagger
            - tagger.json

    Returns
    =======
    chunker : NPChunker
        NPChunker object, typically used by calling its chunk() method
    """

    def __init__(self, model_dir=os.path.join(os.path.dirname(__file__), 'models', 'redshift_model')):
        if not REDSHIFT_AVAILABLE:
            raise NotImplementedError('Redshift not installed. RedshiftNPChunker is unavailable')
        self.parser = RedshiftParser(model_dir)

    def chunk(self, text, postagged=False, sent_tokenized=False, output_tags=False, split_words=False):
        if sent_tokenized:
            sentences = text
        else:
            sentences = nltk.sent_tokenize(text)
        if not postagged:
            sentences = [' '.join('/'.join(word_pos)
                                  for word_pos in postagger.tag(sent))
                         for sent in sentences]
        else:
            # Sentences are postagged. It can be ['sent/NN 1/CD ./.'] format (no change required) or
            # [('sent','NN'), ('1','CD'), ('.','.')] (change to the earlier format is required)
            if len(sentences) > 0 and not (isinstance(sentences[0], str) or isinstance(sentences[0], unicode)):
                sentences = [' '.join('/'.join(word_pos)
                                      for word_pos in sent)
                             for sent in sentences]
        # Convert into Redshift sentence object
        sentences = [Input.from_pos(sent) for sent in sentences]
        for sentence in sentences:
            # This will store the depparse result in each sentence object
            self.parser.parse(sentence)
        result = []
        for sentence in sentences:
            chunks = []
            if split_words:
                noun_phrase = []
            else:
                noun_phrase = ''
            noun_head_idx = None
            #length = sentence.length
            for token in reversed(list(sentence.tokens)):
                idx = token.id
                word = token.word
                pos = token.tag
                parent = token.head
                #rel = token.label
                word = reverse_map_paren(word)
                if word in RIGHT_PAREN:
                    continue
                if parent == noun_head_idx and word not in LEFT_PAREN:
                    if output_tags:
                        if split_words:
                            noun_phrase[0:0] = (str(word), str(pos))
                        else:
                            noun_phrase = str(word) + '/' + str(pos) + ' ' + noun_phrase
                    else:
                        if split_words:
                            noun_phrase[0:0] = str(word)
                        else:
                            noun_phrase = str(word) + ' ' + noun_phrase
                else:
                    if noun_phrase:
                        chunks[0:0] = [noun_phrase]
                        noun_phrase = None
                        noun_head_idx = None
                    if pos.startswith('NN'):
                        if output_tags:
                            if split_words:
                                noun_phrase = [(str(word), str(pos))]
                            else:
                                noun_phrase = str(word) + '/' + str(pos)
                        else:
                            if split_words:
                                noun_phrase = [str(word)]
                            else:
                                noun_phrase = word
                        noun_head_idx = idx
            if noun_phrase:
                chunks[0:0] = [noun_phrase]
            if sent_tokenized:
                # Input is a list of sentences, output the chunks grouped by sentences
                result.append(chunks)
            else:
                # Input is a plain text, output the chunks as one long list
                result.extend(chunks)
        return result


class DefaultNPChunker(MaxentNPChunker):

    """The class representing the default NP chunker (currently MaxentNPChunker)"""
    pass


def _take_np(tree_obj, output_tags=False, split_words=False):
    if output_tags:
        if not split_words:
            return filter(None, [' '.join(word + '/' + pos for word, pos in tag if word not in RIGHT_PAREN)
                                 for tag in tree_obj if hasattr(tag, 'pos')
                                                        and not (len(tag) == 1 and tag[0][1] == 'IN')])
        else:
            return filter(None, [[(word, pos) for word, pos in tag if word not in RIGHT_PAREN]
                                 for tag in tree_obj if hasattr(tag, 'pos')
                                                        and not(len(tag) == 1 and tag[0][1] == 'IN')])
    else:
        if not split_words:
            return filter(None, [' '.join(word for word, pos in tag if word not in RIGHT_PAREN)
                                 for tag in tree_obj if hasattr(tag, 'pos')
                                                        and not (len(tag) == 1 and tag[0][1] == 'IN')])
        else:
            return filter(None, [[word for word, pos in tag if word not in RIGHT_PAREN]
                                 for tag in tree_obj if hasattr(tag, 'pos')
                                                        and not(len(tag) == 1 and tag[0][1] == 'IN')])


def main():
    import sys
    from unidecode import unidecode
    if len(sys.argv) > 1 and sys.argv[1] == '1':
        output_tags = True
    else:
        output_tags = False
    sentence = ('The health app market currently is worth about $718 million and is expected to double by the end of '
                'the year, according to Research2Guidance, a global mobile research group. ')
    conll = MaxentNPChunker()
    regex = RegexNPChunker()
    opennlp = OpennlpNPChunker()
    redshift = RedshiftNPChunker()
    while True:
        with Timing('Testing a sentence using CONLL...'):
            result_conll = conll.chunk(sentence, output_tags=output_tags)
        with Timing('Testing a sentence using Regex...'):
            result_regex = regex.chunk(sentence, output_tags=output_tags)
        with Timing('Testing a sentence using Opennlp...'):
            result_opennlp = opennlp.chunk(sentence, output_tags=output_tags)
        with Timing('Testing a sentence using Redshift...'):
            result_redshift = redshift.chunk(sentence, output_tags=output_tags)
        print 'CONLL Chunker:'
        print '\n'.join(result_conll)
        print
        print 'Regex Chunker:'
        print '\n'.join(result_regex)
        print
        print 'Opennlp Chunker:'
        print '\n'.join(result_opennlp)
        print
        print 'Redshift Chunker:'
        print '\n'.join(result_redshift)
        sentence = unidecode(raw_input('Enter a sentence: ').decode('UTF-8'))

if __name__ == '__main__':
    main()
