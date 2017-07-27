from nose.tools import assert_list_equal

from BS.knx.text.chunker import MaxentNPChunker, RegexNPChunker, OpennlpNPChunker, RedshiftNPChunker


def test_chunker():
    sentence = 'I have found that studying natural language processing (NLP) makes you "a bit smarter".'
    conll = MaxentNPChunker()
    regex = RegexNPChunker()
    opennlp = OpennlpNPChunker()
    redshift = RedshiftNPChunker()
    result_conll = conll.chunk(sentence, output_tags=False)
    result_regex = regex.chunk(sentence, output_tags=False)
    result_opennlp = opennlp.chunk(sentence, output_tags=False)
    result_redshift = redshift.chunk(sentence, output_tags=False)

    expected_conll = ['I', 'natural language processing', 'NLP', 'you', 'a bit smarter']
    expected_regex = ['studying natural language processing', 'NLP', 'a bit']
    expected_opennlp = ['I', 'natural language processing', 'NLP -RRB-', 'you', 'a bit smarter \'\'']
    expected_redshift = ['language', 'processing', 'NLP', 'a bit smarter']

    assert_list_equal(result_conll, expected_conll)
    assert_list_equal(result_regex, expected_regex)
    assert_list_equal(result_opennlp, expected_opennlp)
    assert_list_equal(result_redshift, expected_redshift)

    result_conll = conll.chunk(sentence, output_tags=True)
    result_regex = regex.chunk(sentence, output_tags=True)
    result_opennlp = opennlp.chunk(sentence, output_tags=True)
    result_redshift = redshift.chunk(sentence, output_tags=True)

    expected_conll = ['I/PRP', 'natural/JJ language/NN processing/NN', 'NLP/NNP', 'you/PRP', 'a/DT bit/NN smarter/JJ']
    expected_regex = ['studying/VBG natural/JJ language/NN processing/NN', 'NLP/NNP', 'a/DT bit/NN']
    expected_opennlp = ['I/PRP', 'natural/JJ language/NN processing/NN', 'NLP/NNP -RRB-/-RRB-', 'you/PRP',
                        'a/DT bit/NN smarter/RBR \'\'/\'\'']
    expected_redshift = ['language/NN', 'processing/NN', 'NLP/NN', 'you/PRP ``/`` a/DT bit/NN smarter/NN']

    assert_list_equal(result_conll, expected_conll)
    assert_list_equal(result_regex, expected_regex)
    assert_list_equal(result_opennlp, expected_opennlp)
    assert_list_equal(result_redshift, expected_redshift)
