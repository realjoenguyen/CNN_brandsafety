import os

from nose.tools import assert_equal
from scipy.sparse import coo_matrix

from BS.knx.text.feature_to_arff import FeatureToArff, _escape_string


def _escape_newlines(string):
    return string.replace('\n', '\\n')


def test_escape_string():
    string = 'This is a _string_ to be replaced. Don\'t you know?'
    escaped = _escape_string(string)
    expected = 'This_is_a_\\_string\\__to_be_replaced._Don\\\'t_you_know?'
    assert_equal(escaped, expected)


def test_feature_to_arff():
    matrix = coo_matrix([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]])
    fta = FeatureToArff(matrix, relation='Cool relation', names={2: 'Third column'})
    fta.add_column([-1, -2, -3])
    fta.add_column(['row 1', 'row 2', 'row 3'], name='Row ID', type_='string')
    fta.add_column(['yes', 'yes', 'no'], name='Last Row', type_=['yes', 'no'])

    dense_filename = 'knx_text_test_feature_to_arff_dense.arff'
    fta.dump('/tmp/' + dense_filename, sparse=False)
    with open(os.path.join(os.path.dirname(__file__), dense_filename), 'r') as infile:
        expected = infile.read()
    with open('/tmp/' + dense_filename, 'r') as infile:
        dumped = infile.read()
    assert_equal(expected, dumped, 'Expected:\n%s\nGot:\n%s' % (_escape_newlines(expected), _escape_newlines(dumped)))

    fta = FeatureToArff(matrix, names={2: 'Third column'}, types={3: 'integer'})
    fta.add_column(['row 1', 'row 2', 'row 3'], name='Row ID', type_='string')
    fta.add_column(['yes', 'yes', 'no'], name='Last Row', type_=['yes', 'no'])
    sparse_filename = 'knx_text_test_feature_to_arff_sparse.arff'
    fta.dump('/tmp/' + sparse_filename, sparse=True)
    with open(os.path.join(os.path.dirname(__file__), sparse_filename), 'r') as infile:
        expected = infile.read()
    with open('/tmp/' + sparse_filename, 'r') as infile:
        dumped = infile.read()
    assert_equal(expected, dumped, 'Expected:\n%s\nGot:\n%s' % (_escape_newlines(expected), _escape_newlines(dumped)))

if __name__ == '__main__':
    test_escape_string()
    test_feature_to_arff()
