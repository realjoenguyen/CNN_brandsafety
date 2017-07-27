#!/usr/bin/python
from scipy.sparse import coo_matrix
import re
import logging

LOGGER = logging.getLogger(__name__)


def _escape_string(string):
    return re.sub('([_\'])', r'\\\1', str(string)).replace(' ', '_')


class FeatureToArff:
    docsScore = None
    hasString = False
    additionalColumns = {}
    names = {}
    types = {}
    isNominal = {}
    nrow = 0
    ncol = 0

    def _set_type(self, colNum, type_):
        if type(type_) is list:
            self.types[colNum] = '{%s}' % ','.join([str(obj) for obj in type_])
            self.isNominal[colNum] = True
        else:
            self.types[colNum] = type_

    def _get_repr(self, rowNum, colNum, obj):
        if obj is None:
            obj = self.additionalColumns[colNum][rowNum]
        if colNum in self.types and self.types[colNum] is 'string':
            return "'%s'" % _escape_string(obj)
        else:
            return str(obj)

    def _get_sparse_repr(self, rowNum, colNum, obj):
        if obj is None:
            obj = self.additionalColumns[colNum][rowNum]
        if colNum in self.types and self.types[colNum] is 'string':
            return "%d '%s'" % (colNum, _escape_string(obj))
        else:
            return "%d %s" % (colNum, str(obj))

    def __init__(self, docsScore, relation='unknown', names=None, types=None):
        """Initialize this instance with the data given in coo_matrix format from scipy.sparse

        Parameters
        ----------
        relation : string, 'unknown' by default
            The relation name to be put in the ARFF file.

        names : dict from int to str
            Mapping from column index to column (attribute) names used to give custom names to attributes.
            By default the attributes will be named as 'attr<column_index>'

        types : dict from int to str or list
            Mapping from column index to attribute type used to give custom type.
            To set a nominal type, provide a list of possible nominal types
            By default the attributes will have the type 'numeric'

        Example
        -------
            FeatureToArff(docsTfidf, relation='tfidf_relation', names={0:'name',103:'weight',104:'label'},
                          types={0:'string',104:['yes','no']} )
        """
        self.docsScore = docsScore
        (self.nrow, self.ncol) = docsScore.shape
        self.relation = relation
        self.names = {}
        self.types = {}
        self.additionalColumns = {}
        self.isNominal = {}
        if names is not None:
            for colNum, name in names.iteritems():
                self.names[colNum] = name
        if types is not None:
            for colNum, type_ in types.iteritems():
                self._set_type(colNum, type_)

    def add_column(self, columnData, name=None, type_='numeric'):
        self.additionalColumns[self.ncol] = columnData
        if name is None:
            self.names[self.ncol] = 'attr%d' % self.ncol
        else:
            self.names[self.ncol] = name
        self._set_type(self.ncol, type_)
        if type_ is 'string':
            self.hasString = True
        self.ncol = self.ncol + 1

    def dump(self, filename, sparse=True):
        """Print the feature matrix into ARFF file given to filename

        sparse - Whether the output ARFF file should be in sparse ARFF format
        """
        if sparse and self.hasString:
            LOGGER.warn('\n'
                        '####################################################\n'
                        '# WARNING: string attribute in sparse ARFF format. #\n'
                        '# This may cause some problems if the string       #\n'
                        '# contains quotes                                  #\n'
                        '# Please check the output file                     #\n'
                        '####################################################')
        out = file(filename, 'w')
        out.write('@relation %s\n\n' % self.relation)

        for i in range(self.ncol):
            if i not in self.names:
                out.write('@attribute attr%d ' % i)
            else:
                out.write('@attribute %s ' % self.names[i])
            if i not in self.types:
                out.write('numeric\n')
            else:
                out.write(self.types[i] + '\n')

        out.write('\n@data\n')

        if sparse:
            docsScoreCSR = self.docsScore.tocsr()
            indptr = docsScoreCSR.indptr
            indices = docsScoreCSR.indices
            data = docsScoreCSR.data
            for i in range(self.nrow):
                rowLen = indptr[i + 1] - indptr[i] + len(self.additionalColumns)
                additionalIndices = range(self.docsScore.shape[1], self.ncol)
                rowIndices = list(indices[indptr[i]:indptr[i + 1]]) + additionalIndices
                row = sorted(map(self._get_sparse_repr,
                                 [i] * rowLen,
                                 rowIndices,
                                 data[indptr[i]:indptr[i + 1]]),
                             key=lambda x: int(x.split(' ')[0]))
                out.write('{%s}\n' % ','.join(row))
        else:
            for i in range(self.nrow):
                row = map(self._get_repr,
                          [i] * self.ncol,
                          range(self.ncol),
                          self.docsScore.getrow(i).todense().tolist()[0])
                out.write(','.join(row) + '\n')

        out.close()


def main():
    # Example
    scores = FeatureToArff(coo_matrix([[0, 1, 0, 0, 0],
                                       [0, 0, 2, 2, 0],
                                       [3, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 5]]),
                           relation='test_relation',
                           names={4: 'ID'},
                           types={3: 'integer'})
    scores.add_column(['yes', 'yes', 'no', 'no', 'no'], type_=['yes', 'no'])
    scores.dump('test.arff', sparse=True)

if __name__ == '__main__':
    main()
