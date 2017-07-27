import sys

from nose.tools import assert_false
from nose.tools import assert_in
from nose.tools import assert_raises

from BS.knx.text.classifier import TheNationClassifier as TNC


def test_the_nation():
    tnc = TNC()
    try:
        tnc.classify_one('')
        assert_false('No Exception when classify_one is called before initialize')
    except Exception as e:
        assert_in('classifier has not been fitted', str(e).lower())

    tnc.initialize()
    result = tnc.classify_one('Property owners are rejoicing because of the rise in property prices')
    assert result[0][0] == 'property'

    result = tnc.classify_one('Condominiums, or condos, houses, residential housing are examples of property')
    assert result[0] == ('property', 1.0)

    tnc = TNC(logging=True)
    old_stdout = sys.stdout
    sys.stdout = None
    assert_raises(AttributeError, tnc.initialize)

    tnc = TNC(logging=False)
    try:
        tnc.initialize()
    except AttributeError:
        assert_false('Print when logging=False')

    sys.stdout = old_stdout

if __name__ == '__main__':
    test_the_nation()
