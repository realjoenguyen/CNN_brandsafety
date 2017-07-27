import sys
import time
from knx.util.logging import Timing, Unbuffered
from StringIO import StringIO


def test_timing():
    old_stdout = sys.stdout
    test_stdout = StringIO()
    test_message = 'A message...'
    try:
        sys.stdout = test_stdout
        with Timing(test_message, True) as timing:
            assert timing is not None
            assert test_stdout.getvalue() == test_message
            time.sleep(1)
            assert time.time() - timing.start_time >= 1.0
    finally:
        sys.stdout = old_stdout

    try:
        test_stdout = StringIO()
        sys.stdout = test_stdout
        with Timing('Should not be printed!', False) as timing:
            assert timing is not None
    finally:
        sys.stdout = old_stdout
    assert test_stdout.getvalue() == ''


def test_unbuffered():
    old_stdout = sys.stdout
    try:
        result = []
        sys.stdout.flush = lambda: result.append(1)
        sys.stdout = Unbuffered(sys.stdout)
        print 'This will be printed immediately',
    finally:
        sys.stdout = old_stdout
    assert len(result) > 0
