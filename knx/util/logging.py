#!/usr/bin/python
import time
import sys


class Timing(object):

    """A context manager for timing a block of code

    Parameters
    ----------
    message : string
        The message to be printed before executing the code.
    logging: boolean, True by default
        Whether to print any messages at all.

    Examples
    --------
    Use this object in a `with` statement like this:

        with Timing('Calculating the answer to everything...'):
            for i in range(42):
                num += 1

    which will print:

        Calculating the answer to everything... Done in 0.000s
    """

    def __init__(self, message='', logging=True):
        self.message = message
        self.logging = logging
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        if self.logging and self.message:
            print self.message,
            try:
                sys.stdout.flush()
            except AttributeError:
                pass
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if not self.logging:
            return
        print 'Done in %.3fs\n' % self.duration


class Unbuffered(object):

    """Print immediately on each call of print"""

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def print_error(string):
    sys.stderr.write('%s\n' % string)
