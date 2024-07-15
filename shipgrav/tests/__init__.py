"""
Tests for shipgrav
"""

from pkg_resources import resource_filename
import sys
import unittest


def run():
    loader = unittest.TestLoader()
    test_dir = resource_filename('shipgrav', 'tests')
    suite = loader.discover(test_dir)
    runner = unittest.runner.TextTestRunner()  # verbosity=2)
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)
