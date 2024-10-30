"""
Tests for shipgrav
"""

import importlib.resources as importlib_resources
import sys
import unittest


def run():
    loader = unittest.TestLoader()
    ref = importlib_resources.files('shipgrav') / 'tests'
    with importlib_resources.as_file(ref) as path:
        suite = loader.discover(ref)
    runner = unittest.runner.TextTestRunner()  # verbosity=2)
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)
