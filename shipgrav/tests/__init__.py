"""
Tests for shipgrav
"""

import sys
import unittest
import importlib.resources as importlib_resources


def run():
    loader = unittest.TestLoader()
    ref = importlib_resources.files('shipgrav') / 'tests'
    with importlib_resources.as_file(ref) as path:
        suite = loader.discover(ref)
    runner = unittest.runner.TextTestRunner()  # verbosity=2)
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)
