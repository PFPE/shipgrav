"""
Tests for shipgrav.utils
"""
import unittest
import shipgrav.utils as sgu
import numpy as np


class utilsTestCase(unittest.TestCase):
    def test_gaussian_filter(self):
        # make a spike, filtfilt for no shift, check amplitude
        test = np.zeros(101)
        test[50] = 1
        test_fl = sgu.gaussian_filter(test, 10)
        test_ffll = sgu.gaussian_filter(test_fl[::-1], 10)
        self.assertTrue(test_ffll[50] - 0.12975 < 0.001)

    def test_status_decode(self):
        # make a code, decode it, check a few of the 16 status bits
        stat = 12345
        decoded = sgu.decode_dgs_status_bits(stat)
        self.assertEqual(decoded['clamp status'], '0')
        self.assertEqual(decoded['GPSsync'], '0')
        self.assertEqual(decoded['GPStime'], '1')
        self.assertEqual(decoded['feedback'], '1')


def suite():
    return unittest.makeSuite(utilsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
