"""
Tests for shipgrav.grav
"""
import unittest
import numpy as np
import shipgrav.grav as sgg


class gravNoDataTestCase(unittest.TestCase):
    def test_grav1d(self):
        # arbitrary numbers here, not real
        test_grav = sgg.grav1d_padded(np.linspace(
            0, 100, 10), np.linspace(10, 15, 10), 1, 0.4)
        self.assertTrue(test_grav[1] - 0.000157 < 0.001)

    def test_halfspace_T(self):
        T, W = sgg.therm_halfspace(np.array([20e3]), np.array([1e3]), u=0.02)
        self.assertTrue(T[0] - 135.2263 < 0.001)
        self.assertTrue(W[0] - 368.3393 < 0.001)

    def test_halfspace_Z(self):
        Z, W = sgg.therm_Z_halfspace(np.array([20e3]), 135.2263, u=0.02)
        self.assertTrue(Z[0] - 1e3 < 0.001)
        self.assertTrue(W[0] - 368.3393 < 0.001)

    def test_plate_T(self):
        T, W = sgg.therm_plate(np.array([20e3]), np.array([1e3]), u=0.02)
        self.assertTrue(T[0] - 134.0867 < 0.001)
        self.assertTrue(W[0] - 381.6513 < 0.001)

    def test_plate_Z(self):
        Z = sgg.therm_Z_plate(
            np.array([20e3]), np.array([134.0867]), u=0.02)
        self.assertEqual(Z[0], 1000.)

    def test_crustalthickness(self):
        rng = np.random.default_rng(123)  # seeded
        C = sgg.crustal_thickness_2D(10*rng.random(1000))
        self.assertTrue(np.real(C[0])[0] - 0.04019 < 0.001)


def suite():
    return unittest.makeSuite(gravNoDataTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
