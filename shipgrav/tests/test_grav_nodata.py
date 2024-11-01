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
        signal = 10*rng.random(1000)
        C = sgg.crustal_thickness_2D(signal)
        self.assertTrue(np.real(C[0])[0] - 0.04019 < 0.001)

        C2 = sgg.crustal_thickness_2D(signal, back=True)
        self.assertEqual(np.real(C[0])[0], np.real(C2[0][-1])[0])

    def test_grav2d_folding(self):
        rng = np.random.default_rng(123)  # seeded
        X = np.arange(5); Y = np.arange(5)
        Z = rng.random((5,5))
        sdat = sgg.grav2d_folding(X, Y, Z, 100, 100,  drho=0.6, dz=6000, ifold=True, npower=5)

        self.assertTrue(sdat[0,0] + 0.0043244755 < 0.001)

    def test_grav2d_layer(self):
        rng = np.random.default_rng(123)  # seeded
        rho = 1e4*rng.random((5,5))
        sdat = sgg.grav2d_layer_variable_density(rho, 100, 100, 3, 5)
        self.assertTrue(sdat[0,0] - 130.877533 < 0.001)

def suite():
    return unittest.makeSuite(gravNoDataTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
