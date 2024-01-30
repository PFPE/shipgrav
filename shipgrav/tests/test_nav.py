"""
Tests for shipgrav.nav
"""
import unittest
import shipgrav.nav as sgn
import numpy as np

class navTestCase(unittest.TestCase):
    def setUp(self):
        lons = np.ones(100)*70
        lats = np.linspace(40,41,100)
        ve,vn = sgn.ll2en(lons,lats)   # lat/lon to easting and northing *velocities*
        self.ve = ve
        self.vn = vn

    def test_ll2en(self):
        self.assertTrue(self.ve[5] - 1121.5717 < 0.001)
        self.assertTrue(self.vn[5] < 1e-7)

    def test_en2cv(self):
        course, vel = sgn.vevn2cv(self.ve,self.vn)
        self.assertTrue(course[5] - 90. < 0.001)
        self.assertEqual(vel[5],self.ve[5])

    def test_rot_acc(self):
        course, vel = sgn.vevn2cv(self.ve,self.vn)
        eacc = 1e5*np.convolve(self.ve,sgn.tay10,'same')
        nacc = 1e5*np.convolve(self.vn,sgn.tay10,'same')
        cross,long = sgn.rot_acc_EN_cl(course, eacc, nacc)
        self.assertTrue(cross[10] < 1e-7)
        self.assertEqual(long[10],eacc[10])

def suite():
    return unittest.makeSuite(navTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
