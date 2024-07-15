"""
Tests for shipgrav.grav
These use 1000 seconds of sample data to calculate corrections and check values
ccp coeffs are not objectively good because the time series is short but it works for tests
Not bothering with the leveling correction here because exactly how to calculate it
is left to the user
"""
import unittest
import numpy as np
import shipgrav.grav as sgg
import shipgrav.io as sgi
import shipgrav.nav as sgn
from glob import glob


class gravDataTestCase(unittest.TestCase):
    def setUp(self):  # actions to take before running each test: load in some test data
        # test data is part of one of the files supplied by DGS (original file is 30M/86400
        # lines for 24 hr; we cut to 1 hr?)
        gfiles = glob('ex_files/DGStest*.dat')
        data = sgi.read_dgs_laptop(gfiles, 'DGStest')
        data['tsec'] = [e.timestamp()
                        for e in data['date_time']]  # get posix timestamps
        data['grav'] = data['rgrav'] + 969143
        self.data = data

    def test_longman(self):
        lt = sgg.longman_tide_prediction(
            self.data['lon'], self.data['lat'], self.data['date_time'])
        self.assertTrue(lt[0] + 0.079599 < 0.001)

    def test_eotvos(self):
        eotvos = sgg.eotvos_full(self.data['lon'].values, self.data['lat'].values,
                                 np.zeros(len(self.data)), 1)
        self.assertTrue(eotvos[0] + 56.90367 < 0.001)

    def test_fa2ord(self):
        fa2 = sgg.free_air_second_order(
            self.data['lat'], np.zeros(len(self.data)))
        self.assertEqual(fa2.iloc[0], 0.)

    def test_up_vecs(self):
        lat_corr = sgg.wgs_grav(self.data['lat']) + sgg.free_air_second_order(self.data['lat'],
                                                                              np.zeros(len(self.data)))
        ve, vn = sgn.latlon_to_EN(
            self.data['lon'].values, self.data['lat'].values)
        eacc = 1e5*np.convolve(ve, sgn.tay10, 'same')
        nacc = 1e5*np.convolve(vn, sgn.tay10, 'same')
        crse, vel = sgn.ENvel_to_course_heading(ve, vn)
        acrss, along = sgn.rotate_acceleration_EN_to_cl(crse, eacc, nacc)
        up_vecs = sgg.up_vecs(1, lat_corr, acrss, along,
                              0, 240, 0.7071, 240, 0.7071)
        self.assertEqual(up_vecs[0, 0], 0.)
        self.assertEqual(up_vecs[2, 0], 1.)

    def test_ccpcalc(self):
        lt = sgg.longman_tide_prediction(
            self.data['lon'], self.data['lat'], self.data['date_time'])
        eotvos = sgg.eotvos_full(self.data['lon'].values, self.data['lat'].values,
                                 np.zeros(len(self.data)), 1)
        lat_corr = sgg.wgs_grav(self.data['lat']) + sgg.free_air_second_order(self.data['lat'],
                                                                              np.zeros(len(self.data)))
        faa = self.data['grav'] - lat_corr + eotvos + lt
        _, model = sgg.calc_cross_coupling_coefficients(faa, self.data['vcc'].values, self.data['ve'].values,
                                                        self.data['al'].values, self.data['ax'].values, np.zeros(len(self.data)))
        self.assertTrue(model.params.ax + 3.428237 < 0.0001)
        self.assertTrue(model.params.ve - 0.055732 < 0.0001)


def suite():
    return unittest.makeSuite(gravDataTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
