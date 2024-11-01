"""
Tests for shipgrav.io
This does not test every option for read (because there are many ships)
but for each overall read function it reads a snippet of an example file
and checks that the values in key columns are correct
"""
import os
import unittest

import shipgrav.io as sgi


class ioTestCase(unittest.TestCase):
    @property
    def ex_files(self):
        return __file__.rsplit(os.sep, 1)[0] + os.sep + "ex_files"

    def test_read_nav_Thompson(self):
        nav = sgi.read_nav('Thompson', f'{self.ex_files}'+os.sep+'TN400_nav.Raw', progressbar=False)
        self.assertEqual(nav.iloc[0].time_sec, 1647129603)
        self.assertTrue(nav.iloc[0].lon + 118.6524 < 0.001)

    def test_read_nav_Atlantis(self):
        nav = sgi.read_nav('Atlantis', f'{self.ex_files}'+os.sep+'AT01_nav.gps', progressbar=False)
        self.assertEqual(nav.iloc[0].time_sec, 1656677853)
        self.assertTrue(nav.iloc[0].lon + 70.67185265 < 0.001)

    def test_read_nav_Langseth(self):
        nav = sgi.read_nav('Langseth', f'{self.ex_files}'+os.sep+'MGL2003_nav.y2020d244', progressbar=False)
        self.assertEqual(nav.iloc[0].time_sec, 1598832000.6212)
        self.assertTrue(nav.iloc[0].lon + 132.620965893 < 0.001)

    def test_read_nav_Revelle(self):
        nav = sgi.read_nav('Revelle', f'{self.ex_files}'+os.sep+'RR2212_nav.txt', progressbar=False)
        self.assertEqual(nav.iloc[0].time_sec, 1667606400)
        self.assertTrue(nav.iloc[0].lon + 119.3889582 < 0.001)

    def test_read_nav_Ride(self):
        nav = sgi.read_nav('Ride', f'{self.ex_files}'+os.sep+'SR2302_nav.raw',talker='INGGA', progressbar=False)
        self.assertEqual(nav.iloc[0].time_sec, 1674228947.583)
        self.assertTrue(nav.iloc[0].lon + 117.23672175 < 0.001)

    def test_read_nav_NBP(self):
        nav = sgi.read_nav('NBP', f'{self.ex_files}'+os.sep+'NBP_2301_nav.d013', progressbar=False)
        self.assertEqual(nav.iloc[0].time_sec, 1673568001.81)
        self.assertTrue(nav.iloc[0].lon + 179.7605851 < 0.001)

    def test_read_nav_nope(self):
        nav = sgi.read_nav('Titanic', f'{self.ex_files}'+os.sep+'TN400_nav.Raw', progressbar=False)
        self.assertEqual(nav, -999)

    def test_read_bgm_rgs(self):
        bgm = sgi.read_bgm_rgs(f'{self.ex_files}'+os.sep+'AT05_01_bgm.RGS', 'Atlantis', progressbar=False)
        self.assertEqual(bgm.iloc[0]['date_time'].timestamp(), 1656633600.445)
        self.assertEqual(bgm.iloc[0]['grav'], 980329.272)

    def test_read_bgm_raw_Atlantis(self):
        bgm = sgi.read_bgm_raw(f'{self.ex_files}'+os.sep+'AT01_bgm.BGM', 'Atlantis', progressbar=False)
        self.assertEqual(bgm.iloc[0]['date_time'].timestamp(), 1656677853.492)
        self.assertEqual(bgm.iloc[0]['counts'], 24963)
        self.assertEqual(bgm.iloc[0]['rgrav'], 124666.983189576)

    def test_read_bgm_raw_Thompson(self):
        bgm = sgi.read_bgm_raw(f'{self.ex_files}'+os.sep+'TN400_bgm.Raw', 'Thompson', progressbar=False)
        self.assertEqual(bgm.iloc[0]['date_time'].timestamp(), 1647129602.449)
        self.assertEqual(bgm.iloc[0]['counts'], 25529)
        self.assertEqual(bgm.iloc[0]['rgrav'], 127730.60800402702)

    def test_read_bgm_raw_Langseth(self):
        bgm = sgi.read_bgm_raw(f'{self.ex_files}'+os.sep+'MGL2003_bgm.y2020d244', 'Langseth', progressbar=False)
        self.assertEqual(bgm.iloc[0]['date_time'].timestamp(), 1598832000.3244)
        self.assertEqual(bgm.iloc[0]['counts'], 25229)
        self.assertEqual(bgm.iloc[0]['rgrav'], 126145.0)

    def test_read_bgm_raw_Revelle(self):
        bgm = sgi.read_bgm_raw(f'{self.ex_files}'+os.sep+'RR2212_bgm.txt', 'Revelle', progressbar=False)
        self.assertEqual(bgm.iloc[0]['date_time'].timestamp(), 1667841225.329611)
        self.assertEqual(bgm.iloc[0]['counts'], 24882)
        self.assertEqual(bgm.iloc[0]['rgrav'], 124405.2114591)

    def test_read_bgm_raw_nope(self):
        bgm = sgi.read_bgm_raw(f'{self.ex_files}'+os.sep+'TN400_bgm.Raw', 'Boaty McBoatface', progressbar=False)
        self.assertEqual(bgm, -999)

    def test_read_dgs_dat_general(self):
        dgs = sgi.read_dgs_laptop(f'{self.ex_files}'+os.sep+'DGStest_laptop.dat', 'DGStest', progressbar=False)
        self.assertEqual(dgs.iloc[0]['date_time'].timestamp(), 1562803200.0)
        self.assertEqual(dgs.iloc[0]['ve'], 0.81098)
        self.assertTrue(dgs.iloc[0]['rgrav'] - 12295.691114 < 0.0001)

    def test_read_dgs_dat_Thompson(self):
        dgs = sgi.read_dgs_laptop(f'{self.ex_files}'+os.sep+'TN400_dgs_proc.Raw', 'Thompson', progressbar=False)
        self.assertEqual(dgs.iloc[0]['date_time'].timestamp(), 1647101047.033)
        self.assertEqual(dgs.iloc[0]['ve'], 2e-5)
        self.assertTrue(dgs.iloc[0]['rgrav'] - 9995.95186 < 0.0001)

    def test_read_dgs_dat_nope(self):
        dgs = sgi.read_dgs_laptop(f'{self.ex_files}'+os.sep+'DGStest_laptop.dat', 'Katama', progressbar=False)
        self.assertEqual(dgs, -999)

    def test_read_dgs_raw_general(self):
        dgs = sgi.read_dgs_raw(f'{self.ex_files}'+os.sep+'SR2312_dgs_raw.txt', 'Ride', progressbar=False)
        self.assertEqual(
            dgs.iloc[0]['date_time'].timestamp(), 1686873600.857719)
        self.assertEqual(dgs.iloc[0]['Gravity'], -218747)
        self.assertTrue(dgs.iloc[0]['vcc'] - 76.8771 < 0.0001)

    def test_read_dgs_raw_Thompson(self):
        dgs = sgi.read_dgs_raw(f'{self.ex_files}'+os.sep+'TN400_dgs_raw.Raw', 'Thompson', progressbar=False)
        self.assertEqual(
            dgs.iloc[0]['date_time'].timestamp(), 1647101046.634)
        self.assertEqual(dgs.iloc[0]['Gravity'], -82)
        self.assertTrue(dgs.iloc[0]['vcc'] - 0.0357100 < 0.0001)

    def test_read_mru(self):
        mru, cols = sgi.read_other_stuff(
            f'{self.ex_files}'+os.sep+'IXBlue.yaml', f'{self.ex_files}'+os.sep+'SR2312_mru.txt', 'PASHR')
        self.assertEqual(mru.iloc[0]['Pitch:g'], -0.41)
        self.assertEqual(mru.iloc[0]['Roll:g'], 2.03)
        self.assertEqual(mru.iloc[0]['Heave:g'], -0.6)


def suite():
    return unittest.makeSuite(ioTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
