"""
Tests for shipgrav.io
This does not test every option for read (because there are many ships)
but for each overall read function it reads a snippet of an example file
and checks that the values in key columns are correct
"""
import unittest
import shipgrav.io as sgi


class ioTestCase(unittest.TestCase):
    def test_read_nav(self):  # NOTE only tests one of the ship functions
        # but it does include extracting clock time -> datetime -> posix
        nav = sgi.read_nav('Thompson', 'ex_files/TN400_nav.Raw')
        self.assertEqual(nav.iloc[0].time_sec, 1647129603)
        self.assertTrue(nav.iloc[0].lon + 118.6524 < 0.001)

    def test_read_bgm_rgs(self):
        bgm = sgi.read_bgm_rgs('ex_files/AT05_01_bgm.RGS', 'Atlantis')
        self.assertEqual(bgm.iloc[0]['date_time'].timestamp(), 1656633600.445)
        self.assertEqual(bgm.iloc[0]['grav'], 980329.272)

    def test_read_bgm_raw(self):
        bgm = sgi.read_bgm_raw('ex_files/TN400_bgm.Raw', 'Thompson')
        self.assertEqual(bgm.iloc[0]['date_time'].timestamp(), 1647129602.449)
        self.assertEqual(bgm.iloc[0]['counts'], 25529)
        self.assertEqual(bgm.iloc[0]['rgrav'], 127730.60800402702)

    def test_read_dgs_dat(self):
        dgs = sgi.read_dgs_laptop('ex_files/DGStest_laptop.dat', 'DGStest')
        self.assertEqual(dgs.iloc[0]['date_time'].timestamp(), 1562803200.0)
        self.assertEqual(dgs.iloc[0]['ve'], 0.81098)
        self.assertTrue(dgs.iloc[0]['rgrav'] - 12295.691114 < 0.0001)

    def test_read_dgs_raw(self):
        dgs = sgi.read_dgs_raw('ex_files/SR2312_dgs_raw.txt', 'Ride')
        self.assertEqual(
            dgs.iloc[0]['date_time'].timestamp(), 1686873600.857719)
        self.assertEqual(dgs.iloc[0]['Gravity'], -218747)
        self.assertTrue(dgs.iloc[0]['vcc'] - 76.8771 < 0.0001)

    def test_read_mru(self):
        mru, cols = sgi.read_other_stuff(
            'ex_files/IXBlue.yaml', 'ex_files/SR2312_mru.txt', 'PASHR')
        self.assertEqual(mru.iloc[0]['Pitch:g'], -0.41)
        self.assertEqual(mru.iloc[0]['Roll:g'], 2.03)
        self.assertEqual(mru.iloc[0]['Heave:g'], -0.6)


def suite():
    return unittest.makeSuite(ioTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
