import numpy as np
import shipgrav.grav as sgg
import shipgrav.nav as sgn
import shipgrav.io as sgi
import shipgrav.utils as sgu
import tomli as tm
from pandas import concat, to_datetime
from glob import glob
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt, coherence
import matplotlib.pyplot as plt
import os
import sys

########################################################################
# Example script for reading DGS laptop data and MRUs from an R/V Ride
# transit, and calculating coherence between MRUs and monitors
#
# Read DGS laptop and navigation files
# Correct for meter bias with info from shipgrav
# Use timestamps to sync more accurate nav with the gravity data.
# Calculate FAA (free air anomaly) for laptop data
# Read MRUs (pitch/roll/heave)
# for all pairs of monitors and MRUs, interpolate
# MRU to grav sample rate and calculate coherence
# between monitor and 1000-pt moving average of MRU
# (and plot coherence)
########################################################################

# set some general metadata
ship = 'Ride'
cruise = 'SR2312'       # this is used for filepaths
sampling = 1            # 1 Hz - data should be at this rate already

# read a few constants etc from our toml database file
with open('../shipgrav/database.toml', 'rb') as f:
    info = tm.load(f)
nav_tag = info['nav-talkers'][ship]
biases = info['bias-values'][ship]

# set up file paths, get lists of input files
root = 'data/'
dgs_path = os.path.join(root, ship, cruise, 'gravimeter/DGS')
nav_path = os.path.join(root, ship, cruise, 'NAV')
dgs_files = np.sort(glob(os.path.join(dgs_path, 'AT1M-*.dat')))
nav_files = np.sort(
    glob(os.path.join(nav_path, '*mru_seapath330_navbho*.txt')))

# read and sort the nav data
gps_nav = sgi.read_nav(ship, nav_files, talker='GPGGA')
gps_nav.sort_values('time_sec', inplace=True)
gps_nav.reset_index(inplace=True, drop=True)

# read and sort the DGS laptop data
dgs_data = sgi.read_dgs_laptop(dgs_files, ship)
dgs_data.sort_values('date_time', inplace=True)
dgs_data.reset_index(inplace=True, drop=True)
dgs_data['tsec'] = [e.timestamp()
                    for e in dgs_data['date_time']]  # get posix timestamps
dgs_data['grav'] = dgs_data['rgrav'] + biases['dgs']

# sync data geographic coordinates to nav by interpolating with timestamps
# (interpolators use posix timestamps, not datetimes)
gps_lon_int = interp1d(gps_nav['time_sec'].values, gps_nav['lon'].values,
                       kind='linear', fill_value='extrapolate')
gps_lat_int = interp1d(gps_nav['time_sec'].values, gps_nav['lat'].values,
                       kind='linear', fill_value='extrapolate')
dgs_data['lon_new'] = gps_lon_int(dgs_data['tsec'].values)
dgs_data['lat_new'] = gps_lat_int(dgs_data['tsec'].values)

# calculate corrections for FAA
ellipsoid_ht = np.zeros(len(dgs_data))  # we are working at sea level
lat_corr = sgg.wgs_grav(dgs_data['lat_new']) + \
    sgg.free_air_second_order(dgs_data['lat_new'], ellipsoid_ht)
eotvos_corr = sgg.eotvos_full(dgs_data['lon_new'].values, dgs_data['lat_new'].values,
                              ellipsoid_ht, sampling)
tide_corr = sgg.longman_tide_prediction(
    dgs_data['lon_new'], dgs_data['lat_new'], dgs_data['date_time'])

dgs_data['faa'] = dgs_data['grav'] - lat_corr + eotvos_corr + tide_corr
dgs_data['full_field'] = dgs_data['grav'] + eotvos_corr + tide_corr

# apply a lowpass filter to FAA
taps = 2*240
freq = 1./240
# we resampled to the specified sampling rate when reading the data
nyquist = sampling/2
wn = freq/nyquist       # (if that wasn't the rate to begin with)
B = firwin(taps, wn, window='blackman')  # approx equivalent to matlab fir1

ffaa = filtfilt(B, 1, dgs_data['faa'])

# read in some other time series from MRUs
mru_path = os.path.join(root, ship, cruise, 'openrvdas/data')
mru_files = np.sort(
    glob(os.path.join(mru_path, 'SR2312_mru_hydrins_navbho*.txt')))
yaml_file = os.path.join(
    root, ship, cruise, 'openrvdas/doc/devices/IXBlue.yaml')
talk = 'PASHR'
dats = []
for mf in mru_files:
    dat, col_info = sgi.read_other_stuff(yaml_file, mf, talk)
    dats.append(dat)
mru_dat = concat(dats, ignore_index=True)
del dats, dat  # cleanup

# we have some prior knowledge about this data that lets us find the timestamps:
mru_dat['date_time'] = to_datetime(mru_dat['mystery'], utc=True)
mru_dat.drop('mystery', axis=1, inplace=True)
mru_dat['tsec'] = [e.timestamp() for e in mru_dat['date_time']]

# if we want to look at coherence between monitors and MRUs, have to interpolate first
# because monitors are at 2Hz
for motion in ['Pitch', 'Roll', 'Heave']:
    plt.figure()
    plt.title('%s coherence with monitors' % motion)
    # interpolate a sort of max envelope value for each MRU to the grvimeter sample rate
    mru_interp = np.interp(dgs_data.tsec, mru_dat.tsec,
                           mru_dat['%s:g' % motion].rolling(1000).max())
    mru_interp[np.isnan(mru_interp)] = 0  # Nans are no good for coherence
    for monitor in ['vcc', 've', 'al', 'ax']:
        freq, coh = coherence(dgs_data[monitor], mru_interp)
        plt.semilogy(freq, coh, label=monitor)
    plt.legend(fontsize=8)
    plt.xlabel('Frequency [Hz]')

plt.show()
