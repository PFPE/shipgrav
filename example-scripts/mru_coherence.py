# %% [markdown]
# ### Example script for reading DGS laptop data and MRUs from an R/V Ride transit, and calculating coherence between MRUs and monitors
#
# Data files are downloaded by the script using pooch
#
# Read DGS laptop and navigation files
#
# Correct for meter bias with info from shipgrav
#
# Use timestamps to sync more accurate nav with the gravity data.
#
# Calculate FAA (free air anomaly) for laptop data
#
# Read MRUs (pitch/roll/heave) for all pairs of monitors and MRUs, interpolate MRU to grav sample rate and calculate coherence between monitor and 1000-pt moving average of MRU
#
# (and plot coherence)

# %%
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pooch
import tomli as tm
from pandas import concat, to_datetime
from scipy.interpolate import interp1d
from scipy.signal import coherence, filtfilt, firwin
from tqdm import tqdm

import shipgrav.grav as sgg
import shipgrav.io as sgi

# %%
# set some general metadata
ship = 'Ride'
cruise = 'SR2312'       # this is used for filepaths
sampling = 1            # 1 Hz - data should be at this rate already

# read a few constants etc from our toml database file
with open('../shipgrav/database.toml', 'rb') as f:
    info = tm.load(f)
nav_tag = info['nav-talkers'][ship]
biases = info['bias-values'][ship]

# %%
# get the DGS and nav data from R2R
dgs_files = pooch.retrieve(url="https://service.rvdata.us/data/cruise/SR2312/fileset/157179", 
        known_hash="53f53c45aa59ce19cd1e75e3d847b5697123ad0a1296aa2be28bf26ff0ad19ac", progressbar=True, 
        processor=pooch.Untar(
            members=['SR2312/157179/data/AT1M-25_20230616_167.dat',
                     'SR2312/157179/data/AT1M-25_20230617_168.dat',
                     'SR2312/157179/data/AT1M-25_20230618_169.dat',
                     'SR2312/157179/data/AT1M-25_20230619_170.dat',
                     'SR2312/157179/data/AT1M-25_20230620_171.dat']))

nav_files = pooch.retrieve(url="https://service.rvdata.us/data/cruise/SR2312/fileset/157188", 
        known_hash="eb5992eadd87b09f66308a4d577c6ba6965d8ed8489959ffaed8bf5556ee712f", progressbar=True, 
        processor=pooch.Untar(
            members=['SR2312/157188/data/SR2312_mru_seapath330_navbho-2023-06-16.txt',
                     'SR2312/157188/data/SR2312_mru_seapath330_navbho-2023-06-17.txt',
                     'SR2312/157188/data/SR2312_mru_seapath330_navbho-2023-06-18.txt',
                     'SR2312/157188/data/SR2312_mru_seapath330_navbho-2023-06-19.txt',
                     'SR2312/157188/data/SR2312_mru_seapath330_navbho-2023-06-20.txt']))

# %%
# read and sort the nav data
gps_nav = sgi.read_nav(ship, nav_files, talker='GPGGA')
gps_nav.sort_values('time_sec', inplace=True)
gps_nav.reset_index(inplace=True, drop=True)

# %%
# read and sort the DGS laptop data
dgs_data = sgi.read_dgs_laptop(dgs_files, ship)
dgs_data.sort_values('date_time', inplace=True)
dgs_data.reset_index(inplace=True, drop=True)
dgs_data['tsec'] = [e.timestamp()
                    for e in dgs_data['date_time']]  # get posix timestamps
dgs_data['grav'] = dgs_data['rgrav'] + biases['dgs']

# %%
# sync data geographic coordinates to nav by interpolating with timestamps
# (interpolators use posix timestamps, not datetimes)
gps_lon_int = interp1d(gps_nav['time_sec'].values, gps_nav['lon'].values,
                       kind='linear', fill_value='extrapolate')
gps_lat_int = interp1d(gps_nav['time_sec'].values, gps_nav['lat'].values,
                       kind='linear', fill_value='extrapolate')
dgs_data['lon_new'] = gps_lon_int(dgs_data['tsec'].values)
dgs_data['lat_new'] = gps_lat_int(dgs_data['tsec'].values)

# %%
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

# %%
# apply a lowpass filter to FAA
taps = 2*240
freq = 1./240
# we resampled to the specified sampling rate when reading the data
nyquist = sampling/2
wn = freq/nyquist       # (if that wasn't the rate to begin with)
B = firwin(taps, wn, window='blackman')  # approx equivalent to matlab fir1

ffaa = filtfilt(B, 1, dgs_data['faa'])

# %%
# get some MRU time series files from zenodo
# read in some other time series from MRUs
mru_files = pooch.retrieve(url="https://service.rvdata.us/data/cruise/SR2312/fileset/157177", 
        known_hash="ca12af51a33983f77d775098c06744e509baf31531a2afc37e39f1775f51d7f8", progressbar=True, 
        processor=pooch.Untar(
            members=['SR2312/157177/data/SR2312_mru_hydrins_navbho-2023-06-16.txt',
                     'SR2312/157177/data/SR2312_mru_hydrins_navbho-2023-06-17.txt',
                     'SR2312/157177/data/SR2312_mru_hydrins_navbho-2023-06-18.txt',
                     'SR2312/157177/data/SR2312_mru_hydrins_navbho-2023-06-19.txt',
                     'SR2312/157177/data/SR2312_mru_hydrins_navbho-2023-06-20.txt']))
yaml_file = pooch.retrieve(url="https://zenodo.org/records/12733929/files/data.zip", 
        known_hash="md5:83b0411926c0fef9d7ccb2515bb27cc0", progressbar=True, 
        processor=pooch.Unzip(
            members=['data/Ride/SR2312/openrvdas/doc/devices/IXBlue.yaml']))

# %%
talk = 'PASHR'
dats = []
# note that the MRU data will need to be sorted in the same time series order as the DGS data
# sorting by filename accomplishes this since the paths have dates
for mf in tqdm(np.sort(mru_files),desc='reading MRUs'):
    dat, col_info = sgi.read_other_stuff(yaml_file[0], mf, talk)
    dats.append(dat)
mru_dat = concat(dats, ignore_index=True)
del dats, dat  # cleanup

# %%
# we have some prior knowledge about this data that lets us find the timestamps:
mru_dat['date_time'] = to_datetime(mru_dat['mystery'], utc=True)
mru_dat.drop('mystery', axis=1, inplace=True)
mru_dat['tsec'] = [e.timestamp() for e in mru_dat['date_time']]

# %%
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

# %%
