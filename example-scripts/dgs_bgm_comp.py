# %% [markdown]
# ### Example script for reading and lightly processing DGS laptop data from a Thompson cruise, and comparing to BGM serial files.
#
# Data files are downloaded by the script using pooch
#
# Read DGS and navigation files
#
# Read BGM files
#
# Correct for meter bias with info from shipgrav
#
# Compare BGM and DGS outputs
#
# Use timestamps to sync more accurate nav with the gravity data.
#
# Calculate FAA (free air anomaly)
#
# Plot
#
# satellite data tracked from v32.1 Global Gravity grid, which
# includes data from SIO, NOAA, and NGA.
#
# Reference: Sandwell et al. (2014) New global marine gravity model
# from CryoSat-2 and Jason-1 reveals buried tectonic struture.
# Science 346(6205), DOI: 10.1126/science.1258213

# %%
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pooch
import shipgrav.grav as sgg
import shipgrav.io as sgi
import tomli as tm
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, firwin


# %%
# set some general metadata
ship = 'Thompson'
cruise = 'TN400'        # this is used for filepaths
sampling = 1            # 1 Hz - data should be at this rate already

# read a few constants etc from our toml database file
with open('../shipgrav/database.toml', 'rb') as f:
    info = tm.load(f)
nav_tag = info['nav-talkers'][ship]
biases = info['bias-values'][ship]

# %%
# get files: DGS laptop from zenodo, BGM serial and nav from R2R
dgs_files = pooch.retrieve(url="https://zenodo.org/records/12733929/files/data.zip", 
        known_hash="md5:83b0411926c0fef9d7ccb2515bb27cc0", progressbar=True, 
        processor=pooch.Unzip(
            members=['data/Thompson/TN400/gravimeter/DGS/AT1M-Grav-PROC_20220314-000001.Raw',
                    'data/Thompson/TN400/gravimeter/DGS/AT1M-Grav-PROC_20220313-000001.Raw']))

bgm_files = pooch.retrieve(url="https://service.rvdata.us/data/cruise/TN400/fileset/151470",
        known_hash='b0ccc52e60334284dd79573e27e050eead77f5ac1d59f510385c7e05a8474bcf',progressbar=True,
        processor=pooch.Untar(
            members=['TN400/151470/data/BGM3-GRAV-RAW_20220313-000001.Raw',
                    'TN400/151470/data/BGM3-GRAV-RAW_20220314-000001.Raw']))

nav_files = pooch.retrieve(url="https://service.rvdata.us/data/cruise/TN400/fileset/151457",
        known_hash="76e66365c41d393510bb7ab9637305296282e9041415c1343faa171af28abf85",progressbar=True,
        processor=pooch.Untar(
            members=['TN400/151457/data/POSMV-V5-INGGA-RAW_20220313-000001.Raw',
                    'TN400/151457/data/POSMV-V5-INGGA-RAW_20220314-000001.Raw']))

# %%
# read and sort the nav data
gps_nav = sgi.read_nav(ship, nav_files)
gps_nav.sort_values('time_sec', inplace=True)
gps_nav.reset_index(inplace=True, drop=True)

# %%
# we happen to know that there are some weird nav dropouts in this dataset
# so clean them up here
bad_inds = np.where(np.diff(gps_nav['lon']) > 1)[0]
gps_nav.drop(bad_inds, axis=0, inplace=True)

# %%
# read and sort the DGS laptop data
dgs_data = sgi.read_dgs_laptop(dgs_files, ship)
dgs_data.sort_values('date_time', inplace=True)
dgs_data.reset_index(inplace=True, drop=True)
dgs_data['tsec'] = [e.timestamp()
                    for e in dgs_data['date_time']]  # get posix timestamps
dgs_data['grav'] = dgs_data['rgrav'] + biases['dgs']

# %%
# read and sort the BGM data
bgm_data = sgi.read_bgm_raw(bgm_files, ship)
bgm_data.sort_values('date_time', inplace=True)
bgm_data.reset_index(inplace=True, drop=True)
bgm_data['tsec'] = [e.timestamp() for e in bgm_data['date_time']]
bgm_data['grav'] = bgm_data['rgrav'] + biases['bgm']

# %%
# sync data geographic coordinates to nav by interpolating with timestamps
# (interpolators use posix timestamps, not datetimes)
gps_lon_int = interp1d(gps_nav['time_sec'].values, gps_nav['lon'].values,
                       kind='linear', fill_value='extrapolate')
gps_lat_int = interp1d(gps_nav['time_sec'].values, gps_nav['lat'].values,
                       kind='linear', fill_value='extrapolate')
dgs_data['lon_new'] = gps_lon_int(dgs_data['tsec'].values)
dgs_data['lat_new'] = gps_lat_int(dgs_data['tsec'].values)

bgm_data['lon_new'] = gps_lon_int(bgm_data['tsec'].values)
bgm_data['lat_new'] = gps_lat_int(bgm_data['tsec'].values)

# %%
# calculate corrections for FAA for DGS and BGM
for df in [dgs_data, bgm_data]:
    ellipsoid_ht = np.zeros(len(df))  # we are working at sea level
    lat_corr = sgg.wgs_grav(df['lat_new']) + \
        sgg.free_air_second_order(df['lat_new'], ellipsoid_ht)
    eotvos_corr = sgg.eotvos_full(df['lon_new'].values, df['lat_new'].values,
                                  ellipsoid_ht, sampling)
    tide_corr = sgg.longman_tide_prediction(
        df['lon_new'], df['lat_new'], df['date_time'])

    df['faa'] = df['grav'] - lat_corr + eotvos_corr + tide_corr
    df['full_field'] = df['grav'] + eotvos_corr + tide_corr

# %%
# apply a lowpass filter
taps = 2*240
freq = 1./240
# we resampled to the specified sampling rate when reading the data
nyquist = sampling/2
wn = freq/nyquist       # (if that wasn't the rate to begin with)
B = firwin(taps, wn, window='blackman')  # approx equivalent to matlab fir1

dfaa = filtfilt(B, 1, dgs_data['faa'])
bfaa = filtfilt(B, 1, bgm_data['faa'])

# %%
# load satellite data for comparison
sat_path = pooch.retrieve(url="https://zenodo.org/records/12733929/files/data.zip", 
        known_hash="md5:83b0411926c0fef9d7ccb2515bb27cc0", progressbar=True, 
        processor=pooch.Unzip(
            members=['data/Thompson/TN400/sandwell_tracked.llg']))
sat_grav = np.loadtxt(sat_path[0], usecols=(3,),
                      delimiter=',', skiprows=1)

# %%
# plot this data and satellite data (trim edge effects from filtering)
plt.figure(figsize=(11, 4.8))
plt.plot(dgs_data.iloc[taps:-taps//2]['date_time'],
         dfaa[taps:-taps//2], label='TN400 DGS')
plt.plot(bgm_data.iloc[taps:-taps//2]['date_time'],
         bfaa[taps:-taps//2], label='TN400 BGM')
plt.plot(dgs_data.iloc[taps:-taps//2]['date_time'],
         sat_grav[taps:-taps//2], label='satellite')
plt.xlabel('Timestamp')
plt.ylabel('Free air anomaly [mGal]')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
