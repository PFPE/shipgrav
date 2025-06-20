# %% [markdown]
# ### Example script using data from TN400 to illustrate how cross-coupling coefficients are calculated and applied.
#
# Data files are downloaded by the script using pooch
#
# Read DGS data files
# (not bothering with navigation syncing; see dgs_bgm_comp.py)
#
# Calculate FAA
#
# Calculate various kinematic parameters
#
# Fit for cross-coupling coefficients
#
# Correct for cross-coupling
#
# Plot with and without corrections

# %%
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pooch
import shipgrav.grav as sgg
import shipgrav.io as sgi
import shipgrav.nav as sgn
import tomli as tm
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, firwin

# %%
# set some general metadata
ship = 'Thompson'
cruise = 'TN400'
sampling = 1
# read a few constants etc from our toml database file
with open('../shipgrav/database.toml', 'rb') as f:
    info = tm.load(f)
# nav_tag = info['nav-talkers'][ship]  # could use this for nav file read
bias_dgs = info['bias-values'][ship]['dgs']

# %%
# get the files from Zenodo: TN400 DGS laptop data
# the archive download includes some things not used for this example; those are not unpacked
dgs_files = pooch.retrieve(url="https://zenodo.org/records/12733929/files/data.zip", 
        known_hash="md5:83b0411926c0fef9d7ccb2515bb27cc0", progressbar=True, 
        processor=pooch.Unzip(
            members=['data/Thompson/TN400/gravimeter/DGS/AT1M-Grav-PROC_20220314-000001.Raw',
                    'data/Thompson/TN400/gravimeter/DGS/AT1M-Grav-PROC_20220313-000001.Raw']))

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
# read and sort the gravimeter data
dgs_data = sgi.read_dgs_laptop(dgs_files, ship)
dgs_data.sort_values('date_time', inplace=True)
dgs_data.reset_index(inplace=True, drop=True)
dgs_data['tsec'] = [e.timestamp()
                    for e in dgs_data['date_time']]  # get posix timestamps
dgs_data['grav'] = dgs_data['rgrav'] + bias_dgs

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
# calculate kinematic variables and corrections for tilt correction
# (maybe not strictly necessary? depends who you ask)
gps_vn, gps_ve = sgn.latlon_to_EN(
    dgs_data['lon_new'].values, dgs_data['lat_new'].values)
gps_vn = sampling*gps_vn
gps_ve = sampling*gps_ve

gps_eacc = 1e5*sampling*np.convolve(gps_ve, sgn.tay10, 'same')
gps_nacc = 1e5*sampling*np.convolve(gps_vn, sgn.tay10, 'same')

crse, vel = sgn.ENvel_to_course_heading(gps_ve, gps_vn)
crse[np.isnan(crse)] = 0
vel[np.isnan(vel)] = 0

acc_cross, acc_long = sgn.rotate_acceleration_EN_to_cl(
    crse, gps_eacc, gps_nacc)  # gps-derived cross and long accel

# %%
# tilt correction
up_vecs = sgg.up_vecs(1/sampling, lat_corr, acc_cross,
                      acc_long, 0, 240, 0.7071, 240, 0.7071)
igf_in_plat = lat_corr*up_vecs[2, :]  # latitude correction on platform
cross_in_plat = acc_cross*up_vecs[1, :]
long_in_plat = acc_long*up_vecs[0, :]
level_error = lat_corr - igf_in_plat - long_in_plat + cross_in_plat

dgs_data = dgs_data.iloc[2:]  # the first two points' FAA has some edge effect

# %%
# calculate cross-coupling coefficients
_, model = sgg.calc_cross_coupling_coefficients(dgs_data['faa'].values, dgs_data['vcc'].values, dgs_data['ve'].values,
                                                dgs_data['al'].values, dgs_data['ax'].values, level_error.values)

print('cross coupling parameters from fit:')
print('ve %.3f, vcc %.3f, al %.3f, ax %.3f, lev %.3f' % (model.params.ve,
      model.params.vcc, model.params.al, model.params.ax, model.params.lev))

# %%
# apply cross-coupling correction and plot the (filtered) FAA
dgs_data['faa_ccp'] = dgs_data['faa'] + model.params.ve*dgs_data['ve'] + \
    model.params.vcc*dgs_data['vcc'] + \
    model.params.al*dgs_data['al'] + \
    model.params.ax*dgs_data['ax']

# filter FAA
taps = 2*240
freq = 1./240
# we resampled to the specified sampling rate when reading the data
nyquist = sampling/2
wn = freq/nyquist       # (if that wasn't the rate to begin with)
B = firwin(taps, wn, window='blackman')  # approx equivalent to matlab fir1

ffaa = filtfilt(B, 1, dgs_data['faa'])
cfaa = filtfilt(B, 1, dgs_data['faa_ccp'])

# %%
# load satellite data for comparison
sat_path = pooch.retrieve(url="https://zenodo.org/records/12733929/files/data.zip", 
        known_hash="md5:83b0411926c0fef9d7ccb2515bb27cc0", progressbar=True, 
        processor=pooch.Unzip(
            members=['data/Thompson/TN400/sandwell_tracked.llg']))
sat_grav = np.loadtxt(sat_path[0], usecols=(3,),
                      delimiter=',', skiprows=1)
sat_grav = sat_grav[2:]


# %%
plt.figure(figsize=(11, 4.8))
plt.plot(dgs_data.iloc[taps:-taps//2]['date_time'],
         ffaa[taps:-taps//2], label='no ccp')
plt.plot(dgs_data.iloc[taps:-taps//2]['date_time'],
         cfaa[taps:-taps//2], label='with ccp')
plt.plot(dgs_data.iloc[taps:-taps//2]['date_time'],
         sat_grav[taps:-taps//2], label='satellite')
plt.xlabel('Timestamp')
plt.ylabel('Free air anomaly [mGal]')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
