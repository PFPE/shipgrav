import numpy as np
import shipgrav.io as sgi
import shipgrav.grav as sgg
import shipgrav.nav as sgn
from scipy.signal import firwin, filtfilt
from glob import glob
import os, sys

########################################################################
# Example script using sample data from DGS to illustrate how cross-
# coupling coefficients are calculated and applied.
# DGS provided a set of coefficients for this dataset, and the calculated
# values are compared to those.
########################################################################

# set some general metadata
ship = 'DGStest'
cruise = 'DG1'
sampling = 1
bias_dgs = 969143

# set up file paths, get lists of input files
root = 'data/'
dgs_path = os.path.join(root,ship,cruise,'gravimeter/DGS')
dgs_files = np.sort(glob(os.path.join(dgs_path,'AT1M*.dat')))

# read and sort the gravimeter data
dgs_data = sgi.read_dat_dgs(dgs_files, ship, scale_ccp=False)
dgs_data.sort_values('date_time',inplace=True)
dgs_data.reset_index(inplace=True,drop=True)
dgs_data['tsec'] = [e.timestamp() for e in dgs_data['date_time']]  # get posix timestamps
dgs_data['grav'] = dgs_data['rgrav'] + bias_dgs

# calculate corrections for FAA
ellipsoid_ht = np.zeros(len(dgs_data))  # we are working at sea level
lat_corr = sgg.wgs_grav(dgs_data['lat']) + sgg.fa_2ord(dgs_data['lat'],ellipsoid_ht)
eotvos_corr = sgg.eotvos_full(dgs_data['lon'].values, dgs_data['lat'].values,
                              ellipsoid_ht, sampling)
tide_corr = sgg.longman_tide_pred(dgs_data['lon'],dgs_data['lat'],dgs_data['date_time'])

dgs_data['faa'] = dgs_data['grav'] - lat_corr + eotvos_corr + tide_corr
dgs_data['full_field'] = dgs_data['grav'] + eotvos_corr + tide_corr

# calculate kinematic variables and corrections for tilt correction
# (maybe not strictly necessary? depends who you ask)
gps_vn, gps_ve = sgn.ll2en(dgs_data['lon'].values, dgs_data['lat'].values)
gps_vn = sampling*gps_vn; gps_ve = sampling*gps_ve

gps_eacc = 1e5*sampling*np.convolve(gps_ve, sgn.tay10, 'same')
gps_nacc = 1e5*sampling*np.convolve(gps_vn, sgn.tay10, 'same')

crse, vel = sgn.vevn2cv(gps_ve, gps_vn)
crse[np.isnan(crse)] = 0
vel[np.isnan(vel)] = 0
# TODO check for "stopped"?

acc_cross, acc_long = sgn.rot_acc_EN_cl(crse, gps_eacc, gps_nacc)  # gps-derived cross and long accel

# tilt correction
up_vecs = sgg.up_vecs(1/sampling, lat_corr, acc_cross, acc_long, 0, 240, 0.7071, 240, 0.7071)
igf_in_plat = lat_corr*up_vecs[2,:]  # latitude correction on platform
cross_in_plat = acc_cross*up_vecs[1,:]
long_in_plat = acc_long*up_vecs[0,:]
level_error = lat_corr - igf_in_plat - long_in_plat + cross_in_plat

# calculate cross-coupling coefficients
_, model = sgg.calc_ccp(dgs_data['faa'].values, dgs_data['vcc'].values, dgs_data['ve'].values, dgs_data['al'].values, dgs_data['ax'].values, level_error.values)

print('cross coupling parameters from fit:')
print('ve %.3f, vcc %.3f, al %.3f, ax %.3f, lev %.3f' % (model.params.ve, model.params.vcc,model.params.al, model.params.ax,model.params.lev))
print('cross coupling parameters provided by DGS for this data:')
print('ve %.3f, vcc %.3f, al %.3f, ax %.3f' % (-0.3509, 109., 0.251, 0.23))

# apply cross-coupling correction and plot the (filtered) FAA
dgs_data['faa_ccp'] = dgs_data['faa'] + model.params.ve*dgs_data['ve'] + \
                                        model.params.vcc*dgs_data['vcc'] + \
                                        model.params.al*dgs_data['al'] + \
                                        model.params.ax*dgs_data['ax']

taps = 2*240
freq = 1./240
nyquist = sampling/2    # we resampled to the specified sampling rate when reading the data
wn = freq/nyquist       # (if that wasn't the rate to begin with)
B = firwin(taps,wn,window='blackman')  # approx equivalent to matlab fir1

ffaa = filtfilt(B, 1, dgs_data['faa'])
cfaa = filtfilt(B, 1, dgs_data['faa_ccp'])
plt.figure(figsize=(11,4.8))
plt.plot(dgs_data.iloc[taps:-taps//2]['date_time'], ffaa[taps:-taps//2],label='no ccp')
plt.plot(dgs_data.iloc[taps:-taps//2]['date_time'], cfaa[taps:-taps//2],label='with ccp')
plt.xlabel('Timestamp')
plt.ylabel('Free air anomaly [mGal]')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

