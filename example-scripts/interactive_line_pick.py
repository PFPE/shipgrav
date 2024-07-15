import numpy as np
import shipgrav.grav as sgg
import shipgrav.nav as sgn
import shipgrav.io as sgi
import shipgrav.utils as sgu
import tomli as tm
from glob import glob
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backend_bases import MouseEvent
from shutil import rmtree
import os
import sys

########################################################################
# Example script for picking out segments of gravimeter data from an
# R/V Ride transit using interactive clicking on a map/plot. This is
# useful if you want some straight line segments of a ship track for
# calculating MBA or RMBA and the ship did not conveniently change
# course at midnight when the gravity files turn over.
# Segments of grav data are written to files for rereading later.
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
cal_factor = info['dgs-stuff']['calibration_factor']

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
dgs_data['faa_filt'] = ffaa

# trim off filter edge effects
dgs_data = dgs_data.iloc[taps:-taps//2]
dgs_data.reset_index(inplace=True, drop=True)
ffaa = ffaa[taps:-taps//2]

while True:
    plt.ion()
    # set up a figure for the interactive picking
    fig = plt.figure(constrained_layout=True, figsize=(10, 7))
    gs = GridSpec(2, 1, figure=fig)
    ax2 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    ax2.plot(dgs_data['lon_new'], dgs_data['lat_new'])
    S1 = ax1.plot(dgs_data['tsec'], ffaa)

    dot = ax2.plot(dgs_data.iloc[0]['lon_new'],
                   dgs_data.iloc[0]['lat_new'], marker='o', color='k')

    cursor = sgu._SnappingCursor(ax1, ax2, S1[0], dot[0], dgs_data)
    cid = plt.connect('motion_notify_event', cursor.on_mouse_move)
    cid2 = plt.connect('button_press_event', cursor.on_mouse_click)

    print('Left click in the lower panel to set points for the ends of line segments in the upper panel')
    print('Points should be selected in order, and in pairs (start and end) for each segment')
    print('Right click in the lower panel to remove the last clicked point')
    print('When you have selected all of the points you want,')
    input('press enter to save the points and close the plot -> ')
    plt.disconnect(cid)
    plt.disconnect(cid2)
    plt.close()

    # get the points defining the segment ends
    inds = cursor.i_seg  # indices, theoretically doubled, to split on

    if len(inds) % 2 != 0:
        print('odd number of points selected; deleting the last point whether you like it or not')
        inds.pop()

    segs = []
    for i in range(0, len(inds), 2):  # by twos to get pairs
        segs.append((min(inds[i:i+2]), max(inds[i:i+2])))

    plt.figure()
    for s in segs:
        plt.plot(dgs_data.iloc[s[0]:s[1]]['lon_new'],
                 dgs_data.iloc[s[0]:s[1]]['lat_new'])

    print('here are your segments!')
    print('to accept them, press enter')
    iq = input(
        "to go back and re-pick them, enter 'n' and then press enter -> ") or 'y'
    if iq == 'y':
        break
    elif iq == 'n':
        pass

# with segments safely selected, we'll write out the data for each into a separate file that can be re-read later for further processing
# make a directory for this segmented data
seg_path = os.path.join(root, ship, cruise, 'gravimeter/DGS/line-segments')
os.makedirs(seg_path, exist_ok=True)

if os.listdir(seg_path) != []:
    print('there are segments defined for this cruise already')
    print('to clear those segments, enter c')
    print('to keep those segments (overwrite or add new files), enter k')
    iq = input('-> ') or 'k'
    if iq == 'c':
        rmtree(seg_path)  # delete dir and its contents
        os.makedirs(seg_path)    # remake empty dir

# loop segments, write files
for i, s in enumerate(segs):
    opath = os.path.join(seg_path, 'segment_s%i_%i-%i.dat' % (i, s[0], s[1]))
    sel = dgs_data.iloc[s[0]:s[1]]
    sel.to_csv(opath, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
