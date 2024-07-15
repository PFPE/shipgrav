import numpy as np
from pandas import read_csv
import shipgrav.grav as sgg
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
from scipy.signal import firwin, filtfilt
from scipy.interpolate import interp1d
from glob import glob
import os
import sys

########################################################################
# Example script for calculating the mantle Bouger anomaly (MBA) and
# residual MBA (RMBA), and estimating crustal thickness variations, using
# data from a Ride transit.

# This script works with a file from the interactive line picker script
# that is located in
# data/Ride/SR2312/gravimeter/DGS/line-segments/

# It is hard-coded to run on a specific example file but could be
# adjusted (see below)

# Tthe RMBA thermal correction is highly sensitive to edge effects,
# Here we will demo the process of embedding the line segment
# in a longer line pulled from global grids: FAA, bathymetry,
# sediment thickness, and plate age.
# For a cruise with multibeam and magnetics (for bathymetry and age)
# both of those could be embedded alongside FAA. For this cruise,
# everything but the FAA is entirely pulled from global grids.

# Since this is an example running on a known input file, we have pre-
# tracked the coordinates in the following global grids:
# Bathymetry:
# The GEBCO_2014 Grid, version 20141103, http://www.gebco.net
# FAA:
# Sandwell, D. T., R. D. Müller, W. H. F. Smith, E. Garcia, R. Francis (2014).
# New global marine gravity model from CryoSat-2 and Jason-1 reveals buried
# tectonic structure, Science 346(6205), pp. 65-67,
# doi: 10.1126/science.1258213
# Sediment thickness:
# Divins, D.L. (2003). Total Sediment Thickness of the World's Oceans &
# Marginal Seas, NOAA National Geophysical Data Center, Boulder, CO.
# Plate age:
# Seton, M., Müller, R. D., Zahirovic, S., Williams, S., Wright, N., Cannon, J.,
# Whittaker, J., Matthews, K., McGirr, R., (2020). A global dataset of
# present-day oceanic crustal age and seafloor spreading parameters,
# Geochemistry, Geophysics, Geosystems, doi: 10.1029/2020GC009214
# and
# Müller, R. D., Zahirovic, S., Williams, S. E., Cannon, J., Seton, M.,
# Bower, D. J., Tetley, M. G., Heine, C., Le Breton, E., Liu, S., Russell, S. H. J.,
# Yang, T., Leonard, J., and Gurnis, M. (2019). A global plate model
# including lithospheric deformation along major rifts and orogens since the
# Triassic. Tectonics, 38, doi: 10.1029/2018TC005462
# (age grids accessed from earthbyte.org)

# To run this example using a different input file, you will need
# to supply paths to grids of bathymetry, FAA, sediment thickness,
# and plate age; and make sure that units match what the script expects.
########################################################################

ship = 'Ride'
cruise = 'SR2312'
root = 'data/'
seg_path = os.path.join(root, ship, cruise, 'gravimeter/DGS/line-segments')
# seg_files = np.sort(glob(os.path.join(seg_path,'segment*.dat')))
# we want to use one particular file that is saved here, which is:
seg_file = os.path.join(seg_path, 'example_segment_s3_330234-428488.dat')

# read the data
data = read_csv(seg_file, parse_dates=[16,])  # fully offshore Newport, OR
# this segment is a straight-ish line; it has a lot of stations along it
# where the ship was more or less stationary but that's ok

# set up coordinates for extending the ends of the line
wgs = Geodesic.WGS84  # object for calculating line extensions etc
faz01 = wgs.Inverse(data.iloc[0]['lat_new'], data.iloc[0]['lon_new'],
                    data.iloc[-1]['lat_new'], data.iloc[-1]['lon_new'])['azi2']  # faz at end
faz10 = wgs.Inverse(data.iloc[-1]['lat_new'], data.iloc[-1]['lon_new'],
                    data.iloc[0]['lat_new'], data.iloc[0]['lon_new'])['azi2']  # faz from 0
l_seg = wgs.Inverse(data.iloc[0]['lat_new'], data.iloc[0]['lon_new'],
                    data.iloc[-1]['lat_new'], data.iloc[-1]['lon_new'])['s12']  # segment length (m)
n_seg = len(data)
l_ext = 500.e3  # default 500 km extension on each end for long lines
if l_seg < l_ext:  # but if it's a short segment, just use the segment length on each end
    l_ext = l_seg
# calculate distances btwn points to get avg spacing for extensions
xrng = np.zeros(len(data))
for i in range(1, len(data)):
    xrng[i] = wgs.Inverse(data.iloc[0]['lat_new'], data.iloc[0]['lon_new'],
                          data.iloc[i]['lat_new'], data.iloc[i]['lon_new'])['s12']
dx_avg = np.mean(np.diff(xrng))
n_ext = int(l_ext/dx_avg)

# get the actual coordinate points for each extended side
frnt_ext = np.zeros((n_ext, 2))
back_ext = np.zeros((n_ext, 2))
for i in range(n_ext):
    frnt_pt = wgs.Direct(data.iloc[0]['lat_new'],
                         data.iloc[0]['lon_new'], faz10, dx_avg*(i+1))
    back_pt = wgs.Direct(data.iloc[-1]['lat_new'],
                         data.iloc[-1]['lon_new'], faz01, dx_avg*(i+1))
    frnt_ext[i, 0] = frnt_pt['lon2']
    frnt_ext[i, 1] = frnt_pt['lat2']
    back_ext[i, 0] = back_pt['lon2']
    back_ext[i, 1] = back_pt['lat2']

# compile full coordinate lists for the extended line, write out for grdtrack
a_lon = np.hstack(
    [frnt_ext[:, 0][::-1], data['lon_new'].values, back_ext[:, 0]])
a_lat = np.hstack(
    [frnt_ext[:, 1][::-1], data['lat_new'].values, back_ext[:, 1]])
########################################################################
# this would be the point where you would track through gridfiles
# if using a different line than the example file.
########################################################################
# with open('temp.ll','w') as f:
#    for i in range(len(a_lon)):
#        f.write('%15f %15f\n' % (a_lon[i], a_lat[i]))
#
# track FAA, bathymetry, ages, sediments in global grids
# dep_grd = os.path.expanduser('~/Data/GEBCO/RN-4105_1418140094903/gebco.grd')
# faa_grd = os.path.expanduser('~/Data/other_grids/sandwell_grid/grav_32.1.nc')
# sed_grd = os.path.expanduser('~/Data/other_grids/sedthick_world.grd')
# age_grd = os.path.expanduser('~/Data/Muller_agegrids/age.2020.1.GeeK2007.1m.nc')
# os.system('gmt grdtrack temp.ll -G%s -G%s -G%s -G%s > tracked.llm' % (dep_grd, faa_grd, sed_grd, age_grd))
########################################################################
# since we've already pre-tracked the example line, we'll just load in the
# info here
track = read_csv(os.path.join(seg_path, 'tracked.llm'), sep='\t',
                 names=['lon', 'lat', 'dep', 'faa', 'sed', 'age'])
# NOTE that if you are working with data from near a coast, line extensions
# might end up on land. Check for NaN values in your tracked file.

# filter FAA from the segment since Sandwell grid is very long-wavelength
sampling = 1
taps = 2*240
freq = 1./240
nyquist = sampling/2
wn = freq/nyquist
B = firwin(taps, wn, window='blackman')  # approx equivalent to matlab fir1
ffaa = filtfilt(B, 1, data['faa'])

# embed our FAA in the tracked line (everything else remains tracked)
track['faa'].iloc[n_ext+taps:-n_ext-taps] = ffaa[taps:-taps]
# NOTE you may want to check your embedded FAA for large jumps at the ends
# and smooth them out into the gridded data. Not bothering here because
# the jumps aren't *too* big* and also we don't care that much about the
# results of this example, but for real analyses, you should care!

# interpolate everything to an even X spacing (after recalculating total distance)
xpts_line = np.zeros(len(track))
for i in range(1, len(track)):
    xpts_line[i] = wgs.Inverse(track.iloc[0]['lat'], track.iloc[0]['lon'],
                               track.iloc[i]['lat'], track.iloc[i]['lon'])['s12']
# NOTE be careful with the number of points here
xobs = np.linspace(0, xpts_line[-1], int(1e3))
# bc it determines the dx for the crustal thickness calc which has an effect on
# frequencies/wavelengths that can reasonably be used. if dx is too small zdown
# has to also be very small, which makes less sense for crustal thickness
# Basically, know that there *is* such thing as sampling too finely. 1e3 works ok here.

fgrav = interp1d(xpts_line, track['faa'].values)
FAA_int = fgrav(xobs)
fdep = interp1d(xpts_line, track['dep'].values)
dep_int = fdep(xobs)
fage = interp1d(xpts_line, track['age'].values)
age_int = fage(xobs)
fsed = interp1d(xpts_line, track['sed'].values)
sed_int = fsed(xobs)

# calculate topography, sediment, and crust corrections (for MBA)
# and thermal correction (for RMBA)
# sed density approx based on Hamilton 1978, doi: 10.1121/1.381747
rho = np.array((1000, 2100, 2900))
a = 3.e-5  # thermal expansion coeff; 3e-5 is default value for therm_Z calculation
Tm = 1350
dtemp = 50
temps = np.arange(0, Tm+1, dtemp)
rhom = 3300
rhos = rhom*(1-a*(temps-Tm))
rho = np.append(rho, rhos)
drho = np.diff(rho)
dz_c = 6000

anom_w = sgg.grav1d_padded(xobs, dep_int, 0, drho[0])  # topography
anom_s = sgg.grav1d_padded(xobs, dep_int-sed_int, 0, drho[1])  # sediment
anom_c = sgg.grav1d_padded(xobs, dep_int-sed_int-dz_c, 0, drho[2])  # crust

MBA = FAA_int - anom_w - anom_s - anom_c  # mantle Bouger anomaly

# correct for plate cooling (RMBA)
corrs = np.zeros((len(temps)-1, len(xobs)))  # calculate gravity due to temps
for i in range(len(temps)-1):  # loop over a set of isotherms defined above
    ziso, _ = sgg.therm_Z_halfspace(age_int, temps[i], time=True, Tm=Tm, a=a)
    corrs[i, :] = sgg.grav1d_padded(
        xobs, dep_int-sed_int-ziso-dz_c, 0, drho[i+3])

# RMBA is MBA minus summed thermal correction
RMBA = MBA - np.sum(corrs, axis=0)

# estimate crustal thickness
nx = len(xobs)
ny = 1  # we're only working with a line, but the function works for a 2D region
dx = np.mean(np.diff(xobs))/1e3  # spacing is even; this is in km
dy = 0
zdown = 5  # in km; might use something more like 10 for a longer line/larger dx
rho = 0.4  # density contrast
# in km; keep these 2 values small for short line (otherwise more like 100/45)
wlarge = 30
wsmall = 5

# do the calculation with back=True so it also returns the recovered
# RMBA as a check
cthick, rev_grav = sgg.crustal_thickness_2D(RMBA, nx=nx, ny=ny, dx=dx, dy=dy,
                                            zdown=zdown, rho=rho, wlarge=wlarge, wsmall=wsmall, back=True)

# make some plots to see what we got out of this
plt.figure()
plt.plot(xobs/1e3, np.real(rev_grav), label='recovered RMBA')
plt.plot(xobs/1e3, RMBA, label='RMBA')
plt.xlabel('distance along line [km]')
plt.ylabel('anomaly [mGal]')
plt.legend(fontsize=8)

cutoff = 100  # index for trimming edge effects from the crustal thickness calc; somewhat arbitrary
plt.figure()
plt.plot(xobs[cutoff:-cutoff]/1e3, np.real(cthick[cutoff:-cutoff]))
plt.xlabel('distance along line [km]')
plt.ylabel('crustal thickness variation [km]')

plt.show()
