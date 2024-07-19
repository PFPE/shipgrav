import numpy as np
from scipy.signal import lfilter, firwin
from pandas import DataFrame
from statsmodels.api import OLS, add_constant
from datetime import datetime, timezone
from math import factorial
from scipy.special import erf, erfcinv
from copy import copy

# impulse response of 10th order Taylor series differentiator
tay10 = [1/1260, -5/504, 5/84, -5/21, 5/6,
         0, -5/6, 5/21, -5/84, 5/504, -1/1260]

########################################################################
# tidal correction
########################################################################


def _convert_datetime_tidetime(timestamp):
    """Calculate julian century and hour from a timestamp.

    The reference point is noon on December 31, 1899 per Longman's paper.

    :param timestamp: time to convert to century + hour 
    :type timestamp: datetime.datetime

    :returns: 
        - **julian century** (*int*) - centuries since reference date, days/36525
        - **julian_hour** (*float*) - hours since midnight
    """
    origin_time = datetime(1899, 12, 31, 12, 0, 0, tzinfo=timezone.utc)
    dt = timestamp - origin_time
    days = dt.days + dt.seconds/3600./24.
    julian_century = days/36525
    julian_hour = (timestamp.hour + timestamp.minute /
                   60. + timestamp.second/3600.)

    return julian_century, julian_hour


def longman_tide_prediction(lon, lat, times, alt=0, return_components=False):
    """ Calculate predicted tidal effect on gravity using the Longman algorithm.

    The calculation is taken directly from

        Longman (1959). Formulas for Computing the Tidal Accelerations Due to
        the Moon and the Sun. Journal of Geophysical Research 64(12), 
        DOI: 10.1029/JZ064i012p02351

    as are all of the constant and variable descriptions.
    Tidal contribution(s) returned are in mGal.

    :param lon: longitude in decimal degrees, positive E
    :type lon: array_like
    :param lat: latitude in decimal degrees, positive N
    :type lat: array_like
    :param times: times for geographic locations
    :type times: array_like, datetime.datetime
    :param alt: altitude in meters (0 for sea level, for marine grav)
    :type alt: array_like
    :param return_components: if True, return lunar and solar components with total.
    :type return_components: bool

    :returns:
        - **g0** (*ndarray*)- total tidal effect in mGal
        - **gm** (*ndarray*)- lunar tidal effect in mGal (optional, if return_components is True)
        - **gs** (*ndarray*)- solar tidal effect in mGal (optional, if return_components is True)
    """

    assert len(lon) == len(lat), 'lengths of input vectors must be the same'
    assert len(lon) == len(times), 'lengths of input vectors must be the same'

    # convert the timestamps, referenced to Longman origin time
    T, t0 = np.empty(len(times)), np.empty(len(times))
    for i, stamp in enumerate(times):
        a, b = _convert_datetime_tidetime(stamp)
        T[i] = a
        t0[i] = b
        if t0[i] < 0:
            t0[i] += 24
        if t0[i] >= 24:
            t0[i] == 24

    TT = T*T    # squares and cubes are used a lot so shortcut them
    TTT = T*T*T

    # define a bunch of constants for the calculation
    mu = 6.673e-8  # Newton's gravitational constant
    M = 7.3537e25  # Mass of the moon in grams
    S = 1.993e33  # Mass of the sun in grams
    e = 0.05490  # Eccentricity of the moon's orbit
    m = 0.074804  # Ratio of mean motion of the sun to that of the moon
    c = 3.84402e10  # Mean distance between the centers of the earth and the moon
    c1 = 1.495e13  # Mean distance between centers of the earth and sun in cm
    h2 = 0.612  # Love parameter
    k2 = 0.303  # Love parameter
    a = 6.378270e8  # Earth's equitorial radius in cm
    i = 0.08979719  # (i) Inclination of the moon's orbit to the ecliptic
    # Inclination of the Earth's equator to the ecliptic 23.452 degrees
    omega = np.radians(23.452)
    lamb = np.radians(lat)  # (lambda) Latitude of point
    H = alt * 100.  # (H) Altitude above sea-level of point P in cm

    # Longman convention has W+/E- for unknown reasons
    L = -lon

    # Lunar gravity effects (Shureman 1941 coeffs)
    # mean longitude of the moon in its orbit, reckoned from the referred equinox
    s = 4.720023434 + 8399.709299*T + 0.0000440696*TT  # skip 3rd order bc it is tiny
    # mean longitude of lunar perigee
    p = 5.835124713 + 71.01800936*T - 0.000180545*TT - 0.00000021817*TTT
    # mean longitude of the sun
    h = 4.88162792 + 628.3319509*T + 0.00000527962*TT
    # longitude of moon's ascending node in orbit reckoned from referred equinox
    N = 4.523588564 - 33.75715303*T + 0.000036749*TT  # skip 3rd order bc it is tiny

    # inclination of the moon's orbit to the equator
    I = np.arccos(np.cos(omega)*np.cos(i) - np.sin(omega)*np.sin(i)*np.cos(N))
    # longitude in the celestial equator of its intersection A with the moon's orbit
    nu = np.arcsin(np.sin(i)*np.sin(N)/np.sin(I))
    # hour angle of mean sun measured westward from the place of observations
    t = np.radians(15. * (t0 - 12) - L)

    # right ascension of meridian of place of observations reckoned from A
    chi = t + h - nu
    # cos(alpha) where alpha is defined in eq. 15 and 16 of Longman 1959
    cos_alpha = np.cos(N)*np.cos(nu) + np.sin(N)*np.sin(nu)*np.cos(omega)
    # sin(alpha) where alpha is defined in eq. 15 and 16 of Longman 1959
    sin_alpha = np.sin(omega)*np.sin(N)/np.sin(I)
    # alpha from eq. 18 of Longman 1959
    alpha = 2*np.arctan(sin_alpha/(1 + cos_alpha))
    # longitude in the moon's orbit of its ascending intersection with the celestial equator
    xi = N - alpha

    # mean longitude of moon in radians in its orbit reckoned from A
    sigma = s - xi
    # longitude of moon in its orbit reckoned from its ascending intersection with the equator
    L_moon = (sigma + 2*e*np.sin(s - p) + (5./4)*e*e*np.sin(2*(s - p)) +
              (15./4)*m*e*np.sin(s - 2*h + p) + (11./8)*m*m*np.sin(2*(s - h)))

    # Solar calculations
    #  mean longitude of solar perigee
    p1 = 4.908229461 + 0.03000526416*T + 0.000007902463*TT  # skip tiny 3rd order term
    # eccentricity of Earth's orbit
    e1 = 0.01675104 - 0.0000418*T - 0.000000126*TT

    # right ascension of meridian of place of observations reckoned from the vernal equinox
    chi1 = t + h
    # longitude of sun in the ecliptic reckoned from the vernal equinox
    L_sun = h + 2*e1*np.sin(h - p1)
    # cosine(theta) where theta is the zenith angle of the moon
    cos_theta = (np.sin(lamb)*np.sin(I)*np.sin(L_moon) + np.cos(lamb)*(np.cos(0.5*I)**2
                                                                       * np.cos(L_moon - chi) + np.sin(0.5*I)**2*np.cos(L_moon + chi)))
    # cosine(phi) where phi is the zenith angle of the sun
    cos_phi = (np.sin(lamb)*np.sin(omega)*np.sin(L_sun) + np.cos(lamb) *
               (np.cos(0.5*omega)**2*np.cos(L_sun - chi1) +
               np.sin(0.5*omega)**2*np.cos(L_sun + chi1)))

    # Distance
    # distance parameter, eq. 34 in Longman 1959
    C = np.sqrt(1./(1 + 0.006738*np.sin(lamb)**2))
    # distance from point P to the center of the Earth
    r = C*a + H
    # distance parameter, eq. 31 in Longman 1959
    aprime = 1. / (c * (1 - e * e))
    # distance parameter, eq. 32 in Longman 1959
    aprime1 = 1. / (c1 * (1 - e1 * e1))
    # distance between centers of the Earth and the moon
    d = (1./((1./c) + aprime*e*np.cos(s - p) + aprime*e*e *
             np.cos(2*(s - p)) + (15./8)*aprime*m*e*np.cos(s - 2*h + p)
             + aprime*m*m*np.cos(2*(s - h))))
    # distance between centers of the Earth and the sun
    D = 1./((1./c1) + aprime1*e1*np.cos(h - p1))

    # vertical component of tidal acceleration due to the moon
    gm = ((mu*M*r/(d*d*d))*(3*cos_theta**2 - 1) + (3./2) *
          (mu*M*r*r/(d*d*d*d))*(5*cos_theta**3 - 3*cos_theta))
    # vertical component of tidal acceleration due to the sun
    gs = mu*S*r/(D*D*D)*(3*cos_phi**2 - 1)

    love = (1 + h2 - 1.5*k2)
    g0 = (gm + gs) * 1e3*love
    if return_components:
        return g0, gm*1e3*love, gs*1e3*love
    else:
        return g0

########################################################################
# Eotvos correction
########################################################################


def eotvos_full(lon, lat, ht, samp, a=6378137.0, b=6356752.3142):
    """ Full Eotvos correction in mGals.

    The Eotvos correction is the effect on measured gravity due to horizontal
    motion over the Earth's surface.

    This formulation is from Harlan (1968), "Eotvos Corrections for Airborne Gravimetry" in
    *Journal of Geophysical Research*, 73(14), DOI: 10.1029/JB073i014p04675

    Implementation modified from matlab script written by Sandra Preaux, NGS, NOAA August 24 2009

    Components of the correction:

    * rdoubledot
    * angular acceleration of the reference frame
    * corliolis
    * centrifugal
    * centrifugal acceleration of Earth

    :param lon: longitudes in degrees
    :type lon: array_like
    :param lat: latitudes in degrees
    :type lat: array_like
    :param ht: elevation (referenced to sea level)
    :type ht: array_like
    :param samp: sampling rate
    :type samp: float
    :param a: major axis of ellipsoid (default is WGS84)
    :type a: float, optional
    :param b: minor axis of ellipsoid (default is WGS84)
    :type b: float, optional

    :return: **E** (*ndarray*), Eotvos correction in mGal
    """
    We = 0.00007292115    # siderial rotation rate, radians/sec
    mps2mgal = 100000     # m/s/s to mgal
    ecc = (a-b)/a

    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)

    # get time derivatives of position
    dlat = center_diff(latr, 1, samp)
    ddlat = center_diff(latr, 2, samp)
    dlon = center_diff(lonr, 1, samp)
    ddlon = center_diff(lonr, 2, samp)
    dht = center_diff(ht, 1, samp)
    ddht = center_diff(ht, 2, samp)

    # sines and cosines etc
    slat = np.sin(latr[1:-1])
    clat = np.cos(latr[1:-1])
    s2lat = np.sin(2*latr[1:-1])
    c2lat = np.cos(2*latr[1:-1])

    # calculate r' and its derivatives
    rp = a*(1 - ecc*slat*slat)
    drp = -a*dlat*ecc*s2lat
    ddrp = -a*ddlat*ecc*s2lat - 2*a*dlat*dlat*ecc*c2lat

    # calculate deviation from normal and derivatives
    D = np.arctan(ecc*s2lat)
    dD = 2*dlat*ecc*c2lat
    ddD = 2*ddlat*ecc*c2lat - 4*dlat*dlat*ecc*s2lat

    # define r and its derivatives
    r = np.vstack((-rp*np.sin(D), np.zeros(len(rp)), -
                  rp*np.cos(D) - ht[1:-1])).T
    rdot = np.vstack((-drp*np.sin(D) - rp*dD*np.cos(D),
                     np.zeros(len(rp)), -drp*np.cos(D) + rp*dD*np.sin(D) - dht)).T
    ci = -ddrp*np.sin(D) - 2.*drp*dD*np.cos(D) - rp * \
        (ddD*np.cos(D) - dD*dD*np.sin(D))
    ck = -ddrp*np.cos(D) + 2.*drp*dD*np.sin(D) + rp * \
        (ddD*np.sin(D) + dD*dD*np.cos(D) - ddht)
    rdotdot = np.vstack((ci, np.zeros(len(ci)), ck)).T

    # define w and derivative
    w = np.vstack(((dlon + We)*clat, -dlat, -(dlon + We)*slat)).T
    wdot = np.vstack((dlon*clat - (dlon + We)*dlat*slat, -
                     ddlat, -ddlon*slat - (dlon + We)*dlat*clat)).T

    w2xrdot = np.cross(2*w, rdot)
    wdotxr = np.cross(wdot, r)
    wxr = np.cross(w, r)
    wxwxr = np.cross(w, wxr)

    # calcualte wexwexre, centrifugal acceleration due to the Earth
    re = np.vstack((-rp*np.sin(D), np.zeros(len(rp)), -rp*np.cos(D))).T
    we = np.vstack((We*clat, np.zeros(len(slat)), -We*slat)).T
    wexre = np.cross(we, re)
    wexwexre = np.cross(we, wexre)
    wexr = np.cross(we, r)
    wexwexr = np.cross(we, wexr)

    # calculate total acceleration for the aircraft
    a = rdotdot + w2xrdot + wdotxr + wxwxr

    # Eotvos correction is the vertical component of the total acceleraton of
    # the aircraft minus the centrifugal acceleration of the Earth, convert to mGal
    E = (a[:, 2] - wexwexr[:, 2])*mps2mgal
    E = np.hstack((E[0], E, E[-1]))

    return E

########################################################################
# latitude correction functions
########################################################################


def free_air_second_order(lat, ht):
    """ 2nd order free-air correction

    :param lat: latitude, degrees
    :type lat: array_like
    :param height: elevation, meters
    :type height: array_like

    :return: free-air correction, mGal
    """
    s2lat = np.sin(np.deg2rad(lat))**2

    return -((0.3087691 - 0.0004398*s2lat)*ht) + 7.2125e-8*(ht**2)


def wgs_grav(lat):
    """ Theoretical gravity for WGS84 ellipsoid

    :param lat: latitude, degrees
    :type lat: array_like

    :return: uniform ellipsoid gravity, mGal
    """
    sinsq = np.sin(np.deg2rad(lat))**2

    num = 1 + 0.00193185265241*sinsq   # something like (b*gp - a*ge)/(a*ge)
    den = np.sqrt(1 - 0.00669437999014*sinsq)  # this is e2
    return 978032.53359*(num/den)

########################################################################
# cross-coupling coefficients
########################################################################


def calc_cross_coupling_coefficients(faa_in, vcc_in, ve_in, al_in, ax_in, level_in, times=None, samplerate=1):
    """ Calculate cross-coupling coefficients from FAA via ordinary linear regression

    The cross-coupling coefficients are returned in `model`, in the `params` attribute

    :param faa_in: free air anomaly, filtered
    :type faa_in: array_like
    :param vcc_in: vcc monitor
    :type vcc_in: array_like
    :param ve_in: ve monitor
    :type ve_in: array_like
    :param al_in: al monitor
    :type al_in: array_like
    :param ax_in: ax monitor
    :type ax_in: array_like
    :param level_in: tilt/leveling correction, often negligible. 
        Use a vector of zeros to ignore this component.
    :type level_in: array_like
    :param times: timestamps to use for dividing the data into continuous sections
    :type times: array_like, floats, optional
    :param samplerate: used with times to detect large sampling gaps in the data
    :type samplerate: float, optional

    :return: 
        - **df** (*pd.DataFrame*)- double-differenced and filtered monitors and gravity
        - **model** (*statsmodels.OLS*)- linear regression model
    """

    end_inds = np.array([len(faa_in),])  # just the one
    if times is not None:  # supplied a vector of timestamps to go with everything else
        # split data into segments of continuous even sample rate
        tdiff = np.diff(times)
        if np.any(tdiff > 5*samplerate):
            # 5 sec gap doesn't matter much I hope???
            end_inds = np.where(tdiff > 5*samplerate)[0]

    end_inds = np.append(-1, end_inds)  # add starting point

    faa_tf = np.array([])
    vcc_tf = np.array([])
    ve_tf = np.array([])
    al_tf = np.array([])
    ax_tf = np.array([])
    le_tf = np.array([])  # things to fit

    for i in range(1, len(end_inds)):  # loop continuous-time segments
        faa = faa_in[end_inds[i-1]+1:end_inds[i]]
        vcc = vcc_in[end_inds[i-1]+1:end_inds[i]]
        ve = ve_in[end_inds[i-1]+1:end_inds[i]]
        al = al_in[end_inds[i-1]+1:end_inds[i]]
        ax = ax_in[end_inds[i-1]+1:end_inds[i]]
        level = level_in[end_inds[i-1]+1:end_inds[i]]
        if len(faa) < 1000:
            continue  # no point for very short segments
        # double derivatives with taylor series
        gpp = np.convolve(np.convolve(faa, tay10, 'same'), tay10, 'same')
        vccpp = np.convolve(np.convolve(vcc, tay10, 'same'), tay10, 'same')
        vepp = np.convolve(np.convolve(ve, tay10, 'same'), tay10, 'same')
        alpp = np.convolve(np.convolve(al, tay10, 'same'), tay10, 'same')
        axpp = np.convolve(np.convolve(ax, tay10, 'same'), tay10, 'same')
        levpp = np.convolve(np.convolve(level, tay10, 'same'), tay10, 'same')

        # trim the ends
        gpp = gpp[10:-10]
        vccpp = vccpp[10:-10]
        vepp = vepp[10:-10]
        alpp = alpp[10:-10]
        axpp = axpp[10:-10]
        levpp = levpp[10:-10]

        # fairly narrow filter to get rid of any high-freq noise generated by the double derivation
        filterlength = 100  # Aliod code has this as 10...but that gives VERY different cc values
        taps = int(2*filterlength)
        BM = firwin(taps, 1/taps, window='blackman')
        fgpp = lfilter(BM, 1, gpp)
        fvccpp = lfilter(BM, 1, vccpp)
        fvepp = lfilter(BM, 1, vepp)
        falpp = lfilter(BM, 1, alpp)
        faxpp = lfilter(BM, 1, axpp)
        flevpp = lfilter(BM, 1, levpp)

        # trim off filter transients
        fgpp = fgpp[taps:-taps]
        fvccpp = fvccpp[taps:-taps]
        fvepp = fvepp[taps:-taps]
        falpp = falpp[taps:-taps]
        faxpp = faxpp[taps:-taps]
        flevpp = flevpp[taps:-taps]

        faa_tf = np.append(faa_tf, fgpp)
        vcc_tf = np.append(vcc_tf, fvccpp)
        ve_tf = np.append(ve_tf, fvepp)
        al_tf = np.append(al_tf, falpp)
        ax_tf = np.append(ax_tf, faxpp)
        le_tf = np.append(le_tf, flevpp)

    faa_tf = np.array(faa_tf).flatten()
    vcc_tf = np.array(vcc_tf).flatten()
    ve_tf = np.array(ve_tf).flatten()
    al_tf = np.array(al_tf).flatten()
    ax_tf = np.array(ax_tf).flatten()
    le_tf = np.array(le_tf).flatten()

    # solve for curvature equation by simple regression (OLS)
    df = DataFrame({'ve': ve_tf, 'vcc': vcc_tf, 'al': al_tf,
                   'ax': ax_tf, 'lev': le_tf, 'g': -faa_tf})
    x = df[['ve', 'vcc', 'al', 'ax', 'lev']]
    y = df['g']
    x = add_constant(x)
    model = OLS(y, x).fit()
    return df, model

########################################################################
# things loosely connected to tilt corrections
########################################################################


def center_diff(y, n, samp):
    """ Numerical derivatives, central difference of nth order

    :param y: data to differentiate
    :type y: array_like
    :param n: order, either 1 or 2
    :type n: int
    :param samp: sampling rate
    :type samp: float

    :return: 1st or 2nd order derivative of **y**
    """
    if n == 1:
        return (y[2:] - y[:-2])*(samp/2)
    elif n == 2:
        return (y[:-2] - 2*y[1:-1] + y[2:])*(samp**2)
    else:
        print('bad order for derivative')
        return -999


def up_vecs(dt, g, cacc, lacc, on_off, cper, cdamp, lper, ldamp):
    """ Calculate 3xN matrix of platform up-pointing vectors in (cross, long) coordinates

    :param dt: sampling interval in seconds
    :type dt: float
    :param g: latitudinal correction term (2nd orfer FA plus ellipsoid)
    :type g: array_like
    :param cacc: cross-axis acceleration
    :type cacc: array_like
    :param lacc: long-axis acceleration
    :type lacc: array_like
    :param on_off: flag for "good" data - can be used to zero out data points when the meter
        is clamped or otherwise not operational
    :type on_off: array_like
    :param cper: platform period in seconds for the cross-axis tilt filter
    :type cper: float
    :param cdamp: platform damping term for cross-axis tilt filter
    :type cdamp: float
    :param lper: platform period in seconds for the long-axis tilt filter
    :type lper: float
    :param ldamp: platform damping term for long-axis tilt filter
    :type ldamp: float

    :return: **up_vecs** (*3xN ndarray*) - cross, long, and up vectors for the platform

    """
    # clean out any nans in the accelerations
    cacc[np.isnan(cacc)] = 0
    lacc[np.isnan(lacc)] = 0

    # apply the mysterious on_off
    cacc[on_off > 0] = 0
    lacc[on_off > 0] = 0

    # make the cross-axis tilt filter
    cnum, cden = _tilt_filter(cper, dt, cdamp)

    # calculate cross-axis driving term, do tilt filtering
    drive = cacc/g
    drive[np.isnan(drive)] = 0
    ctilt = lfilter(cnum, cden, drive)

    # repeat all that for the long axis
    lnum, lden = _tilt_filter(lper, dt, ldamp)
    drive = lacc/g
    drive[np.isnan(drive)] = 0
    ltilt = lfilter(lnum, lden, drive)

    # combine the pieces to get the platform up-pointing vectors
    up_vecs = _calc_up_vecs(np.arctan(ctilt), np.arctan(ltilt))

    return up_vecs


def _tilt_filter(per, dt, damp=False):
    """ Filter coefficients for L&R platform tilt computation

    :param per: platform period, seconds
    :type per: float
    :param dt: sample increment, seconds
    :type dt: float
    :param damp: platform damping term (default: sqrt(2)/2)
    :type damp: float, optional

    :return: filter coefficients
    """

    if not damp:
        damp = np.sqrt(2)/2

    # frequencies:
    w0 = (2*np.pi)/per
    ws = 1/np.sqrt(6371100/9.8)

    om0 = (2/dt)*np.tan(w0*dt/2)
    omS = (2/dt)*np.tan(ws*dt/2)

    # first stage substitutions
    a = (omS**2) - (om0**2)
    b = 2*damp*om0*(2/dt)
    c = 4/(dt**2)
    d = om0**2

    # second stage substitutions
    d0 = b + c + d
    b0 = (a - b)/d0
    b1 = 2*a/d0
    b2 = (a + b)/d0
    a1 = 2*(d - c)/d0
    a2 = (c + d - b)/d0

    num = np.array([b0, b1, b2])
    den = np.array([1, a1, a2])

    return num, den


def _calc_up_vecs(ctilt, ltilt):
    """ calculate 3xN matrix of platform up-vectors in (cross, long, up) coordinates

    :param ctilt: cross-axis tilt angles in radians
    :type ctilt: array_like
    :param ltilt: long-axis tilt angles in radians
    :type ltilt: array_like

    :return: **up_vecs** (*3xN ndarray*) - (cross, long, up) for platform
    """

    # get increments, assuming initial is 0
    inc_ct = np.append(ctilt[0], np.diff(ctilt))
    inc_lt = np.append(ltilt[0], np.diff(ltilt))

    # trig functions of tilts
    sc = np.sin(ctilt)
    cc = np.cos(ctilt)
    sl = np.sin(ltilt)
    cl = np.cos(ltilt)

    # set up array to hold output
    n = len(ctilt)
    up_vecs = np.zeros((3, n))

    # do rotations
    for i in range(n):
        # rotation matrics
        rotc = np.array([[cc[i], 0, -sc[i]], [0, 1, 0], [sc[i], 0, cc[i]]])
        rotl = np.array([[1, 0, 0], [0, cl[i], -sl[i]], [0, sl[i], cl[i]]])

        # alternate order of rotations for reasons that I do not understand
        plat_up = [0, 0, 1]  # start with vertical platform
        if i % 2 == 0:
            plat_up = np.matmul(rotc, np.matmul(rotl, plat_up))
        else:
            plat_up = np.matmul(rotl, np.matmul(rotc, plat_up))

        up_vecs[:, i] = plat_up

    return up_vecs

########################################################################
# MBA and RMBA functions (including thermal models)
########################################################################


def grav1d_padded(xtopo, topo, zlev, rho):
    """Calculate the gravity anomaly due to a density contrast across topography, along a line.

    The input (1D) topography is padded on both ends to reduce edge effects

    This function uses the method from Parker and Blakely:

        R. L. Parker (1972). The Rapid Calculation of Potential Anomalies,
        Geophys J R astr Soc 31, 447-455, DOI: 10.1111/j.1365-246X.1973.tb06513.x

        R. J. Blakely (1995). "Ch. 11: Fourier-Domain Modeling" in **Potential Theory in Gravity 
        and Magnetic Applications**, Cambridge University Press, DOI: 10.1017/CBO9780511549816

    .. Original 2D function written by Mark Behn, November 6, 2003
       Translated to Python in 1D with padding by Hannah Mark, 6 October 2017

    :param xtopo: x coordinates of the surface in meters (must be equally spaced)
    :type xtopo: array_like
    :param topo: z coordinates of the surface in meters
    :type ztopo: array_like
    :param zlev: vertical distance for upwards continuation in meters.
    :type zlev: float
    :param rho: density contrast across the topography in kg/m^3.
    :type rho: float

    :return: **anom** (*ndarray*) - gravity anomaly in mgal
    """
    G = 6.673*1e-8
    grav = 2*np.pi*G*rho
    baselev = np.mean(topo)  # mean topography for baseline

    # spacing of coordinates, must be constant
    dx = (xtopo[-1]-xtopo[0])/(len(xtopo)-1)
    nx = len(xtopo)

    wing = np.ones(2*len(topo))  # *** extra padding

    # extend or mirror the profile
    padtopo = np.append(topo[0]*wing, np.append(topo, topo[-1]*wing))
    # padtopo = np.append(baselev*wing,np.append(topo,baselev*wing))
    # padtopo = np.append(topo,topo[::-1])
    nxt = len(padtopo)
    mfx = (nxt/2.) + 1

    k = (2*np.pi)/(nxt*dx)  # calculate wavenumbers
    k2 = k**2
    xi1 = np.arange(1, mfx+1)
    xi = np.append(xi1, xi1[-2:0:-1])
    xxk = (xi-1)*(xi-1)*k2
    kwn1 = np.sqrt(xxk[:nxt])

    Ftopo = np.fft.fft(padtopo)  # Fourier transform the topography
    Ftopo[0] = 0

    npower = 5
    SUMtopo = copy(Ftopo)
    for ip in range(2, npower+1):  # summation per eq. 11.41 in Blakely
        Ftopo = np.fft.fft(padtopo**ip)
        Ftopo = Ftopo*(kwn1**(ip-1))/factorial(ip)

        SUMtopo = SUMtopo + Ftopo

    data = SUMtopo*grav*np.exp(-(zlev-baselev)*kwn1)  # upward continuation

    data = np.fft.ifft(data)  # inverse Fourier transform
    anom = np.real(data[2*nx:3*nx])*100  # factor of 100 for mgal output

    return anom


def grav2d_folding(X, Y, Z, dx, dy, drho=0.6, dz=6000, ifold=True, npower=5):
    """
    Parker [1972] method for calculating gravity from 2D topographic surface with a density contrast.

    The `ifold` option enables folding the input topography grid in x and y to mitigate edge effects

    .. Modified from parker.m by Mark Behn, which was in turn modified from
       parker.f by Ban-Yuan Kuo and Jian Lin.

    :param X: vector of N X coordinates
    :type X: ndarray
    :param Y: vector of M Y coordinates
    :type Y: ndarray
    :param Z: matrix of Z coordinates
    :type Z: ndarray, NxM
    :param dx: x grid spacing, for wavenumbers [km]
    :type dx: float
    :param dy: y grid spacing, for wavenumbers [km]
    :type dy: float
    :param drho: density contrast across surface. Ex: 1.7 for water to crust, 0.6 for crust to mantle
    :type drho: float
    :param dz: offset depth for layer interface, added to baselevel for upward continuation [m]
    :type dz: float
    :param ifold: switch for folding
    :type ifold: bool
    :param npower: power of Taylor series expansion (default: 5)
    :type npower: int

    :return: (*ndarray*) gravity anomaly in mgal
    """
    nx = len(X)  # size of the grid
    ny = len(Y)

    zmax = max(Z.flatten())   # get min and max for scaling
    zmin = min(Z.flatten())
    if zmax < -1e10:
        zmax = -1e10
    if zmin > 1e10:
        zmin = 1e10

    slev = np.mean(Z.flatten())

    Z = 100*(slev-Z)
    slev = slev*100
    dx = 100000*dx
    dy = 100000*dy
    dz = dz*100

    # to fold or not to fold--
    if ifold:
        nxt = nx*2
        nyt = ny*2
    else:
        nxt = nx
        nyt = ny

    G = 6.673e-8   # gravitational constant
    conv = 1000  # gals to mgals
    grav1 = 2*np.pi*G*conv

    grav = grav1*drho

    # for wavenumbers
    kint1 = (2*np.pi)/(nxt*dx)
    kint2 = (2*np.pi)/(nyt*dy)
    kx2 = kint1*kint1
    ky2 = kint2*kint2

    # folding frequency:
    mfx = int(nxt/2 + 1)
    mfy = int(nyt/2 + 1)
    mfx2 = mfx*2
    mfy2 = mfy*2

    # the important part:
    # (1) compute wavenumbers
    # (2) transform topography with fft
    # (3) sum over powers with those wavenumbers
    # (4) upward continue, transform back to space domain, multiply
    #     by constants

    # wavenumbers:
    yj1 = np.arange(1, mfy+1)
    yj = np.append(yj1, yj1[mfy-2:0:-1])
    yyk = (yj-1)*(yj-1)*ky2

    xi1 = np.arange(1, mfx+1)
    xi = np.append(xi1, xi1[mfx-2:0:-1])
    xxk = (xi-1)*(xi-1)*kx2

    kwn1 = np.zeros((nxt, nyt))
    for j in range(nyt):
        kwn1[:, j] = np.sqrt(xxk + yyk[j])

    kwn1 = kwn1.T

    # fold the data
    if ifold == 1:
        Z = np.vstack((Z, np.flipud(Z)))  # fold in X
        Z = np.hstack((Z, np.fliplr(Z)))  # fold in Y

    # first power
    data = np.copy(Z)
    data = np.fft.fft2(data)
    data[0, 0] = 0

    # sum over the other powers
    csum = np.copy(data)  # for adding summation terms
    fact = 1
    for i in range(2, npower+1):
        fact = fact*i
        data = np.copy(Z)**i
        data = np.fft.fft2(data)

        data = data*(kwn1**(i-1))/fact
        k, l = np.where(kwn1 == 0)
        data[k, l] = 0

        csum = csum + np.copy(data)

    # upward continuation
    zlev = slev+dz
    data = csum*grav*np.exp(-zlev*kwn1)

    data = np.fft.ifft2(data)
    sdata = np.real(data[:ny, :nx])  # back in the spatial domain

    return sdata


def grav2d_layer_variable_density(rho, dx, dy, z1, z2):
    """
    Calculate the gravity contribution from a layer of equal thickness with
    an inhomogenous density distribution in x and y (homogeneous in z)

    .. Based on glayer.m by Mark Behn

    :param rho: 2D density distribution [kg/m^3]
    :type rho: ndarray
    :param dx,dy: sample intervals in km
    :type dx,dy: float
    :param z1,z2: depth to top and bottom of layer in km (both >0)
    :type z1,z2: float

    :return: (*ndarray*) gravity in mgal
    """

    si2mg = 1e5
    km2m = 1e3
    G = 6.673e-11
    grav = 2*np.pi*G

    ny, nx = rho.shape
    dkx = 2*np.pi/(nx*dx)
    dky = 2*np.pi/(ny*dy)

    ifrho = np.fft.fft2(rho)  # take 2D fft of densities

    crho = np.empty((ny, nx), dtype=complex)
    for j in range(nx):
        for i in range(ny):
            kx, ky = _kvalue(i, j, nx, ny, dkx, dky)
            k = np.sqrt(kx**2 + ky**2)
            if k == 0:
                crho[i, j] = 0
            else:
                crho[i, j] = ifrho[i, j]*grav*(np.exp(-k*z1)-np.exp(-k*z2))/k

    grho = np.fft.ifft2(crho)
    grho = np.real(grho)*si2mg*km2m

    return grho


def _kvalue(i, j, nx, ny, dkx, dky):
    """
    Get wavenumber coordinates of one element of a rectangular grid

    :param i,j: indices in ky,kx directions
    :type i,j: int
    :param nx,ny: dimensions of the grid in the ky,kx directions
    :type nx,ny: int
    :param dkx,dky: sample intervals in kx,ky directions
    :type dkx,dky: float

    :return: **kx, ky** (*float*) - x and y wavenumbers at (i, j)
    """

    nyqx = nx/2+1
    nyqy = ny/2+1

    if j <= nyqx:
        kx = (j)*dkx
    else:
        kx = (j-nx)*dkx

    if i <= nyqy:
        ky = (i)*dky
    else:
        ky = (i-ny)*dky
    return kx, ky


def therm_halfspace(x, z, u=0.01, Tm=1350, time=False, rhom=3300, rhow=1000,
                    a=3.e-5, k=1.e-6):
    """Calculate thermal structure for a half space cooling model.

    Reference:

        D. Turcotte & G. Schubert (2014). Geodynamics. Cambridge
        University Press. DOI: 10.1017/CBO9780511843877
        Relevant pages: 161-162, 174-176 in 2nd or 3rd ed?

    .. Written by Mark D. Behn, November 20, 2003.
       Translated to Python + modded for plate age by Hannah Mark, October 2017

    :param x: vector of across-axis distance (meters) OR of plate ages (Myr)
    :type x: array_like
    :param z: vector of depth (meters)
    :type z: array_like
    :param u: spreading rate (m/yr)
    :type u: float
    :param time: switch for x vs age input: if ages, set time=True
        and u will be ignored.
    :type time: bool
    :param rhom: mantle density, kg/m^3, default 3300
    :type rhom: float, optional
    :param rhow: water density, kg/m^3, default 1000
    :type rhow: float, optional
    :param a: coefficient of thermal expansion, m^2/sec, default 3e-5
    :type a: float, optional
    :param k: thermal diffusivity, m^2/sec, default 1e-6
    :type k: float, optional

    :return:
        - **T** (*ndarray*) - gridded temperature over (x, z)
        - **W** (*ndarray*) - seafloor subsidence in meters
    """

    To = 0  # surface temperature [K]
    # a = 6.e-5  # coeff of thermal expansion [m^2/sec]  # used for SCARF calcs (???)
    # k = 2.e-6  # thermal diffusivity [m^2/sec]

    secyr = 365.25*24*3600  # seconds per year

    if time:
        x = x*1e6*secyr  # convert Myr to sec
    elif not time:
        x = abs(x)/(u/secyr)  # convert m to sec

    X, Z = np.meshgrid(x, z)  # grid up x,z pairs

    # mantle temperature
    T = To + (Tm-To) * erf(Z/(2*(k*X)**.5))

    # seafloor subsidence
    W = ((2*rhom*a*(Tm-To))/(rhom-rhow)) * \
        (((k*X[0, :])/(np.pi))**.5)

    return T, W


def therm_Z_halfspace(x, T, u=0.01, Tm=1350, time=False, rhom=3300, rhow=1000,
                      a=3.e-5, k=1.e-6):
    """Calculate depth to an isotherm for a half-space cooling model.

    :param x: vector of across-axis distance [m] OR plate age [Myr]
    :type x: array_like
    :param T: isotherm of choice [K]
    :type T: float
    :param u: spreading rate [m/yr], default 0.01
    :type u: float
    :param time: switch for x vs age input - if time=True,
        u is ignored, default False
    :type time: bool
    :param Tm: mantle potential temperature [K], default 1350
    :type Tm: float, optional
    :param rhom: mantle density, kg/m^3, default 3300
    :type rhom: float, optional
    :param rhow: water density, kg/m^3, default 1000
    :type rhow: float, optional
    :param a: coefficient of thermal expansion, m^2/sec, default 3e-5
    :type a: float, optional
    :param k: thermal diffusivity, m^2/sec, default 1e-6
    :type k: float, optional

    :returns:
        - **Z** (*ndarray*) - depth of this isotherm below the seafloor in meters
        - **W** (*ndarray*) - seafloor subsidence in meters
    """

    To = 0  # surface temperature [K]

    secyr = 365.25*24*3600  # seconds per year

    if time:
        x = x*1e6*secyr  # convert Myr to sec
    elif not time:
        x = abs(x)/(u/secyr)  # convert m to sec

    Z = 2*np.sqrt((k*x))*erfcinv((T-Tm)/(To-Tm))

    W = ((2*rhom*a*(Tm-To))/(rhom-rhow)) * \
        (((k*x)/(np.pi))**.5)

    return Z, W


def therm_plate(x, z, u=0.01, zL0=100.e3, Tm=1350, time=False, rhom=3300, rhow=1000,
                a=3.e-5, k=1.e-6):
    """Calculate thermal structure for the plate cooling model.  

    Reference:

        D. Turcotte & G. Schubert (2014). Geodynamics. Cambridge
        University Press. DOI: 10.1017/CBO9780511843877

    .. Written by Mark D. Behn, November 20, 2003.
       Translated to Python + modded for plate age by Hannah Mark, October 2017

    :param x: vector of across-axis distance (meters) OR plate age (Myr)
    :type x: array_like
    :param z: vector of depth (meters)
    :type z: array_like
    :param u: spreading rate (m/yr), default 0.01
    :type u: float
    :param zL0: plate thickness (meters), default 100e3
    :type zL0: float
    :param time: switch for x vs age input. If time=True, u is ignored.
    :type time: bool
    :param Tm: mantle potential temperature (K), default 1350
    :type Tm: float, optional
    :param rhom: mantle density, kg/m^3, default 3300
    :type rhom: float, optional
    :param rhow: water density, kg/m^3, default 1000
    :type rhow: float, optional
    :param a: coefficient of thermal expansion, m^2/sec, default 3e-5
    :type a: float, optional
    :param k: thermal diffusivity, m^2/sec, default 1e-6
    :type k: float, optional

    :returns:
        - **T** (*ndarray*) - gridded temperature over (x, z)
        - **W** (*ndarray*) - seafloor subsidence in meters
    """

    To = 0  # surface temperature [K]

    secyr = 365.25*24*3600  # seconds per year

    X, Z = np.meshgrid(x, z)

    if time:
        t = X*1e6*secyr  # convert Myr to sec
    elif not time:
        t = abs(X)/(u/secyr)  # convert across-axis distance to time in SECONDS

    Tterm2 = (2/np.pi)*np.exp(-k*(np.pi**2)*t/(zL0**2))*np.sin(np.pi*Z/zL0)
    Tterm3 = (1/np.pi)*np.exp(-4*k*(np.pi**2)*t/(zL0**2))*np.sin(2*np.pi*Z/zL0)
    Tterm4 = (2/np.pi/3)*np.exp(-9*k*(np.pi**2)
                                * t/(zL0**2))*np.sin(3*np.pi*Z/zL0)
    Tterm5 = (2/np.pi/4)*np.exp(-16*k*(np.pi**2)
                                * t/(zL0**2))*np.sin(4*np.pi*Z/zL0)
    Tterm6 = (2/np.pi/5)*np.exp(-25*k*(np.pi**2)
                                * t/(zL0**2))*np.sin(5*np.pi*Z/zL0)
    Tterm7 = (2/np.pi/6)*np.exp(-36*k*(np.pi**2)
                                * t/(zL0**2))*np.sin(6*np.pi*Z/zL0)
    Tterm8 = (2/np.pi/7)*np.exp(-49*k*(np.pi**2)
                                * t/(zL0**2))*np.sin(7*np.pi*Z/zL0)
    Tterm9 = (2/np.pi/8)*np.exp(-64*k*(np.pi**2)
                                * t/(zL0**2))*np.sin(8*np.pi*Z/zL0)
    Tterm10 = (2/np.pi/9)*np.exp(-81*k*(np.pi**2)
                                 * t/(zL0**2))*np.sin(9*np.pi*Z/zL0)
    Tterm11 = (2/np.pi/10)*np.exp(-100*k*(np.pi**2)
                                  * t/(zL0**2))*np.sin(10*np.pi*Z/zL0)

    T = To + (Tm - To)*(Z/zL0 + Tterm2 + Tterm3 + Tterm4 + Tterm5 + Tterm6 + Tterm7 +
                        Tterm8 + Tterm9 + Tterm10 + Tterm11)

    mantle = np.where(Z > zL0)[0]
    T[mantle] = Tm

    Wterm2 = (4/np.pi**2)*np.exp(-k*(np.pi**2)*t[0, :]/(zL0**2))
    Wterm3 = (4/(9*np.pi**2))*np.exp(-k*9*(np.pi**2)*t[0, :]/(zL0**2))
    Wterm4 = (4/(25*np.pi**2))*np.exp(-k*25*(np.pi**2)*t[0, :]/(zL0**2))

    W = ((rhom*a*(Tm-To)*zL0)/(rhom-rhow)) * (1./2 - Wterm2 - Wterm3 - Wterm4)

    return T, W


def therm_Z_plate(x, T, u=0.01, zL0=100.e3, Tm=1350, time=False,
                  minz=0, maxz=100e3, zsp=1e2, rhom=3300, rhow=1000,
                  a=3.e-5, k=1.e-6):
    """Calculate approximate depth to an isotherm in the plate cooling model

    This is done by calculating a temperature field with a decent z spacing 
    and finding the closest points to the isotherm, so it depends strongly 
    on the z spacing that you use. If you need depth to multiple isotherms
    it's most efficient to get them all at once (using a longer array for
    T) so the whole temperature field is only calculated one time.

    :param x: array of across-axis distance (meters) OR plate age (Myr)
    :type x: array_like
    :param T: temperatures for which you want isotherms (K)
    :type T: array_like
    :param u: spreading rate (m/yr), default 0.01
    :type u: float
    :param zL0: plate thickness (meters), default 100e3
    :type zL0: float
    :param Tm: mantle potential temperature (K), default 1350
    :type Tm: float, optional
    :param time: switch for x vs age input. If time=True, u is ignored.
    :type time: bool
    :param minz: minimum z for calculating T field (meters), default 0
    :type minz: float
    :param maxz: maximum z (meters), default 100e3
    :type maxz: float
    :param zps: z spacing for grid (meters), default 1e3
    :type zps: float
    :param rhom:  mantle density, kg/m^3, default 3300
    :type rhom: float, optional
    :param rhow: water density, kg/m^3, default 1000
    :type rhow: float, optional
    :param a: coefficient of thermal expansion, m^2/sec, default 3e-5
    :type a: float, optional
    :param k: thermal diffusivity, m^2/sec, default 1e-6
    :type k: float, optional

    :returns: **ziso** (*ndarray*) - depths to isotherms, z(T,x)
    """

    z = np.arange(minz, maxz, zsp)

    Tp, _ = therm_plate(x, z, u=u, zL0=zL0, Tm=Tm, time=time, rhom=rhom, rhow=rhow,
                        a=a, k=k)  # calculate the whole temperature field

    ziso = np.zeros((len(T), len(x)))
    for i in range(len(T)):  # loop isotherms, find depths
        for j in range(len(x)):
            ziso[i, j] = z[np.argmin(abs(Tp[:, j]-T[i]))]

    return ziso

########################################################################
# crustal thickness functions
########################################################################


def crustal_thickness_2D(ur, nx=1000, ny=1, dx=1.3, dy=0, zdown=10, rho=0.4,
                         wlarge=45, wsmall=25, back=False):
    """
    Downward continuation of gravity to "topographic relief" ie crustal thickness

    This can be used in 2D, but also works for a single line
    given ny=1 (which is the default)

    .. Written by Hannah Mark (MIT/WHOI), October 2017
       Modeled on down_2d.f by Jian Lin (WHOI)

    :param ur: residual gravity anomaly, mgal
    :type ur: array_like
    :param nx: number of points in x direction, default 1000
    :type nx: int
    :param ny: number of points in y direction, default 1 (>1 for 2D)
    :type ny: int
    :param dx: spacing between x points, km, default 1.3
    :type dx: float
    :param dy: spacing between y points, km, default 0 (>0 for 2D)
    :type dy: float
    :param zdown: downward continuation depth, km, default 10
    :type zdown: float
    :param rho: density difference crust to mantle, g/cm^3, default 0.4
    :type rho: float
    :param wlarge: max wavelength for taper/cutoff, km, default 45
    :type wlarge: float
    :param wsmall: min wavelength for taper/cutoff, km, default 25
    :type wsmall: float
    :param back: switch for doing reverse tranform
    :type back: bool

    :returns:
        - **crustal thickness** (*ndarray*) - thickness variation in km
        - **recovered gravity** (*ndarray*) - back-calculated RMBA, (optional, if back=True)
    """
    assert wlarge > wsmall, 'wlarge must be larger than wsmall'

    zmin = min(ur)
    zmax = max(ur)
    ave = np.mean(ur)

    # shift to a new reference and convert milligal to gal
    # sign is switched so that positive residual = crustal thinning
    ur = -0.001*(ur - ave)

    # convert km to cm for spatial params
    dx = dx*1e5
    dy = dy*1e5
    zdown = zdown*1e5

    # calculate wavenumbers for taper/cutoff
    # we will taper btwn kcut1 and kcut2; cutoff >kcut2
    wlarge = wlarge*1e5   # convert km to cm
    wsmall = wsmall*1e5
    kcut1 = 2*np.pi/wlarge
    kcut2 = 2*np.pi/wsmall
    dkcut = kcut2 - kcut1
    mfx = nx/2 + 1
    mfy = ny/2 + 1

    prod = 1./(nx*ny)  # dimension correction factor

    G = 6.673e-8  # cgs gravity
    topo1 = 1/(2*np.pi*G*rho)  # gravity-to-topography transfer

    kint1 = 2*np.pi/(nx*dx)
    if dy != 0:
        kint2 = 2*np.pi/(ny*dy)
    else:
        kint2 = 0

    kwn1 = np.zeros((nx, ny))  # compute wavenumbers in 2D
    for j in range(ny):
        yj = j
        if j > mfy:
            yj = mfy*2 - j
        yyk = kint2**2 * (yj-1)**2

        for i in range(nx):
            xi = i
            if i > mfx:
                xi = mfx*2 - i
            xxk = kint1**2 * (xi-1)**2

            kwn1[i, j] = np.sqrt(xxk + yyk)

    # Fourier transform of the gravity residual
    # this was passed as a 1D array even for a 2D problem
    ur_arr = ur.reshape(nx, ny)
    ur_arr_ft = np.fft.fft2(ur_arr)  # 2D fft

    # apply wavenumbers, taper
    for j in range(ny):
        for i in range(nx):
            wgt = 1
            if kwn1[i, j] > kcut2:
                wgt = 0
            elif kwn1[i, j] > kcut1 and kwn1[i, j] <= kcut2:
                t = (kwn1[i, j]-kcut1)/dkcut*np.pi
                wgt = (np.cos(t)+1)/2

            ur_arr_ft[i, j] = ur_arr_ft[i, j] * \
                np.exp(kwn1[i, j]*zdown)*topo1*wgt

    # inverse Fourier transform
    ur_arr_2 = np.fft.fft2(ur_arr_ft)

    # back to vector, correct for dimensions, convert to km
    ur_arr_2 = ur_arr_2.reshape(-1, 1)
    ur_arr_2 = ur_arr_2*prod/1e5

    if not back:
        return ur_arr_2
    elif back:  # reverse the transform and upward continute to check gravity recovery
        for j in range(ny):
            for i in range(nx):
                ur_arr_ft[i, j] = -ur_arr_ft[i, j] * \
                    np.exp(-kwn1[i, j]*zdown)/topo1
        ur_back = np.fft.fft2(ur_arr_ft)
        ur_back = ur_back.reshape(-1, 1)
        ur_back = ur_back*prod*1000+ave

        return ur_arr_2[::-1], ur_back[::-1]
