import numpy as np

# impulse response of 10th order Taylor series differentiator
tay10 = [1/1260, -5/504, 5/84, -5/21, 5/6, 0, -5/6, 5/21, -5/84, 5/504, -1/1260]


def ll2en(lon,lat,freq=1):
    """ Convert time series of geographic position to E/N velocities.

    Lon/lat pairs should be at a constant sampling rate. The first and last five 
    values of the output arrays are filled with Nan.

    :param lon: np.ndarray of longitude points, decimal degrees
    :param lat: np.ndarray of latitude points, decimal degrees
    :param freq: frequency of the position data, Hz (default: 1 Hz)
    """

    assert hasattr(lon,'__len__') and hasattr(lat,'__len__'), 'lon and lat must be lists or arrays'
    assert len(lon) == len(lat), 'lon and lat must be the same length'
    assert len(lon) > 10, 'at least 11 data points required'

    a = 6378137.0       # WGS84 semi-major
    b = 6356752.3142451 # WGS84 semi-minor
    e2 = 1 - b**2/a**2  # ellipticity**2

    # constants for radii of curvature
    sin2lat = np.sin(np.deg2rad(lat))**2
    e2term = np.sqrt(1 - e2*sin2lat)

    # differentiate lat and lon using 10th order Taylor
    dlat = np.deg2rad(np.convolve(lat,tay10,'same'))
    dlon = np.deg2rad(np.convolve(lon,tay10,'same'))

    # convert dlon/dlat to east and north velocity, scaled by sampling frequency
    vn = a*(1 - e2)*(dlat/(e2term**3))*freq
    ve = a*dlon*(np.cos(np.deg2rad(lat))/e2term)*freq

    # Nan the edges bc they are unreliable
    vn[:5] = np.nan; vn[-5:] = np.nan
    ve[:5] = np.nan; ve[-5:] = np.nan

    return vn, ve


def vevn2cv(ve, vn):
    """Calculate velocity and heading from east and north velocities.

    Output heading is in degrees clockwise from N.
    """
    heading = np.rad2deg(np.arctan2(ve, vn))
    vel = np.sqrt(ve**2 + vn**2)
    return heading, vel


def rot_acc_EN_cl(heading, accE, accN):
    """Rotate acceleration from East/North to Cross/Long reference frame.

    :param heading: heading, in degrees clockwise from N
    :param accE: east acceleration
    :param accN: north acceleration
    """
    cosa = np.cos(np.deg2rad(heading))
    sina = np.sin(np.deg2rad(heading))
    ac = (accE*cosa) - (accN*sina)
    al = (accE*sina) + (accN*cosa)
    return ac, al
