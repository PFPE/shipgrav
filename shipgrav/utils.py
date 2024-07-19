import numpy as np
from datetime import datetime
from pandas import Series
import re

# TODO gaussian filter doesn't *need* time array


def gaussian_filter(x, fl):
    """ Apply a gaussian filter to a vector.

    The filtering is done via a ring buffer; results are not identical to 
    scipy.ndimage.gaussian_filter, which is why this function exists.
    Note that the filter is not applied symmetrically and there is a shift by fl.

    :param x: time series data to be filtered.
    :type x: array_like
    :param fl: length of the filter in # of samples.
    :type fl: int

    :return: **xfilt** (*ndarray*) - filtered time series x
    """

    assert len(x) > fl, 'raw data is shorter than gaussian filter'

    coeffs = _gaussian_coeffs(fl)

    # set up ring buffer
    filt_len = len(coeffs)
    ring_buffer = np.zeros(filt_len)

    xfilt = np.zeros(len(x))

    for i in range(len(x)):
        ring_buffer = np.append(ring_buffer[1:], x[i])
        filt_vec = ring_buffer*coeffs

        xfilt[i] = sum(filt_vec)

    return xfilt


def _gaussian_coeffs(fl):
    """ Calculate a gaussian filter to convolve with things that need filtering.

    :param fl: length of the filter in # of samples
    :type fl: int

    :return: **gauss_prms** (*ndarray*) - filter coefficients
    """
    half_sample = np.floor(.5*fl)
    coeff_len = int(half_sample + 1)

    frac = 6./fl  # same as NAVO and LDEO versions, and I think this is the secret sauce?

    gauss_prms = np.zeros(coeff_len)
    m = 0
    for i in range(coeff_len, 0, -1):
        x = (half_sample - i)*frac
        x = -.5*x**2
        gauss_prms[m] = np.exp(x)
        m += 1

    # reflect
    gauss_prms = np.hstack(
        (gauss_prms[1:][::-1], gauss_prms[0], gauss_prms[1:]))

    # normalize
    gauss_prms = gauss_prms/sum(gauss_prms)

    return gauss_prms


def decode_dgs_status_bits(stat, key='status'):
    """ Decode status bits from integers in DGS gravimeter files.

    Return flags as a dict (for one integer input) or a dataframe (status df input)

    :param stat: status bit(s)
    :type stat: int or pd.DataFrame
    :param key: the DataFrame column for the status bits if input is a DataFrame
    :type key: string, optional

    :return: **stat** (*dict or pd.DataFrame*) - decoded bits
    """
    # flags from Jasmine's code
    # flags = ['clamp','unclamp','GPSsync','feedback','R1','R2','ADlock','Rcvd',\
    #        'NavMod1','NavMod2','Free','SensCom','GPStime','ADsat','Datavalid','PlatCom']
    # flags from DGS documentation?
    flags = ['clamp status', 'GPSsync', 'reserved', 'feedback', 'R1', 'R2', 'ADlock', 'ack host',
             'NavMod1', 'NavMod2', 'dgs1_trouble', 'dgs2_trouble', 'GPStime', 'ADsat', 'nemo1', 'nemo2']

    assert type(stat) == int or type(
        stat) == DataFrame, 'bad input type for status bits'
    if type(stat) == int:
        bt = format(stat, '016b')
        out = {}
        for i, f in enumerate(flags):
            out[f] = bt[i]
        return out
    else:  # dataframe, add columns
        bts = np.array([format(s, '016b') for s in stat[key]])
        for i, f in enumerate(flags):
            stat.insert(i+1, f, [b[i] for b in bts])
        return stat


def clean_ini_to_toml(ini_file):
    """ Read in a .ini file and try to rewrite as toml-compliant.

    This uses simple, prescriptive regex stuff to (hopefully) clean out
    ini conventions that don't work with toml.
    It writes out a toml file with the same name as the input, with
    the extension .ini replaced by .toml

    :param ini_file: path to input ini file
    """

    with open(ini_file, 'r') as file:
        text = file.read()  # just read the whole thing

    # fix comments (// to #)
    text = re.sub('//', '#', text)

    # clean out tabs
    text = re.sub('\t', '  ', text)

    # fix TRUE and FALSE
    text = re.sub('TRUE', 'true', text)
    text = re.sub('FALSE', 'false', text)

    text = re.sub('$', '', text)

    # fix unquoted strings and write
    fo = open(ini_file.rstrip('ini')+'toml', 'w')
    lines = text.split('\n')
    for ln in lines:
        if ln.startswith('$'):
            ln = re.sub('\$', '', ln)  # clean out $ for talkers

        if ln.startswith('[') or ln.startswith('#') or ln == '':
            fo.write(ln)
            fo.write('\n')
            continue

        val = ln.split('=')[1]  # get value from key=value pair
        extra_comm = ''
        if '#' in val:
            val = val.split('#')[0]
            extra_comm = '='.join(val.split('#')[1:])
        val = val.strip()

        if val == 'true' or val == 'false':  # skip bools
            fo.write(ln)
            fo.write('\n')
            continue
        if val.startswith("\""):  # skip strings already quoted
            fo.write(ln)
            fo.write('\n')
            continue
        try:
            float(val)
        except:
            val = "\"" + val + "\""
            ln = '= '.join((ln.split('=')[0], val))
            if extra_comm != '':
                ln = '   #'.join((ln, extra_comm))
        fo.write(ln)
        fo.write('\n')

    fo.close()
    return


class _SnappingCursor:
    """A cross-hair cursor that snaps to the closest *x* point on a line

    This is for a two-axis figure where the first axis (ax1) 
    has a cross-hair cursor, and the second axis (ax2) has a dot that's linked
    to the cursor in ax1. Basically, we can plot two different values for one
    time series, and use a cursor over one of them to move a dot around on the other one.

    The time series with the cross hair cursor must have x values that are unique and sorted.

    Since this class is designed for a specific use case, there are some hard-coded assumptions
    about the names of columns in the DataFrame being used for plotting.

    .. The specific use case here is a plot of FAA vs time (so, x unique and sorted) with 
       the cursor, and a dot moving around on a map view of a ship track (x not necessarily
       unique or sorted).

    .. The two are linked via the pandas DataFrame "data", which has columns *tsec*,
       *lon_new*, and *lat_new* 

    The cursor can also register mouse clicks so the user can interactively select points
    that are accessible later.

    For usage, see the example script interactive_line_pick.py included with the
    shipgrav package

    Heavily adapted from the matplotlib cross-hair cursor demo at
    https://matplotlib.org/stable/gallery/event_handling/cursor_demo.html

    :param ax1: first axis, for cursor; x axis unique and sorted (tsec in data)
    :param ax2: second axis, for points; lon_new vs lat_new in data
    :param line: the line plotted in ax1, of type matplotlib.lines.Line2D
    :param dot: single point plotted in ax2 that will move around, also Line2D
    :param data: pd.DataFrame with tsec, lon_new, and lat_new columns that correspond
        to what's plotted in both axes
    """

    def __init__(self, ax1, ax2, line, dot, data):
        self.ax1 = ax1
        self.dot = dot
        self.data = data
        minx = ax1.get_xlim()[0]
        maxy = ax1.get_ylim()[1]
        self.horizontal_line = ax1.axhline(y=maxy, color='k', lw=0.8, ls='--')
        self.vertical_line = ax1.axvline(x=minx, color='k', lw=0.8, ls='--')
        self.x, self.y = line.get_data()
        self._last_index = None

        self.x_seg = []
        self.y_seg = []
        self.i_seg = []
        self.scatters = ax2.plot(
            self.x_seg, self.y_seg, marker='o', color='r', linestyle=None)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes == self.ax1:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax1.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[index]
            y = self.y[index]
            # update the line positions for the cursor
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])

            # and also update ax2 dot
            ind_x = np.argmin(abs(x - self.data['tsec']))
            self.dot.update({'xdata': [self.data.iloc[ind_x]['lon_new'],], 'ydata': [
                            self.data.iloc[ind_x]['lat_new'],]})
            self.ax1.figure.canvas.draw()

    def on_mouse_click(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            ind_x = np.argmin(abs(x - self.data['tsec']))
            self.x_seg.append(self.data.iloc[ind_x]['lon_new'])
            self.y_seg.append(self.data.iloc[ind_x]['lat_new'])
            self.i_seg.append(ind_x)
            self.scatters[0].update({'xdata': self.x_seg, 'ydata': self.y_seg})

        if event.button == 3:  # pop the last element of the lists
            self.x_seg.pop()
            self.y_seg.pop()
            self.i_seg.pop()
            self.scatters[0].update({'xdata': self.x_seg, 'ydata': self.y_seg})

        self.ax1.figure.canvas.draw()
