import numpy as np
import pandas as pd
from datetime import datetime, timezone
import yaml
import mmap
import sys, os
import re

# TODO need a fix for places where we cross the international date line (read_nav)
# TODO check pandas versions for datetime parsing (eg RGS read)
# TODO where are cross and long accel for Thompson? (DGS laptop read)

########################################################################
# navigation i/o (for better synchronization of gps with gravimeter)
########################################################################

def read_nav(ship, pathlist, sampling=1, talker=None, ship_function=None):
    """ Read navigation strings from .GPS (or similar) files.

    Ships have different formats and use different talkers for preferred
    navigation; the ones we know are listed in database.toml, and there is
    also an option to override that by setting the talker kwarg.
    Navigation data is re-interpolated to the given sampling rate.

    This function returns a pandas.DataFrame with time series of 
    geographic coordinates and corresponding timestamps.

    :param ship: name of the ship, string
    :param pathlist: list of paths to navigation files (.GPS), strings
    :param sampling: sampling rate to interpolate to, default 1 Hz
    :param talker: optional override for talker, string. Default behavior is
        to use talker from database.toml if ship is listed there.
    :param ship_function: optional user-supplied function for reading from nav files.
        This function should return arrays of lon, lat, and timestamps.
        Look at _navcoords() and navdate_Atlantis() (and similar functions) for examples.
    """
    # read info on talkers for various ships
    moddir = os.path.dirname(__file__)
    import tomli as tm
    with open(os.path.join(moddir,'database.toml'),'rb') as f:
        info = tm.load(f)
    nav_str = info['nav-talkers']

    # check to make sure we have some way to read nav data for this ship
    if ship not in nav_str.keys() and ship_function == None:
        print('R/V %s not yet supported for nav read; must supply read function' % ship)
        return -999

    # check to make sure we have a talker one way or another
    if talker is not None:
        pass  # use provided talker if it's there
    else:
        if ship not in nav_str.keys():
            print('talker not known for R/V %s' % ship)
            return -999
        else:
            talker = nav_str[ship]

    if type(pathlist) == str:
        pathlist = [pathlist,]  # if just one path is given, make it into a list

    timetime = np.array([]); lonlon = np.array([]); latlat = np.array([])

    for fpath in pathlist:  # loop nav files (may be a lot of them)
        with open(fpath,'r') as f:
            allnav = np.array(f.readlines())  # read the entire file

        if ship_function:  # use a user-supplied function to get all the things
            lon, lat, timest = ship_function(allnav, talker)
        else:
            if ship == 'Atlantis':
                lon, lat = _navcoords(allnav, talker)
                timest = _navdate_Atlantis(allnav, talker)
            elif ship == 'NBP':
                lon, lat = _navcoords(allnav, talker)
                timest = _navdate_NBP(allnav, talker)
            elif ship == 'Thompson':
                lon, lat = _navcoords(allnav, talker)
                timest = _navdate_Thompson(allnav, talker)
            elif ship == 'Revelle':
                lon, lat = _navcoords(allnav, talker)
                timest = _navdate_Revelle(allnav, talker)
            elif ship == 'Ride':
                lon, lat = _navcoords(allnav, talker)
                timest = _navdate_Ride(allnav, talker)
            else:  # in theory we never get to this option, but catch just in case
                print('R/V %s not yet supported for nav read; must supply read function' % ship)
                return -999

        sec_time = np.array([d.timestamp() for d in timest])  # posix, seconds, for interpolation
        _, idx = np.unique(sec_time,return_index=True)
        samp_time = np.arange(min(sec_time),max(sec_time),sampling)  # fenceposting?

        lon_out = np.interp(sec_time,sec_time[idx],lon[idx])  # interpolate to desired sample rate
        lat_out = np.interp(sec_time,sec_time[idx],lat[idx])

        timetime = np.append(timetime,sec_time)
        lonlon = np.append(lonlon,lon_out)
        latlat = np.append(latlat,lat_out)

    # de-duplicate times just in case, and make into a DataFrame to return
    _, idx = np.unique(timetime, return_index=True)
    gps_nav = pd.DataFrame({'time_sec':timetime[idx],'lon':lonlon[idx],'lat':latlat[idx]})
    gps_nav['stamps'] = np.array([datetime.fromtimestamp(e,timezone.utc) for e in timetime[idx]],
                        dtype=datetime)

    # check if we have probable longitude jumps, try to fix them
    ilocs = np.where(abs(np.diff(gps_nav['lon'])) > 1)[0]
    if len(ilocs) > 0:
        gps_nav = _clean_180cross(gps_nav)  # try to get rid of +- jumps (NBP, often)

    return gps_nav

def _clock_time(allnav,talker):
    """Extract clock time from standard talker strings.
    """
    inav = [talker in s for s in allnav]  # find lines of file with this talker
    subnav = allnav[inav]  # select only those lines
    inav = np.where(inav)[0]  # indices in allnav of the lines that are selected
    N = len(subnav)
    hour = np.zeros(N,dtype=int); mint = np.zeros(N,dtype=int)
    sec = np.zeros(N,dtype=int); msec = np.zeros(N,dtype=int)
    
    for i in range(N):
        post = subnav[i].split(talker)[-1].lstrip().split(',')
        if post[0] == '': post = post[1:]
        hour[i] = int(post[0][:2])   # hour
        mint[i] = int(post[0][2:4])  # min
        sec0 = float(post[0][4:])  # sec.msec
        msec[i] = int(int(str(sec0).split('.')[-1])*1e4) # msec
        sec[i] = int(str(sec0).split('.')[0]) # sec

    return hour, mint, sec, msec

def _navdate_Atlantis(allnav, talker):
    """Extract datetime info from Atlantis nav files (at.*GPS).
    """
    hour, mint, sec, msec = _clock_time(allnav, talker)

    inav = [talker in s for s in allnav]  # find lines of file with this talker
    subnav = allnav[inav]  # select only those lines
    N = len(subnav)
    timest = np.empty(N,dtype=datetime)  # array for timestamps, as datetime objects
    
    for i in range(N):
        pre = subnav[i].split(talker)[0]
        date = re.findall('NAV (\d{4})/(\d{2})/(\d{2})',pre)[0]
        year = int(date[0])  # year
        mon = int(date[1])  # month
        day = int(date[2])  # day
        timest[i] = datetime(year,mon,day,hour[i],mint[i],sec[i],msec[i],tzinfo=timezone.utc)
    return timest

def _navdate_NBP(allnav, talker):
    """Extract datetime info from Palmer nav files (NBP*.d*).
    """
    hour, mint, sec, msec = _clock_time(allnav, talker)

    inav = [talker in s for s in allnav]  # find lines of file with this talker
    subnav = allnav[inav]  # select only those lines
    N = len(subnav)
    timest = np.empty(N,dtype=datetime)  # array for timestamps, as datetime objects
    
    for i in range(N):
        pre = subnav[i].split(talker)[0]
        date = re.findall('(\d{2})\+(\d{2,3}):.*',pre)[0]
        year = '20' + date[0]  # year (NBP didn't exist before 2000 so this is ok)
        doy = date[1]  # doy
        timest[i] = datetime.strptime('%s-%s-%02d:%02d:%02d:%06d' % \
                    (year,doy,hour[i],mint[i],sec[i],msec[i]), '%Y-%j-%H:%M:%S:%f')
        timest[i] = timest[i].replace(tzinfo=timezone.utc)
    return timest

def _navdate_Thompson(allnav, talker):
    """Extract datetime info from Thompson nav files (POSMV*.Raw).
    """
    hour, mint, sec, msec = _clock_time(allnav, talker)

    inav = [talker in s for s in allnav]  # find lines of file with this talker
    subnav = allnav[inav]  # select only those lines
    N = len(subnav)
    timest = np.empty(N,dtype=datetime)  # array for timestamps, as datetime objects

    for i in range(N):
        pre = subnav[i].split(talker)[0]
        date = re.findall('(\d{2})/(\d{2})/(\d{4}),*',pre)[0]
        year = int(date[2])
        mon = int(date[0])
        day = int(date[1])
        timest[i] = datetime(year,mon,day,hour[i],mint[i],sec[i],tzinfo=timezone.utc)
    return timest

def _navdate_Revelle(allnav, talker):
    """Extract datetime info from Revelle nav files (mru_seapatah330_rr_navbho-*.txt).
    """
    hour, mint, sec, msec = _clock_time(allnav, talker)

    inav = [talker in s for s in allnav]  # find lines of file with this talker
    subnav = allnav[inav]  # select only those lines
    N = len(subnav)
    timest = np.empty(N,dtype=datetime)  # array for timestamps, as datetime objects
    inds = np.where(inav)[0]  # indices in allnav of talker lines

    for i in range(N):
        # timestamp is on a previous line for Revelle - expect one before 
        # for GPGGA but that is not guaranteed
        if i != 0: j = inds[i-1]  # index of previous talker line
        if i == 0: j = -1
        for k in range(inds[i]-1, j, -1):  # step backwards toward the last talker line 
            before = allnav[k]
            if re.match('(\d{4})-(\d{2})-(\d{2})T*',before):  # date is at the start of this line
                date = re.findall('(\d{4})-(\d{2})-(\d{2})T*',before)[0]
                year = int(date[0])
                mon = int(date[1])
                day = int(date[2])
                timest[i] = datetime(year,mon,day,hour[i],mint[i],sec[i],msec[i],tzinfo=timezone.utc)
                break  # skip the rest of the stepping backwards once date is found
    return timest

def _navdate_Ride(allnav, talker):
    """Extract datetime info from Ride nav files (seapath-navbho_*.raw).
    """
    inav = [talker in s for s in allnav]  # find lines of file with this talker
    subnav = allnav[inav]  # select only those lines
    N = len(subnav)
    timest = np.empty(N,dtype=datetime)  # array for timestamps, as datetime objects

    for i in range(N):
        if talker == 'INGGA': # on Ride, uses posix timestamps
            date = re.findall('(\d+(\.\d*)?) \$%s' % talker,subnav[i])[0]
            timest[i] = datetime.fromtimestamp(float(date[0]),tzinfo=timezone.utc)
        elif talker == 'GPGGA':  # includes time only with date, unlike other GPGGAs
            date = re.findall('(\d{4})\-(\d{2})\-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d.*?)Z',subnav[i])[0]
            year = int(date[0]); mon = int(date[1]); day = int(date[2])
            hour = int(date[3]); mint = int(date[4]); sec = int(date[5]); 
            msec = int(float('.'+date[6])*1e6)
            timest[i] = datetime(year,mon,day,hour,mint,sec,msec,tzinfo=timezone.utc)
    return timest

def _navcoords(allnav, talker):
    """Extract longitude and latitude from standard talker strings.
    """
    inav = [talker in s for s in allnav]  # find lines of file with this talker
    subnav = allnav[inav]  # select only those lines
    N = len(subnav)  # and count the linds
    lon = np.zeros(N); lat = np.zeros(N)  # arrays to hold coordinates
    for i in range(N):  # loop lines, splitting at talker string
        post = subnav[i].split(talker)[-1].lstrip().split(',')
        if post[0] == '': post = post[1:]  # correct for spacing in some files

        lat[i] = int(post[1][:2]) + float(post[1][2:])/60  # convert to decimal degrees
        if post[2] == 'S': lat[i] = -lat[i]         # handle coordinate sign
        lon[i] = int(post[3][:3]) + float(post[3][3:])/60
        if post[4] == 'W': lon[i] = -lon[i]

    return lon, lat

def _clean_180cross(gps_nav):
    """Fix instances where a trackline crosses +/- 180* longitude.
    """

    # which side of the line has more points?
    lpos = gps_nav['lon'].values > 0
    lneg = gps_nav['lon'].values < 0

    newlon = gps_nav['lon'].values

    if sum(lpos) > sum(lneg):
        newlon[lneg] = 180 + newlon[lneg]%180
    else:
        newlon[lpos] = -180 + newlon[lpos]%-180

    # check if there are still some jumpy points (from between +/- reversals)
    ilocs = np.where(abs(np.diff(newlon)) > 1)[0]  # should not jump a degree
    if len(ilocs)%2 != 0:  # these are not paired, so one must be an end?
        pass
    else:  # pairs of indices for jump out and back
        for i in ilocs[::2]:  # assume nice pairwise
            # ad hoc duplication, not great
            newlon[i+1] = (newlon[i] + newlon[i+2])/2

    gps_nav = gps_nav.replace({'lon':newlon})

    return gps_nav

########################################################################
# BGM3 i/o (RGS and serial)
########################################################################

def read_bgm_rgs(fp, ship):
    """Read BGM gravity from RGS files.

    RGS seems to be a standard format, but so far I've only seen
    examples from Atlantis and NBP so there may variations elsewhere.

    This function returns a pandas.DataFrame with timestamps, raw gravity,
    and geographic coordinates.

    :param fp: RGS filepaths, string or list of strings
    :param ship: ship name, string
    """
    supported_ships = ['Atlantis','NBP']
    if ship not in supported_ships:
        print('R/V %s not supported for RGS read yet' % ship)
        return -999

    if type(fp) == str:
        fp = [fp,]  # make a list if only one path is given

    dats = []
    for path in fp:
        dat = pd.read_csv(path, delimiter=' ', names=['date','time','grav','lat','lon'], \
            usecols=(1,2,3,11,12), parse_dates=[[0,1]])
        ndt = [e.tz_localize(timezone.utc) for e in dat['date_time']]
        dat['date_time'] = ndt
        dats.append(dat)
    
    return pd.concat(dats,ignore_index=True)


def read_bgm_raw(fp, ship, scale=None, ship_function=None):
    """Read BGM gravity from raw (serial) files (not RGS).

    This function uses scale factors determined for specific BGM meters
    to convert counts from the raw files to raw gravity. Known scale
    factors are listed in database.toml.

    :param fp: BGM raw filepath(s), string or list of strings
    :param ship: ship name, string
    :param scale: optional, give BGM counts scaling factor outside of database.toml
    :param ship_function: user-supplied function for reading/parsing BGM raw files.
        The function should return a pandas.DataFrame with timestamps and counts.
        Look at _bgmserial_Atlantis() and similar functions for examples.
    """
    moddir = os.path.dirname(__file__)
    import tomli as tm
    with open(os.path.join(moddir,'database.toml'),'rb') as f:
        info = tm.load(f)
    sc_fac = info['BGM-scale']  # get instrument scaling factors from database.toml

    if scale is not None:
        pass  # use provided scale factor if it's there
    else:
        if ship not in sc_fac.keys():
            print('BGM scale factor not known for R/V %s' % ship)
            return -999
        else:
            scale = sc_fac[ship]

    if type(fp) == str:
        fp = [fp,]  # make a list if only one path is given
    dats = []
    for path in fp:
        if ship_function != None:
            dat = ship_function(path)
        else:
            if ship == 'Atlantis':
                dat = _bgmserial_Atlantis(path)
            elif ship == 'Thompson':
                dat = _bgmserial_Thompson(path)
            elif ship == 'Revelle':
                dat = _bgmserial_Revelle(path)
            else:   # shouldn't end up here, but just in case:
                print('BGM serial read not yet supported for R/V %s' % ship)
                return -999
        dat['rgrav'] = scale*dat['counts']
        dats.append(dat)
    return pd.concat(dats,ignore_index=True)

def _bgmserial_Atlantis(path):
    """Read a BGM raw (serial) file from Atlantis.
    """
    count = lambda x: (int(x.split(':')[-1]))  # function to parse counts column
    dat = pd.read_csv(path, delimiter=' ',names=['date','time','counts'],usecols=(1,2,4),\
            parse_dates=[[0,1]],converters={'counts':count})
    ndt = [e.tz_localize(timezone.utc) for e in dat['date_time']]  # timestamps cannot be naive
    dat['date_time'] = ndt
    return dat

def _bgmserial_Thompson(path):
    """Read a BGM raw (serial) file from Thompson.
    """
    count = lambda x: (int(x.split(' ')[0].split(':')[-1]))
    dat = pd.read_csv(path, delimiter=',',names=['date','time','counts'],\
            parse_dates=[[0,1]],converters={'counts':count})
    ndt = [e.tz_localize(timezone.utc) for e in dat['date_time']]
    dat['date_time'] = ndt
    return dat

def _bgmserial_Revelle(path):
    """Read a BGM raw (serial) file from Revelle.
    """
    count = lambda x: (int(x.split(':')[-1]))
    dat = pd.read_csv(path, delimiter=' ',names=['date_time','counts'],usecols=(0,1),\
            parse_dates=[0],converters={'counts':count})
    ndt = [e.tz_localize(timezone.utc) for e in dat['date_time']]
    dat['date_time'] = ndt
    return dat

def _despike_bgm_serial(dat, thresh=8000):
    """Clean out counts spikes in bgm data based on a threshold delta(counts).

    This sometimes works and sometimes doesn't; use at your own risk.
    """
    # find places where counts jump by more than the threshold set
    diff = np.diff(dat.counts)
    meh = np.where(abs(diff) > thresh)[0]
    if len(meh) > 2:
        # for spikes these jumps should be in pairs of +/-
        # we really hope this is the case
        if len(meh)%2 != 0 or np.any(np.diff(meh)[::2] != 1):
            print('something is weird with bgm despike')
            return dat

        # assuming spikes *are* in pairs:
        bad_inds = meh[1::2]  # second of each pair of indices, hopefully
        dat.drop(bad_inds, axis=0, inplace=True)
        dat.reset_index(inplace=True)

    return dat

########################################################################
# DGS i/o ('laptop' and raw)
########################################################################

def read_dat_dgs(fp, ship, ship_function=None):
    """Read DGS 'laptop' file(s), usually written as .dat files.

    :param fp: filepath(s), string or list of strings
    :param ship: ship name, string
    :param ship_function: optional, user-defined function for reading a file.
        The function should return a pandas.DataFrame. See _dgs_laptop_general()
        for an example.
    """
    if type(fp) == str:
        fp = [fp,]  # listify

    dats = []
    for path in fp:
        if ship_function != None:
            dat = ship_function(path)
        else:
            if ship in ['Atlantis','Revelle','NBP','Ride','DGStest']:
                dat = _dgs_laptop_general(path)
            elif ship == 'Thompson':
                dat = _dgs_laptop_Thompson(path)
            else:
                print('R/V %s not supported for dgs laptop file read' % ship)
                return -999

        dats.append(dat)  # append the DataFrame for this filepath

    return pd.concat(dats,ignore_index=True)

def _dgs_laptop_general(path):
    """Read single laptop file for Atlantis, Revelle, NBP, and Ride.
    """
    dat = pd.read_csv(path, delimiter=',',names=['rgrav','long_a','crss_a','status','ve','vcc',\
            'al','ax','lat','lon','year','month','day','hour','minute','second'],\
            usecols=(1,2,3,6,10,11,12,13,14,15,19,20,21,22,23,24))
    dat['date_time'] = pd.to_datetime(dat[['year','month','day','hour','minute','second']],utc=True)
    return dat

def _dgs_laptop_Thompson(path):
    """Read single laptop file for Thompson, which does things its own way.
    """
    dat = pd.read_csv(path, delimiter=',',names=['date','time','rgrav','ve','vcc',\
            'al','ax','lat','lon'],usecols=(0,1,3,12,13,14,15,16,17),\
            parse_dates=[[0,1]])
    ndt = [e.tz_localize(timezone.utc) for e in dat['date_time']]
    dat['date_time'] = ndt
    return dat

def read_raw_dgs(fp, ship, scale_ccp=True):
    """Read raw (serial) output files from DGS AT1M.

    These will be in AD units mostly.
    File formatting is assumed to follow what the DGS documentation
    says, though some things may vary by vessel so if this doesn't
    work that's probably why.

    :param fp: filepath(s), string or list of strings
    :param ship: ship name, string
    """

    if type(fp) == str:
        fp = [fp,]  # listify
    dats = []
    for ip, path in enumerate(fp):
        if ship == 'Thompson':  # always with the special file formats
            dat = _dgs_raw_Thompson(path)
        else:  # there might be exceptions besides Thompson but I don't know about them yet
            dat = _dgs_raw_general(path)

        if scale_ccp:
            # rescale the cross-coupling factors
            dat['ve'] = dat['ve'].mul(0.00001)
            dat['ax'] = dat['ax'].mul(0.00001)
            dat['al'] = dat['al'].mul(0.00001)
            dat['vcc'] = dat['vcc'].mul(0.00001)  #-0.000029)
        dats.append(dat)

    return pd.concat(dats,ignore_index=True)

def _dgs_raw_general(path):
    """Read a DGS raw (serial) file assuming fields are as DGS says they are.
    """
    dat = pd.read_csv(path, delimiter=',',names=['string','Gravity','Long','Cross','Beam','Temp',\
                    'Pressure','ElecTemp','vcc','ve','al','ax','status','checksum','latitude',\
                    'longitude','speed','course','timestamp'],\
                    usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))

    conv_times = True  # assume column 18 is actually timestamps
    if str(dat.iloc[0].timestamp).startswith('1'):  # 1 Hz, 1 second
        conv_times = False # clock not synced, so don't try to convert to stamp

    if conv_times:  # probably UTC stamps
        new_dates = pd.to_datetime(dat['timestamp'],utc=True,format='%Y%m%d%H%M%S')
        dat['date_time'] = new_dates

    # special case of not synced for timestamp, but stamp might be in string elsewhere
    if not conv_times and not dat.iloc[0]['string'].startswith('$'): 
        # split string for possible timestamp
        try:
            times = [e.split(' ')[0] for e in dat['string'].values]            
            dat['date_time'] = pd.to_datetime(times,format='ISO8601')
        except:
            print('raw (serial) timestamps not found/converted')
            pass # if it doesn't work, oh well

    return dat

def _dgs_raw_Thompson(path):
    """Read a DGS raw (serial) file with Thompson conventions.

    Columns are slightly different from Atlantis and Revelle examples;
    in particular, before the $AT1M string there are date and time
    stamps that are also comma-separated so the csv read used for those
    other ships does not work properly.
    """
    dat = pd.read_csv(path, delimiter=',',names=['date','time','string','Gravity','Long','Cross',\
                    'Beam','Temp','Pressure','ElecTemp','vcc','ve','al','ax','status','checksum',\
                    'latitude','longitude','speed','course'],\
                    usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19))
    dat['stamps'] = [':'.join([e[1]['date'],e[1]['time']]) for e in dat.iterrows()]
    dat['date_time'] = pd.to_datetime(dat['stamps'],utc=True,format='%m/%d/%Y:%H:%M:%S.%f')
    dat.drop('stamps',axis=1,inplace=True)

    return dat

########################################################################
# reading other things (MRU etc)
########################################################################

def read_other_stuff(yaml_file,data_file,tag):
    """Read a particular feed (eg, $PASHR) from a data file + yaml file.

    This function parses strings for the desired feed and returns info as a 
    pandas.DatafFame with columns named from the corresponding yaml file.
    If there is a column in the feed strings prior to the tag itself, that 
    will be included as a 'mystery' dataframe column.

    A use case is if you want to check coherence between gravity data and
    one or more MRUs

    :param yaml_file: path to YAML file with info for this feed
    :param data_file: path to data file to be read
    :param tag: the name of the feed, with or without the $ prepended
    """

    if tag.startswith('$'): tag = tag[1:]
    dtag = '$' + tag

    # read yaml, check that tag specs are in this file
    with open(yaml_file,'r') as f:
        ym = yaml.safe_load(f)
    ym = ym[list(ym.keys())[0]]  # skip out of the top level key
    assert tag in ym['format'].keys(), '%s feed specs not present in yaml file %s' % (tag, yaml_file)

    # check to make sure tag appears in the data file
    with open(data_file,'rb') as file:
        s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        result = s.find(bytes(tag,'ascii'))
    assert result != -1, '%s not in data file %s' % (tag, data_file)

    # checks done, read in all the lines for this feed
    lines = []
    s.seek(0)
    ln = s.readline().decode('ascii').rstrip()  # get line, bytes->string, strip \r\n from end
    while s.tell() < s.size():  # when s.tell() == s.size(), we got to the end of the file
        if tag in ln:
            lines.append(ln)
        ln = s.readline().decode('ascii').rstrip()  # next!
    
    # split the lines at the tag, and then after the tag (fields named in yaml are post-tag)
    tagsplit = [e.split(dtag) for e in lines]  # split into pre-tag (might be '') and post-tag
    #tagjoin = [','.join(tagsplit[i][0].strip(),tagsplit[i][1].strip(',')) for i in range(len(tagsplit))]
    named_fields = [tagsplit[i][1].strip(',').split(',') for i in range(len(tagsplit))]

    # make a dataframe, using yaml and including any timestamp-like info that might be before the tag
    # Note that the format of the lines we're working with is usually some variation on:
    # [date/timestamp, maybe], $TAG, comma, separated, fields, listed, in, yaml
    colnames = re.sub('{|}','',ym['format'][tag]).split(',')[1:]
    df = pd.DataFrame(named_fields,columns=colnames)
    df['mystery'] = [tagsplit[i][0] for i in range(len(tagsplit))]  # any pre-tag stuff

    # finally, gather some other column info from the yaml file to output
    col_info = {}
    for c in colnames:
        try:
            col_info[c] = ym['fields'][c.split(':')[0]]
        except KeyError:
            pass

    return df.apply(pd.to_numeric,errors='ignore'), col_info
