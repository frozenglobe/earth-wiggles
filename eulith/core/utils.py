# Generic tools
import ast
import numpy as np
import pandas as pd
from obspy import UTCDateTime


def read_stnlist(path:str):
    """
    Reads a stnlist file and returns a pandas DataFrame with the relevant information.

    :type path: str
    :param path: path to the stnlist file
    :return: pandas DataFrame with the stnlist information
    """
    stnlist = pd.read_csv(path, \
                           dtype={'location':str, 'bandgain':str, \
                                  'latitude':float, 'longitude':float, 'elevation':float, \
                                  'starttime':str, 'endtime':str})
    # reject row if any of these contain NaN
    keys_reject = ['bandgain', 'latitude', 'longitude', 'starttime', 'endtime']
    condition = stnlist[keys_reject].isna().sum(axis=1) > 0
    sl_processed = stnlist.drop(stnlist[condition].index)
    sl_processed['channel'] = sl_processed['channel'].apply(lambda x: ast.literal_eval(x))
    return sl_processed

def read_cmt(path:str):
    """
    Reads a CMT file and returns a pandas DataFrame with the relevant information.
    Converts scalar magnitude to moment magnitude. 

    :type path: str
    :param path: path to the CMT file
    :return: pandas DataFrame with the CMT information
    """
    col_specs = [(0, 123), # cmt line
                 (0, 4),   # year
                 (4, 8),   # jday
                 (8, 11),  # hour
                 (11, 14), # min
                 (14, 19), # sec (including the decimal)
                 (19, 25), # latitude
                 (25, 32), # longitude
                 (32, 38), # depth, in km
                 (38, 49)  # scalar_moment
                ]
    col_names = ['cmt', 'year', 'jday', 'hour', 'minute', 'second', \
                 'evlat', 'evlon', 'evdep', 'scalar_moment']
    cmt_df = pd.read_fwf(path, colspecs=col_specs, names=col_names)

    # obtain milliseconds
    cmt_df['msecond'] = cmt_df['second'].astype(str).apply(lambda x: int(x.split('.')[1]) * 100)
    # convert to datetime
    cmt_df['evtime'] = pd.to_datetime(
        cmt_df['year'].astype(str) + '-' + cmt_df['jday'].astype(str).str.zfill(3) + ' ' +
        cmt_df['hour'].astype(str).str.zfill(2) + ':' + cmt_df['minute'].astype(str).str.zfill(2) + ':' +
        cmt_df['second'].astype(str), 
        format='%Y-%j %H:%M:%S.%f'
    )
    # calculate moment magnitude
    cmt_df['moment_magnitude'] = (np.log10(cmt_df['scalar_moment']) + 7)/ 1.5 - 10.7
    cmt_df = cmt_df[['cmt', 'evtime', 'year', 'jday', 'hour', 'minute', 'second', 'msecond',
                     'evlat', 'evlon', 'evdep', 'moment_magnitude']]
    return cmt_df

### pandas tools

def time_format(df:pd.DataFrame, keys=['starttime', 'endtime']):
    """ 
    not convinced


    :type df: pd.DataFrame
    """
    import copy
    new_df = copy.deepcopy(df)
    for k in keys: new_df[k] = new_df[k].apply(lambda x: UTCDateTime(x))
    return new_df

def latlon_filter(df:pd.DataFrame, 
                  lat_min:float, lat_max:float,
                  lon_min:float, lon_max:float):
    """
    Filters dataframe for specified lat/lon bounds

    :type df: pd.DataFrame
    :param df: DataFrame containing 'latitude' and 'longitude' columns.
    :type lat_min: float
    :param lat_min: Minimum latitude bound.
    :type lat_max: float
    :param lat_max: Maximum latitude bound.
    :type lon_min: float
    :param lon_min: Minimum longitude bound.
    :type lon_max: float
    :param lon_max: Maximum longitude bound.
    :return: Filtered DataFrame with rows within the specified lat/lon bounds.
    """
    latlon_df = df[df['latitude'].between(lat_min, lat_max) & df['longitude'].between(lon_min, lon_max)]
    return latlon_df.reset_index(drop=True)

def replace_60(dt_str:str):
    """
    handles 60 second issue in datetime strings
    
    :type dt_str: str
    :param dt_str: datetime string in the format 'YYYY-MM-DDTHH:MM:SS.ssssss'
    :return: datetime string with seconds replaced if necessary
    """
    date, time = dt_str.split('T')
    hour, minute, seconds = time.split(':')
    if '.' in dt_str: microsecond = seconds.split('.')[-1]
    second = '00'; minute = int(minute) + 1;
    if '.' in dt_str: seconds = second + '.' + microsecond
    else: seconds = second
    if minute == 60: hour = int(hour) + 1; minute = '00'
    concat_string = date + 'T' + ":".join([str(hour), str(minute), seconds])
    return concat_string

def time_in_range(start, end, int_start, int_end): 
    """
    Checks if an interval's start and end are within a given range. e.g. for datetimes

    :param start: Start of the range
    :param end: End of the range
    :param int_start: Start of the interval
    :param int_end: End of the interval
    :return: True if the interval is within the range, False otherwise
    """
    return (start <= int_start <= end) & (start <= int_end <= end)

def overlap(start1, end1, start2, end2):
    """
    Checks if two intervals overlap.
    :param start1: Start of first interval
    :param end1: End of first interval
    :param start2: Start of second interval
    :param end2: End of second interval
    :return: True if intervals overlap, False otherwise
    """
    return start1 <= end2 and start2 <= end1

def convert_n1e2z3(chan_code:str, reverse=False):
    """
    Converts channel codes between 'N', 'E', 'Z' and '1', '2', '3'.

    :type chan_code: str
    :param chan_code: Channel code to convert ('N', 'E', 'Z' or '1', '2', '3').
    :type reverse: bool
    :param reverse: If True, converts from '1', '2', '3' to 'N', 'E', 'Z'. 
    If False (default), converts from 'N', 'E', 'Z' to '1', '2', '3'.
    :return: Converted channel code.
    """
    chan_dict = {'N': '1', 'E': '2', 'Z': '3'}

    if not reverse:
        if chan_code in ['N', 'E', 'Z']: return chan_dict.get(chan_code)
        else: return chan_code
    if reverse:
        if chan_code in ['1', '2', '3']: return {v: k for k, v in chan_dict.items()}.get(chan_code)
        else: return chan_code

### vector operations

def sph_to_car(sphvec:tuple, precision:int=12):
    """
    Converts a spherical vector (r, phi, theta) to Cartesian coordinates (x, y, z)
    :type sphvec: tuple
    :param sphvec: spherical vector (r, phi, theta) [radius, azimuth, polar]
    :param precision: number of decimal places to round the result
    :return: Cartesian coordinates (x, y, z)

    """
    x = sphvec[0]*np.sin(sphvec[2])*np.cos(sphvec[1])
    y = sphvec[0]*np.sin(sphvec[2])*np.sin(sphvec[1])
    z = sphvec[0]*np.cos(sphvec[2])

    out = (x, y, z)

    if precision is not None: out = (round(cmp, precision) for cmp in out)

    return out

def car_to_sph(carvec:tuple, precision:int):
    """
    Converts a Cartesian vector (x, y, z) to spherical coordinates (r, phi, theta)
    :type carvec: tuple
    :param carvec: Cartesian vector (x, y, z)
    :param precision: number of decimal places to round the result
    :return: spherical coordinates (r, phi, theta) [radius, azimuth, polar]
    """
    r = np.linalg.norm(carvec)
    phi = np.arctan2(carvec[1], carvec[0])
    theta = np.arccos(carvec[2] / r)
    if theta > np.pi: theta -= np.pi
    out = (r, phi, theta)

    if precision is not None: out = (round(cmp, precision) for cmp in out)
    
    return out

def cross_product(v1:tuple, v2:tuple, 
                  incoord='sph', outcoord='sph',
                  precision:int=12):
    """
    Compute the cross product between two vectors v1 and v2. Can choose to input and output in spherical or Cartesian coordinates.

    :type v1: tuple
    :param v1: first vector (r, phi, theta) or (x, y, z)
    :type v2: tuple
    :param v2: second vector (r, phi, theta) or (x, y, z)
    :type incoord: str
    :param incoord: input coordinate system ('sph' or 'car')
    :type outcoord: str
    :param outcoord: output coordinate system ('sph' or 'car')
    :type precision: int
    :param precision: number of decimal places to round the result
    :return: cross product vector in specified output coordinate system
    """
    if incoord == 'sph':
        v1 = sph_to_car(v1, precision=precision)
        v2 = sph_to_car(v2, precision=precision)
    
    cross_car = np.cross(v1, v2)

    if outcoord == 'car': out = cross_car
    elif outcoord == 'sph': 
        r = np.linalg.norm(cross_car)
        phi = np.arctan2(cross_car[1], cross_car[0])
        theta = np.arccos(cross_car[2] / r)
        if theta > np.pi: theta -= np.pi
        out = (r, phi, theta)

    if precision is not None: out = (round(cmp, precision) for cmp in out)
    
    return out

def unit_vec_latlon(lat:float, lon:float):
    """
    Returns the unit position vector in Cartesian coordinates for a given latitude and longitude.

    :type lat: float
    :param lat: latitude in degrees
    :type lon: float
    :param lon: longitude in degrees
    :return: unit position vector in Cartesian coordinates (x, y, z)
    """
    return (1, lon*np.pi/180, np.pi/2 - (lat*np.pi/180))

def gcmax(v1:tuple, v2:tuple, incoord:str='sph', outcoord:str='sph'):
    """
    Computes the position vector of the point on the great circle connecting position vectors v1 and v2
    that is maximal in latitude. 

    :type v1: tuple
    :param v1: first vector (r, phi, theta) or (x, y, z)
    :type v2: tuple
    :param v2: second vector (r, phi, theta) or (x, y, z)
    :type incoord: str
    :param incoord: input coordinate system ('sph' or 'car')
    :type outcoord: str
    :param outcoord: output coordinate system ('sph' or 'car')
    :return: position vector of the point on the great circle that is maximal in latitude
    """
    gcp_normal = np.array(cross_product(v1, v2, incoord, 'car'))
    gcp_norm = gcp_normal / np.linalg.norm(gcp_normal)
    car_vec_max_lat = np.array([0, 0, 1]) - gcp_norm[2] * gcp_norm
    if outcoord == 'car': out = car_vec_max_lat
    elif outcoord == 'sph': out = car_to_sph(car_vec_max_lat)

    return out

