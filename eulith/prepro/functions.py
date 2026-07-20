import os
import numpy as np
import pandas as pd
from obspy import Stream, Trace
from obspy.signal.rotate import rotate2zne, rotate_ne_rt
from obspy.clients.fdsn import Client

from ..core.utils import *


### download tools

def multiple_location_check(traces:list[Trace]):
    """
    Check if multiple locations are present in a list of traces.

    :type traces: list of obspy.Trace objects
    :param traces: list of obspy.Trace objects
    :return: list of unique location codes present in the traces    
    """
    if not traces: return []
    return list(set(tr.stats.location for tr in traces))
    
def group_channels(st:Stream):
    """
    Group traces into Z, N, E components based on their channel codes. Can take in N1, E2, etc.
    but returns in ZNE.
    
    :type st: obspy.Stream
    :param st: stream containing traces to be grouped
    :return: Dictionary with keys 'Z', 'N', 'E' and values as lists of obspy.Trace objects.
    """
    if not st: return {'Z': [], 'N': [], 'E': []}
    out = {'Z': st.select(component="Z"),
           'N': st.select(component="[N1]"), # Matches N or 1
           'E': st.select(component="[E2]")  # Matches E or 2
          }
    return out

def count_traces(grouped_dict:dict):
    """
    Count number of traces in each component group. 
    
    :type grouped_dict: dict
    :param grouped_dict: dictionary with keys 'Z', 'N', 'E' and values as lists of obspy.Trace objects
    :return: Dictionary with counts for 'Z', 'N', and 'E'.
    """
    if not grouped_dict: return {'Z': 0, 'N': 0, 'E': 0}
    return {comp: len(st) for comp, st in grouped_dict.items()}

def sum_trace_lengths(grouped_dict):
    """
    Returns the total temporal length of traces in each component group.
    
    :type grouped_dict: dict
    :param grouped_dict: dictionary with keys 'Z', 'N', 'E' and values as lists of obspy.Trace objects
    :return: Dictionary with total lengths for 'Z', 'N', and 'E'.
    """
    if not grouped_dict: return {'Z': 0, 'N': 0, 'E': 0}
    out = {comp: sum(tr.stats.endtime - tr.stats.starttime for tr in st) 
           for comp, st in grouped_dict.items()}
    return out

def _band_sort(band_list:list):
    """
    If band order needs fixing. 

    :type band_list: list
    :param band_list: list of band codes (e.g., ['H', 'S', 'E', 'L', 'B'])
    :return: sorted list of band codes in the order ['L', 'B', 'H', 'E', 'S']
    """
    key = {'L':1, 'B':2, 'H':3}
    return sorted(list(set(band_list)), key=lambda x: (key.get(x[0], 4), x))

### download functions

def event_sort(events):
    """
    Sort events into groups avoiding overlapping time windows. 
    Returns a list of dataframes, each containing events which don't overlap in time.

    :type events: pandas.DataFrame
    :param events: dataframe containing events with 'window_start' and 'window_end' columns
    :return: list of pandas.DataFrame, each containing non-overlapping events
    """
    gp = []
    window_idx = events.columns.get_indexer(['window_start', 'window_end'])
    for i, e in enumerate(events.itertuples()):
        if not gp: gp = [[[i, e]]]; continue
        else: 
            j = i - len(gp)
            condition = overlap(events.iloc[j,window_idx[0]], events.iloc[j,window_idx[1]], e.window_start, e.window_end)
            if condition:
                gp.append([[i, e]]); continue
            else:
                idx = [m for m, g in enumerate(gp) if g[-1][0] == j][0]
                gp[idx].append([i, e]); continue
    return [pd.DataFrame([ev[1] for ev in g]) for g in gp]

def process_multiple_locations(evst:pd.DataFrame, threshold=0.9):
    """
    
    
    """
    multiple_processing = pd.DataFrame({'locations': evst['locations'], \
                                        'traces': [Stream([tr for tr in evst['traces'] if tr.stats.location == loc]) for loc in evst['locations']]})
    multiple_processing['grouped_traces'] = multiple_processing['traces'].apply(group_channels)
    multiple_processing['trace_count'] = multiple_processing['grouped_traces'].apply(count_traces)
    multiple_processing['error'] = [f"missing_components_{''.join(sorted([k.lower() for k, v in d.items() if v == 0]))}"
                                if any(v == 0 for v in d.values()) else None \
                                for d in multiple_processing['trace_count']]

    mask = (multiple_processing['error'].isna())

    multiple_processing.loc[mask, 'window_threshold'] = threshold * (evst['req_end'] - evst['req_start'])
    multiple_processing.loc[mask, 'summed_trace_lengths'] = multiple_processing.loc[mask, 'grouped_traces'].apply(sum_trace_lengths)
    multiple_processing.loc[mask, 'error'] = [f"short_traces_{''.join(sorted([c.lower() for c, tl in trace_lengths.items() if tl < threshold]))}" \
                                if any(tl < threshold for tl in trace_lengths.values()) else None \
                                for trace_lengths, threshold in zip(multiple_processing.loc[mask, 'summed_trace_lengths'], multiple_processing.loc[mask, 'window_threshold'])]

    mask = (multiple_processing['error'].isna())
    if not multiple_processing.loc[mask].empty: 
        selection = multiple_processing.loc[mask].copy()
        return pd.Series([selection.loc[selection['locations'].idxmin(), 'grouped_traces'], None])
    else:
        return pd.Series([None, 'data_exception'])

### metadata fixes

def fix_delta(delta:float):
    """
    Delta is the time interval between samples in a trace. Rounds deltas containing '99...'s,
    based on the position of '99' in its string representation. 

    :type delta: float
    :param delta: time interval between samples in a trace
    :return: rounded delta
    """
    delta_str = f'{delta:e}'
    multiplier = int(delta_str[-2:])
    position = len(delta_str.replace('.', '').split('99')[0])
    return round(delta, position + multiplier)

def check_polarity_ori(station):
    """
    Checks the polarity of a station based on the orientation of its three components. 

    :type station: obspy.Station
    :param station: station inventory object
    :return: 0 for right-handed (negative polarity) and 1 for left-handed (positive polarity). None if
    the polarity cannot be determined.
    """
    try: 
        orientation = {convert_n1e2z3(chan.code[-1]): (1., chan.azimuth*np.pi/180, chan.dip*np.pi/180 + np.pi/2) 
                       for chan in station}
    except Exception as e:
        print(f"Error in check_polarity_ori: {e}")
        return None
    cp = cross_product(orientation['1'], orientation['2'], outcoord='car')
    if cp.sum() == 0: 
        print('Warning: cross product of horizontal components is zero. Polarity cannot be determined.')
        return None # then the horizontal components are parallel and polarity cannot be determined

    else: 
        z_vec = sph_to_car(orientation['3'])
        if cp[-1] * z_vec[-1] > 0: return 0 # right handed (negative polarity)
        if cp[-1] * z_vec[-1] < 0: return 1 # left handed (positive polarity)

def check_polarity_sen(row:pd.Series, st=Stream, client:Client=None):
    """
    Checks the polarity of a station based on the instrument sensitivity of its three components.
    Requires evst dataframe. 

    :type row: pandas.Series
    :param row: row of the station inventory dataframe
    :type st: obspy.Stream
    :param st: stream containing the three components of the station
    :type client: obspy.clients.fdsn.Client
    :param client: ObsPy Client object
    
    """
    if not client: 
        node = row.node
        try: client = Client(fdsn_registry_dict().get(node.lower()))
        except Exception as e: 
            print(f"Error in initialising FDSN client for {node}: {e}")
            return None
    
    try:
        resp_inv = client.get_stations(network=row.network, station=row.station, location=st[0].stats.location, channel=f'{row.channel}H?', level='response')[0][0]
        if len(resp_inv) > 3: return None
        sens = [s.response.instrument_sensitivity.value for s in resp_inv]
        if len(set(sens)) > 1: return None # exit if sensitivities are not the same
        if np.sign(sens[0]) == 1: return 0 # right handed
        if np.sign(sens[0]) == -1: return 1 # left handed
    except Exception as e:
        print(f"Error in check_polarity_sen: {e}")
        return None

def _zne(st:Stream):
    """
    Rotate three-component stream to ZNE. The traces in the stream must have 
    the same start/end time and npts. 

    :type st: obspy.Stream
    :param st: stream to be rotated
    :return: three-component stream in ZNE as tuple of obspy.Trace objects
    """
    da1, da2, da3 = [tr.data for tr in st]
    az1, az2, az3 = [tr.stats.sac.cmpaz for tr in st]
    di1, di2, di3 = [tr.stats.sac.cmpinc-90. for tr in st]
    return rotate2zne(da1, az1, di1, da2, az2, di2, da3, az3, di3)

def zrt(st:Stream, ne=False):
    """
    Rotate three-component stream to ZRT. The traces in the stream must have 
    the same start/end time and npts. Wrapper calls _zne(st).

    :type st: obspy.Stream
    :param st: stream to be rotated
    :type ne: bool
    :param ne: if True, also return the N and E components
    :return: three-component stream in ZRT(NE) as tuple of obspy.Trace objects
    """
    z, n, e = _zne(st)
    r, t = rotate_ne_rt(n, e, st[0].stats.sac.baz)
    if ne: return z, r, t, n, e
    else: return z, r, t

def rotate_rt(st:Stream, plot=False):
    """
    Rotate three-component stream to ZRT. Trims the stream to the overlapping time window of all traces.
    Updates seed and sac metadata.

    :type st: obspy.Stream
    :param st: stream to be rotated
    :type plot: bool
    :param plot: if True, plot the rotated stream
    :return: five-component (ZRTNE) stream in RT as tuple of obspy.Trace objects
    """

    # perform the rotation
    st_trim = trim_trace(st)
    z, r, t, n, e = zrt(st_trim, ne=True)

    # update the metadata
    stats_template = st_trim.select(component='Z')[0].stats.copy()
    stats_template.sac.pop('cmpaz')
    stats_template.sac.pop('cmpinc')
    band = stats_template.channel[:2]
    z_stats = stats_template.copy(); z_stats.channel = f'{band}Z'; z_stats.sac.kcmpnm = f'{band}Z'
    r_stats = stats_template.copy(); r_stats.channel = f'{band}R'; r_stats.sac.kcmpnm = f'{band}R'
    t_stats = stats_template.copy(); t_stats.channel = f'{band}T'; t_stats.sac.kcmpnm = f'{band}T'
    n_stats = stats_template.copy(); n_stats.channel = f'{band}N'; n_stats.sac.kcmpnm = f'{band}N'
    e_stats = stats_template.copy(); e_stats.channel = f'{band}E'; e_stats.sac.kcmpnm = f'{band}E'

    # build the stream
    zt = Trace(data=z, header=z_stats)
    rt = Trace(data=r, header=r_stats)
    tt = Trace(data=t, header=t_stats)
    nt = Trace(data=n, header=n_stats)
    et = Trace(data=e, header=e_stats)
    st_rot = Stream(traces=[zt, rt, tt, nt, et])

    if plot: st_rot.plot(); print()
    return st_rot

