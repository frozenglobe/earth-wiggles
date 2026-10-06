import os
from pathlib import Path
import numpy as np
import pandas as pd
from obspy import Stream, Trace, read, UTCDateTime
from obspy.core.inventory import Inventory, Station
from obspy.signal.rotate import rotate2zne, rotate_ne_rt
from obspy.clients.fdsn import Client

from eulith.core.utils import convert_n1e2z3, sph_to_car, cross_product, overlap, time_in_range
from eulith.core.trace import trim_trace


### download tools

def _prepare_chunk(chunk:pd.DataFrame, band_state:int) -> pd.DataFrame:
    chunk['req_band'] = chunk['bands'].str[band_state] + "?"
    chunk['bulk_request'] = list(zip(chunk['network'], 
                                     chunk['station'], 
                                     ['*'] * len(chunk),
                                     chunk['req_band'], 
                                     chunk['req_start'], 
                                     chunk['req_end']))
    chunk.reset_index(inplace=True)
    return chunk

def _generate_next_evst(evst, band_state) -> pd.DataFrame:
    evst_out = evst.loc[~evst['status'].isin([10, 403])]
    evst_out = evst_out[evst_out['bands'].apply(lambda x: band_state + 1 < len(x))]
    return evst_out

def build_bulk_request(chunk) -> list:
    out = list(zip(chunk['network'], 
                   chunk['station'], 
                   ['*'] * len(chunk),
                   chunk['req_chan'], 
                   chunk['req_start'], 
                   chunk['req_end']))
    return out

def sort_traces(entry, resp, buffer=10):
    """
    Sort traces into the correct event
    """
    return [tr for tr 
            in resp 
            if time_in_range(entry.loc['req_start'] - buffer, \
                             entry.loc['req_end'] + buffer, \
                             tr.stats.starttime, tr.stats.endtime) \
               and (tr.stats.station == entry.loc['station'])]

def sort_traces_new(row:pd.Series, resp:Stream, buffer=10):
    resp = resp.select(station=row.station)
    return [tr for tr
            in resp
            if time_in_range(row.req_start - buffer, row.req_end + buffer, \
                             tr.stats.starttime, tr.stats.endtime)]

def multiple_location_check(traces:list[Trace]):
    """
    Check if multiple locations are present in a list of traces.

    :type traces: list of obspy.Trace objects
    :param traces: list of obspy.Trace objects
    :return: list of unique location codes present in the traces    
    """
    if not traces: return []
    return list(set(tr.stats.location for tr in traces))
    
def group_channels(st:Stream) -> dict:
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

def count_traces(grouped_dict:dict) -> dict:
    """
    Count number of traces in each component group. 
    
    :type grouped_dict: dict
    :param grouped_dict: dictionary with keys 'Z', 'N', 'E' and values as lists of obspy.Trace objects
    :return: Dictionary with counts for 'Z', 'N', and 'E'.
    """
    if not grouped_dict: return {'Z': 0, 'N': 0, 'E': 0}
    return {comp: len(st) for comp, st in grouped_dict.items()}

def sum_trace_lengths(grouped_dict) -> dict:
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

def _band_sort(band_list:list) -> list:
    """
    If band order needs fixing. 

    :type band_list: list
    :param band_list: list of band codes (e.g., ['H', 'S', 'E', 'L', 'B'])
    :return: sorted list of band codes in the order ['L', 'B', 'H', 'E', 'S']
    """
    key = {'L':1, 'B':2, 'H':3}
    return sorted(list(set(band_list)), key=lambda x: (key.get(x[0], 4), x))

def _process_checks(evst:pd.DataFrame, threshold:float=0.9) -> pd.DataFrame:
    """
    Performs checks on the evst dataframe to identify events with missing components, short traces, or multiple traces.

    :type evst: pandas.DataFrame
    :param evst: evst dataframe containing traces to check
    :type threshold: float
    :param threshold: Threshold for summed trace lengths relative to event window [0, 1]
    :return: pandas.Series containing error messages for each event, or None if no errors
    """
    # group traces by component and count
    evst['grouped_traces'] = evst['traces'].apply(group_channels)
    evst['trace_count'] = evst['grouped_traces'].apply(count_traces)
    # check for missing components
    evst['error'] = [f"missing_components_{''.join(sorted([k.lower() for k, v in d.items() if v == 0]))}"
                     if any(v == 0 for v in d.values()) else None \
                     for d in evst['trace_count']]
    mask = (evst['error'].isna())
    # check trace length exceeds threshold (window)
    evst.loc[mask, 'window_threshold'] = threshold * (evst['req_end'] - evst['req_start'])
    evst.loc[mask, 'summed_trace_lengths'] = evst.loc[mask, 'grouped_traces'].apply(sum_trace_lengths)
    evst.loc[mask, 'error'] = [f"short_traces_{''.join(sorted([c.lower() for c, tl in trace_lengths.items() if tl < threshold]))}" \
                                if any(tl < threshold for tl in trace_lengths.values()) else None \
                                for trace_lengths, threshold in zip(evst.loc[mask, 'summed_trace_lengths'], evst.loc[mask, 'window_threshold'])]
    # check for multiple traces
    evst.loc[mask, 'error'] = [f"multiple_traces_{''.join(sorted([k.lower() for k, v in d.items() if v > 1]))}"
                                if any(v > 1 for v in d.values()) else None \
                                for d in evst.loc[mask, 'trace_count']]
    out = evst[['grouped_traces', 'error']]
    return out

def _process_multiple_locations(row, threshold:float=0.9) -> pd.Series:
    """
    Processes events with multiple locations, selecting the 'best' location
    If multiple locations have the same number of traces, selects the one with the minimum summed trace length.

    :type row: need to check
    :param row: containing an event with multiple locations
    :type threshold: float
    :param threshold: Threshold for summed trace lengths relative to event window
    :return: 
    """
    multiple_processing = pd.DataFrame({'locations': row['locations'], \
                                        'traces': [Stream([tr for tr in row['traces'] if tr.stats.location == loc]) for loc in row['locations']]})
    multiple_processing['error'] = _process_checks(multiple_processing, threshold)
    mask = (multiple_processing['error'].isna())
    if not multiple_processing.loc[mask].empty: 
        selection = multiple_processing.loc[mask].copy()
        out = pd.Series([selection.loc[selection['locations'].idxmin(), 'grouped_traces'], None])
    else:
        out = pd.Series([None, 'data_exception'])
    return out

def _write_traces(evst:pd.DataFrame, mt:bool=False) -> None:
    """
    Write out traces to SAC files, creating directories as needed. If mt is True, writes out multiple traces
    per component to directory 'data_mt'.

    :type evst: pandas.DataFrame
    :param evst: evst dataframe containing traces to write out
    :type mt: bool
    :param mt: If True, writes out multiple traces per component to directory 'data_mt'.
    """
    evst['data_path'].apply(lambda x: Path(x).mkdir(parents=True, exist_ok=True))
    filestem = lambda row: f"{row.evtime.strftime('%y%m%d_%H%M%S_')}{row.station.lower()}"
    if not mt: 
        write_tasks = [(st[0], os.path.join(row.data_path, f"{filestem(row)}.{st[0].stats.channel.lower()}"))
                       for row in evst.itertuples() 
                       for st in row.grouped_traces.values()]
    elif mt:
        evst['data_path'] = evst['data_path'].apply(lambda x: x.replace('data/', 'data_mt/'))
        evst['data_path'].apply(lambda x: Path(x).mkdir(parents=True, exist_ok=True))
        write_tasks = []
        for row in evst.itertuples():
            for st in row.grouped_traces.values():
                if not st: continue
                st.sort(keys=['starttime'])
                for idx, tr in enumerate(st):
                        file_path = os.path.join(row.data_path, f"{filestem(row)}.{tr.stats.channel.lower()}_{idx+1}")
                        write_tasks.append((tr, file_path))
    [tr.write(path, format='SAC') for tr, path in write_tasks]
    return

def _write_cmt(evst:pd.DataFrame) -> None:
    """
    Write out single CMT lines to event directories
    """
    for row in evst.itertuples():
        base = row.evpath if not "multiple_traces" in str(row.error) else row.evpath.replace('data', 'data_mt')
        full_path = os.path.join(base, f"cmt{row.evtime.strftime('%y%m%d_%H%M%S')}")
        if not os.path.exists(full_path):
            with open(full_path, 'w') as f: f.write(f"{row.cmt}\n")
    return

### download functions

def event_sort(rows_to_sort:list[list],
               chunksize:int,
               ) -> tuple[list[list[int]], list[list]]:
    """
    :type rows_to_sort: list
    :param rows_to_sort: containing Index, window_start, window_end
    :return: list of list of indices
    """
    sorted_rows = sorted(rows_to_sort, key=lambda x: (x[1], x[2]))
    to_group = sorted_rows
    groups = []
    
    while to_group:
        grouped = [to_group[0][0]] # put first index in first group
        prev_end:UTCDateTime = to_group[0][2] # set previous end time to first event's end time
        to_group_again = [] # list to hold events that will be grouped in the next iteration

        for i in range(1, len(to_group)):
            if len(grouped) >= chunksize:
                to_group_again.extend(to_group[i:])
                break
            if prev_end > to_group[i][1]: # if previous end time is greater than current start time, they overlap
                to_group_again.append(to_group[i])
            else:
                grouped.append(to_group[i][0]) # add index to current group
                prev_end = to_group[i][2] # update previous end time

        groups.append(grouped)
        to_group = to_group_again

    return (groups, to_group)

def chunk_generator(evst:pd.DataFrame, chunksize:int=50) -> list[pd.DataFrame]:
    """
    Generate chunks of events for processing.

    :type evst: pandas.DataFrame
    :param evst: dataframe containing events
    :type chunksize: int
    :param chunksize: number of events per chunk
    :rtype: list[pd.DataFrame]
    :return: list of dataframes, each containing a chunk of events
    """
    sorted_evst_list = event_sort(evst) # retains the original indexing in column 'Index'
    request_chunks = [df.iloc[i:i+chunksize] 
                      for df in sorted_evst_list 
                      for i in range(0, len(df), chunksize)]
    return request_chunks

def request_waveforms(client:Client, request_list:list) -> tuple[Stream | None, str | None]:
    """
    Request waveforms (bulk) from the ObsPy FDSN client for a list of requests.
    :type client: obspy.clients.fdsn.Client
    :param client: ObsPy Client object
    :type request_list: list
    :param request_list: list of lists containing request parameters for each event-station pair
    :rtype: tuple[obspy.Stream | None, str | None]
    :return: tuple containing the response stream and an error message (if any)
    """
    try: 
        resp = client.get_waveforms_bulk(request_list)
        error = None
    except Exception as e:
        resp = None
        if hasattr(e, 'status_code'): 
            error = int(e.status_code) # not sure if this is problematic
            print(f"Error status code: {type(e).__name__}/{e.status_code}")
        else: 
            error = type(e).__name__
            print(f"Error in bulk request: {error}")
    return resp, error

def process_chunk(chunk:pd.DataFrame, resp:Stream, pwt:float,
                  _keep_columns:list=None) -> pd.DataFrame:
    """
    
    :type resp: obspy.Stream
    :type pwt: float
    :param pwt: process window threshold, used for matching traces to events
    """
    if _keep_columns is None: _keep_columns = chunk.columns.tolist()
    # initialise columns
    init_cols = ['error', 'traces', 'grouped_traces', 'trace_count', 'window_threshold', 'summed_trace_lengths', 'locations']
    bool_cols = ['multiple_locations']
    chunk[init_cols] = None; chunk[bool_cols] = False
    # sort traces into events
    chunk['traces'] = chunk.apply(lambda row: Stream(sort_traces(row, resp, pwt)), axis=1)
    # check for events with no traces
    chunk.loc[chunk['traces'].str.len() == 0, 'error'] = "data_exception"
    # check for streams with multiple locations
    chunk['locations'] = chunk['traces'].apply(multiple_location_check)
    chunk.loc[(chunk['locations'].str.len() > 1) & chunk['error'].isna(), ['error', 'multiple_locations']] = ("multiple_locations", True)

    # masks
    ml_mask = chunk['multiple_locations'] # mask for multiple locations
    sl_mask = ~ml_mask & chunk['error'].isna() # mask for single location
    # deal with single location streams
    if not chunk.loc[sl_mask].empty:
        chunk.loc[sl_mask, ['grouped_traces', 'error']] = _process_checks(evst=chunk.loc[sl_mask])
    # deal with multiple location streams
    
    if not chunk.loc[ml_mask].empty: 
        chunk.loc[ml_mask, ['grouped_traces', 'error']] = chunk.loc[ml_mask].apply(lambda row: _process_multiple_locations(row=row), axis=1).values

    mask = (chunk['error'].isna())
    mt_mask = (chunk['error'].str.contains('multiple_traces', na=False))
    ch_mask = mask | mt_mask

    # write channel column
    chunk['band'] = np.where(ch_mask, chunk['req_band'].str[0], None)

    # write status column
    de_mask = (chunk['error'].str.contains('data_exception', na=False))
    mc_mask = (chunk['error'].str.contains('missing_components', na=False))
    st_mask = (chunk['error'].str.contains('short_traces', na=False))
    chunk.loc[de_mask, 'status'] = 2 # data exception
    chunk.loc[mc_mask, 'status'] = 3 # missing components
    chunk.loc[st_mask, 'status'] = 4 # short traces
    chunk.loc[mt_mask, 'status'] = 5 # multiple traces
    chunk.loc[mask, 'status'] = 10 # success

    _keep_columns.extend(['band', 'error', 'grouped_traces'])
    out = chunk[_keep_columns]
    return out

def log_chunk(band_state:int, chunk_idx:int, chunk:pd.DataFrame, path:str) -> None:
    """
    Logs the results of chunk processing to a file. 
    
    """
    for _, group in (chunk.groupby('stcode')):
        log_path = group['log_path'].iloc[0]
        log_lines = [f"{row.Index}, {row.data_path}, {row.band}, {row.status}, {row.error}\n" 
                     for row in group.itertuples()]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a+') as f: f.write(''.join(log_lines))

    chunk_log = chunk[['index', 'data_path', 'band', 'status', 'error']]
    chunk_log.insert(0, 'chunk_idx', chunk_idx)
    chunk_log.insert(0, 'band_state', band_state)
    chunk_log.to_csv(path, mode='a+', header=False, index=False)
    return

### metadata fixes

def fix_delta(delta:float) -> float:
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

def check_polarity_ori(station:Station) -> int | None:
    """
    Checks the polarity of a station based on the orientation of its three components. 

    :type station: obspy.core.inventory.station.Station
    :param station: station inventory object
    :rtype: int | None
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

def check_polarity_sen(row:pd.Series, st=Stream, client:Client=None) -> int | None:
    """
    Checks the polarity of a station based on the instrument sensitivity of its three components.
    Requires evst dataframe. 

    :type row: pandas.Series
    :param row: row of the station inventory dataframe
    :type st: obspy.Stream
    :param st: stream containing the three components of the station
    :type client: obspy.clients.fdsn.Client
    :param client: ObsPy Client object
    :rtype: int | None
    :return: 0 for right-handed (negative polarity) and 1 for left-handed (positive polarity). None if 
    the polarity cannot be determined.
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
    

def update_sac(row, inv:Inventory, client:Client) -> list:
    """
    
    :type inv: obspy.core.inventory.inventory.Inventory
    :param inv: ObsPy Inventory object
    :type client: obspy.clients.fdsn.Client
    :param client: ObsPy Client object
    """
    
    print(f'{row.Index}: reading {row.data_path}')

    # read in traces
    try: st = read(f"{row.data_path}/*") # single stream for event-station pair
    except Exception as e:
        return [(1, row.Index, row.data_path, 'read', str(e))] # read error
    
    location = st[0].stats.location

    try: row_inv = inv.select(network=row.network, station=row.station, location=location, channel=f'{row.channel}H?')[0][0]
    except Exception as e:
        return [(1, row.Index, row.data_path, 'inv', str(e))]
    
    for chan in row_inv:
        if chan.azimuth is None or chan.dip is None:
            return [(1, row.Index, row.data_path, 'inv_chan', 'Missing azimuth or dip in channel inventory.')]

    lpspol = check_polarity_ori(row_inv)
    if lpspol is None: lpspol = check_polarity_sen(row, st, client)
    if lpspol is None: lpspol = 0 # default to right handed if polarity cannot be determined
    
    local_log = []

    for tr in st: 
        stats = tr.stats
        try: chan = row_inv.select(channel=stats.channel, location=stats.location)[0]
        except Exception as e:
            return [(1, row.Index, row.data_path, 'inv_chan', str(e))]

        evid = row.data_path.split('/')[3]
        file_name = f"{evid}_{row.station.lower()}.{stats.channel.lower()}"
        b = (pd.Timestamp(tr.stats.starttime.datetime) - row.evtime).total_seconds()
        e = (pd.Timestamp(tr.stats.endtime.datetime) - row.evtime).total_seconds()

        # check delta:
        if '99' in str(stats.delta): 
            delta = fix_delta(stats.delta)
            tr.stats.delta = delta # update trace stats (takes priority)
        else: delta = stats.delta

        header = {'npts': stats.npts,
                  'b': b,
                  'e': e,
                  'delta': delta,
                  'idep': 5, # unknown before response correction
                  'depmin': np.float32(np.min(tr[:])),
                  'depmax': np.float32(np.max(tr[:])),
                  'depmen': np.float32(np.mean(tr[:])),
                  'nzyear': row.evtime.year,
                  'nzjday': row.evtime.dayofyear,
                  'nzhour': row.evtime.hour,
                  'nzmin': row.evtime.minute,
                  'nzsec': int(row.evtime.second),
                  'nzmsec': int(row.evtime.microsecond / 1000),
                  'iztype': 11, # reference time is event time
                  'kevnm': evid,
                  'knetwk': stats.network,
                  'kstnm': stats.station,
                  'kcmpnm': stats.channel,
                  'cmpaz': chan.azimuth,
                  'cmpinc': chan.dip + 90,
                  'lpspol': lpspol,
                  'stla': row.stlat,
                  'stlo': row.stlon,
                  'stel': row.stelev,
                  'evla': row.evlat,
                  'evlo': row.evlon,
                  'evdp': row.evdep,
                  'lcalda': 1,
                  'user1': row.moment_magnitude,
                  }
        
        tr.stats.sac = header
    
        try: 
            tr.write(f"{row.data_path}/{file_name}", format='SAC')
            print(f'{row.Index}: updated {file_name}')
            local_log.append((0, row.Index, file_name, 'update', 'SAC updated'))
        except Exception as e:
            local_log.append((1, row.Index, file_name, 'write', str(e)))
            
    return local_log

# response

def prefilt():
    tap3=40.    # High-end taper width (%)
    tap2=70.    # Low-end taper width (%)
    hz2=0.002   # Lower bound of signal (500 seconds)
    hz3=2.      # Upper bound of signal (0.5 seconds)

    hz1=(1-0.01*tap2)*hz2 # Results in 0.0006 Hz
    hz4=(1+0.01*tap3)*hz3 # Results in 2.8 Hz

    return [hz1, hz2, hz3, hz4]

def remove_resp(row, st:Stream, inv:Inventory, prefilt:list):
    """ 
    Removes response from traces in a stream, returns the stream and a log of successes/failures

    :type row: <class 'pandas.core.frame.Pandas'> CHECK THIS
    :param row: Generated from itertuples
    :type st: obspy.core.stream.Stream
    :param st: 
    :type inv: obspy.core.inventory.inventory.Inventory
    :param inv: Response level inventory for the day's evst combinations
    :type prefilt: list
    :param prefilt: Corners of the pre-filter for response removal; [hz1, hz2, hz3, hz4]
    """

    location = st[0].stats.location

    try: row_idx = row.Index
    except: row_idx = row.name

    try: 
        row_inv = inv.select(network=row.network, station=row.station, location=location, channel=f'{row.channel}H?')
    except Exception as e:
        print(f'    >>> {row_idx}: failed to select row inventory')
        return [(1, row_idx, row.data_path, 'inv', str(e))]
    
    local_log = []

    for tr in st:
        stats = tr.stats
        # sorry - this is really fucking dumb
        try: kevnm = stats.sac.kevnm
        except: kevnm = row.data_path.split('/')[3]
        finally: file_name = f"{kevnm}_{row.station.lower()}.d{stats.channel[-1].lower()}"

        try: 
            tr \
                .detrend('linear') \
                .taper(0.05) \
                .remove_response(inventory=row_inv, pre_filt=prefilt, output='DISP') \
                .taper(0.1)
            
            tr.data *= 1e9 # convert to nm
            tr.stats.sac.lcalda = 0
            tr.stats.sac.idep = 6 # update header information. 6 means displacement in nm
            local_log.append((0, row_idx, file_name, 'resp', 'success'))

        except Exception as e:
            st.remove(tr) # remove failed trace from stream out
            print(f'    >>> {row_idx}, {stats.channel[-1].lower()}: failed to remove response')
            local_log.append((1, row_idx, file_name, 'resp', str(e)))

    return st, local_log

# rotation

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

