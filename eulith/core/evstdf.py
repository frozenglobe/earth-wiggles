from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup.tau import TauPyModel as TPM
from eulith.core.fdsn import adriaarray_nw, fdsn_connect
from eulith.core.utils import read_stnlist, read_cmt, latlon_filter, overlap

# work in progress

eida_cred = '/space/jhyl3/.eidatoken'
iris_cred = ('cocky_swanson', '6b8TEAqkjpQCUKz8') # (username, password)

def check_restricted(node:str,
                     client:Client=None, eida_cred:str=eida_cred,
                     lat_min:str=32, lat_max:str=72, lon_min:str=-12, lon_max:str=48,
                     starttime:pd.Timestamp=None, endtime:pd.Timestamp=None):
    """
    Checks for restricted stations in a given node, and returns lists of networks and stations to remove, as well as stations to check for restricted access.

    :type node: str
    :param node: FDSN node name
    :type eida_cred: str
    :param eida_cred: path to EIDA token file
    :type lat_min: float
    :param lat_min: minimum latitude for station filtering
    :type lat_max: float
    :param lat_max: maximum latitude for station filtering
    :type lon_min: float
    :param lon_min: minimum longitude for station filtering
    :type lon_max: float
    :param lon_max: maximum longitude for station filtering
    :type starttime: pd.Timestamp
    :param starttime: start time for station filtering
    :type endtime: pd.Timestamp
    :param endtime: end time for station filtering
    :return: tuple of lists (networks to remove, stations to remove, stations to check for restricted access)
    """
    if not client: client = fdsn_connect(node, cred=eida_cred)[0]
    if node == 'NOA': nw_check = ['HL']
    else: nw_check = []
    if node == 'LMU': nw_remove = ['ZJ']
    else: nw_remove = []

    stn_inv = client.get_stations(level='station', startbefore=endtime, endafter=starttime, \
                                  minlatitude=lat_min, maxlatitude=lat_max, minlongitude=lon_min, maxlongitude=lon_max)

    stn_remove = []
    stn_check = []
    for nw in stn_inv:
        if (nw.code in adriaarray_nw()) & (nw.code not in nw_check): continue
        stn_remove += [f"{nw.code}.{s.code}" for s in nw if s.restricted_status == 'closed']
        stn_check += [f"{nw.code}.{s.code}" for s in nw if s.restricted_status not in ['closed', 'open']]
    
    return nw_remove, stn_remove, stn_check

def test_restricted(stn_to_test:pd.DataFrame, 
                    node_path:str,
                    auth:bool, eida_cred:str=eida_cred):
    stn_remove = []
    # for stn in stn_check:
    stn_test_req = list(zip(
                            stn_to_test['network'], 
                            stn_to_test['station'], 
                            ['*'] * len(stn_to_test),
                            ['?H?'] * len(stn_to_test), 
                            stn_to_test['req_start'], 
                            stn_to_test['req_end']
                            ))
        
    count_403 = 0
    if not auth: eida_cred = None
    client = Client(node_path, eida_token=eida_cred)
    
    for req in stn_test_req:
        try:
            client.get_waveforms(*req)
        except Exception as e:
            if hasattr(e, 'status_code'): status = str(e.status_code)
            else: status = type(e).__name__
            if status == '403': 
                count_403 += 1
                stn_remove.append(req[1])
        
        if count_403 > 2:
            client = Client(node_path, eida_token=eida_cred)
            print("Re-authenticated client due to multiple 403 errors.")
            count_403 = 0 # reset count after re-authentication
    
    return stn_remove


def process_stnlist(stnlist_path:str,
                    node:str,
                    lat_min:float, lat_max:float, lon_min:float, lon_max:float,
                    sr_min:float=1., sr_max:float=100., 
                    bands:list=['L', 'B', 'H'], gain:str='H', 
                    starttime:pd.Timestamp=None, endtime:pd.Timestamp=None,
                    screen:bool=True, client:Client=None, verbose:bool=True):
    
    stnlist = read_stnlist(stnlist_path)

    # filtering
    stnlist_filt = stnlist[stnlist['dc'] == node] # filter by node
    stnlist_filt = latlon_filter(stnlist_filt, lat_min, lat_max, lon_min, lon_max) # filter by latlon
    stnlist_filt = stnlist_filt[(stnlist_filt['samplerate'].between(sr_min, sr_max)) & # filter by sample rate
                                (stnlist_filt['bandgain'].apply(lambda x: x[1] == gain)) & # filter by gain
                                (stnlist_filt['bandgain'].apply(lambda x: x[0] in bands)) & # only keep long corner periods
                                (stnlist_filt['channel'].apply(lambda x: len(x) == 3)) # filter by existence of three components
                                ]
    # stnlist_filt = stnlist_filt[stnlist_filt['bandgain'].apply(lambda x: x[1] == gain)]
    # stnlist_filt = stnlist_filt[stnlist_filt['bandgain'].apply(lambda x: x[0] in bands)]
    # stnlist_filt = stnlist_filt[stnlist_filt['channel'].apply(lambda x: len(x) == 3)]

    # remove restricted stations if screen is True
    if screen: 
        nw_remove, stn_remove, stn_check = check_restricted(node=node, client=client, eida_cred=eida_cred,
                                                            lat_min=lat_min, lat_max=lat_max, 
                                                            lon_min=lon_min, lon_max=lon_max, 
                                                            starttime=starttime, endtime=endtime)
        if nw_remove: stnlist_filt = stnlist_filt[~stnlist_filt['network'].isin(nw_remove)].reset_index(drop=True)

        if stn_remove: 
            mask = ~stnlist_filt['stn_code'].isin(stn_remove)
            stn_remove_list = stnlist_filt.loc[~mask, 'stn_code'].unique()
            print(f"    Removing {len(stn_remove_list)} stations from networks {stnlist_filt.loc[~mask, 'network'].unique()}")
            stnlist_filt = stnlist_filt[mask].reset_index(drop=True)

    keys = ['starttime', 'endtime']
    for k in keys: stnlist_filt[k] = pd.to_datetime(stnlist_filt[k], format='mixed')
    ### move this later because it is very time-consuming so best done after other checks

    # filter by operational time window
    dt_condition = stnlist_filt.apply(lambda x: overlap(x['starttime'], x['endtime'], starttime, endtime), axis=1)
    stnlist_filt = stnlist_filt[dt_condition]

    # rename columns to match evstdf
    stnlist_filt.rename(columns={'dc':'node', 'stn_code':'stcode', 'latitude':'stlat', 'longitude':'stlon', 'elevation':'stelev', 'starttime':'ststarttime', 'endtime':'stendtime'}, inplace=True)

    if verbose: print(f"Processed station list: {len(stnlist_filt['stcode'].unique())} stations remain after filtering")
    
    if screen: return stnlist_filt, stn_check
    else: return stnlist_filt

def process_cmt(cmt_path:str, starttime:pd.Timestamp, endtime:pd.Timestamp, verbose:bool=True):
    """
    read CMT catalogue and filter

    :type cmt_path: str
    :param cmt_path: path to the CMT file
    :type starttime: pd.Timestamp
    :param starttime: start time for filtering
    :type endtime: pd.Timestamp
    :param endtime: end time for filtering
    :return: pd.DataFrame with filtered CMT events
    """
    cmt_df = read_cmt(cmt_path)
    cmt_processed = cmt_df[cmt_df['evtime'].between(starttime, endtime)].reset_index(drop=True)
    cmt_processed['evpath'] = cmt_processed['evtime'].dt.strftime('data/%Y/%m/%y%m%d_%H%M%S/')
    cmt_processed = cmt_processed[['cmt', 'evtime', 'evlat', 'evlon', 'evdep', 'moment_magnitude', 'evpath']]
    if verbose:
        print(f"Processed CMT catalogue: {len(cmt_processed)} events remain after filtering")
    return cmt_processed


def evst_generator(cmt_df:pd.DataFrame,
                   stnlist:pd.DataFrame,
                   tt_model:str='ak135',
                   stn_check:list=[],
                   _node_path:str=None,
                   _auth:bool=False,
                   eida_cred:str=eida_cred):
    
    if stnlist.empty: raise SystemExit("No stations available.")

    if cmt_df.empty: raise SystemExit("No events available.")
    
    model = TPM(model=tt_model)
    print(f"    Initialised travel time model: {tt_model}")
    vg_min = 2.8 # minimum group velocity in km/s
    # epicentral distance range in km
    dist_min = 400
    dist_max = 18000
    # something to calculate a threshold for Mw
    ADIS = 4.9
    BDIS = 6.67e-5

    # Filter by events which occur within the station operational time window
    # Compute epicentral distance, filter by magnitude and distance range, remove locations, 
    # remove samplerate and channel in order to prepare for band merging, then merge bands
    # Merge bands (and sort)

    query = f"""
    WITH calc_dist_deg AS (
        SELECT 
            s.*, e.*,
                -- COMPUTE EPICENTRAL DISTANCE
            -- Haversine formula for distance in degrees
            degrees(acos(
                sin(radians(e.evlat)) * sin(radians(s.stlat)) + 
                cos(radians(e.evlat)) * cos(radians(s.stlat)) * cos(radians(s.stlon - e.evlon))
            )) AS dist_deg,
            dist_deg * 6371. * pi()/180. AS epi_dist
        FROM stnlist s
        JOIN cmt_df e
        ON e.evtime BETWEEN s.ststarttime AND s.stendtime
    )
    SELECT *
    FROM calc_dist_deg
        -- FILTER BY MAGNITUDE AND DISTANCE RANGE
    WHERE moment_magnitude >= LEAST(5.7, {ADIS} + {BDIS} * epi_dist)
        AND epi_dist BETWEEN {dist_min} AND {dist_max}
    """

    evst_match = duckdb.query(query).to_df()
    evst_match.drop_duplicates(subset=['stcode', 'bandgain', 'evtime'], inplace=True) # removes unique locations
    evst_match.drop(['location', 'stn_loc', 'channel', 'samplerate'], axis=1, inplace=True) # remove unneeded columns

    print(f"    Generated station-event pairs, filtered by magnitude and epicentral distance.")

    query = """
    SELECT
        node, network, station, stcode, cmt, evtime, evlat, evlon, evdep, moment_magnitude, evpath,
        FIRST(stlat) as stlat, FIRST(stlon) as stlon, FIRST(stelev) as stelev,
        FIRST(dist_deg) as dist_deg, FIRST(epi_dist) as epi_dist,
        LIST(bandgain ORDER BY
            CASE
                WHEN bandgain LIKE 'L%' THEN 1
                WHEN bandgain LIKE 'B%' THEN 2
                WHEN bandgain LIKE 'H%' THEN 3
                ELSE 4
            END, bandgain ASC
        ) AS bands
    FROM evst_match
    GROUP BY ALL
    """
    evst_band_combined = duckdb.query(query).to_df()

    # Compute theoretical P-wave travel times and window start/end times

    evst_band_combined['dist_deg_round'] = evst_band_combined['dist_deg'].round(1)
    evst_band_combined['evdep_round'] = (evst_band_combined['evdep'] / 2).round() * 2
    unique_pairs = evst_band_combined[['evdep_round', 'dist_deg_round']].drop_duplicates()
    cache = {}
    for _, row in unique_pairs.iterrows():
        tt = model.get_travel_times(source_depth_in_km=row.evdep_round, \
                                    distance_in_degree=row.dist_deg_round, \
                                    phase_list=['P'])
        if tt:
            cache[(row.evdep_round, row.dist_deg_round)] = tt[0].time 
            # [0] because we care only about the first p arrival
        
    evst_band_combined['ptime'] = [cache.get((z, d)) for z, d in zip(evst_band_combined.evdep_round, evst_band_combined.dist_deg_round)]

    print(f"    Computed theoretical P-wave travel times.")

    query = f"""
    WITH calc_window AS (
        WITH calc_period AS (
            SELECT *,
                LEAST(350, epi_dist / {vg_min} * 1.25) AS period
            FROM evst_band_combined
            WHERE ptime IS NOT NULL
        )
        SELECT *,
            evtime + INTERVAL (ptime) SECONDS - INTERVAL (5 * period) SECONDS AS window_start,
            evtime + INTERVAL (epi_dist / {vg_min} + 7 * period) SECONDS AS window_end
        FROM calc_period
    )
    SELECT * EXCLUDE (dist_deg, dist_deg_round, evdep_round, period)
    FROM calc_window
    ORDER BY stcode, evtime
    """
    evst_windows = duckdb.query(query).to_df()

    if evst_windows.empty:
        print("No station-event pairs available after computing request windows. Exiting.")
        raise SystemExit("No station-event pairs available.")

    print(f"    Computed request windows. Total station-event pairs to request: {len(evst_windows)}")

    '''
    Prepare datetimes for request, defines `log_path', 
    creates master `evst_results' dataframe,
    sets status to 4 (to be requested), defines `data_path'
    
    Statuses explained:
        0: success
        1: no data
        2: server exception
        3: processing error
        4: to be requested
    '''

    evst_windows['req_start'] = [UTCDateTime(t) for t in evst_windows['window_start']]
    evst_windows['req_end'] = [UTCDateTime(t) for t in evst_windows['window_end']]
    evst_windows['log_path'] = [f"logs/{n}/sta/{c}.log".lower() for n, c in zip(evst_windows.node, evst_windows.stcode)]

    evst_results = evst_windows.copy()
    evst_results['status'] = 0
    path_process = (pd.Series([f"{e}" + str(s)[0] + f"/{c}/" for e, s, c in zip(evst_results.evpath, evst_results.station, evst_results.stcode)])).to_list()
    evst_results['data_path'] = [p[:-1].lower() for p in path_process]

    # test availability of stations in networks with partially restricted access
    evst_stn_check = evst_results[evst_results['stcode'].isin(stn_check)]

    if not evst_stn_check.empty:
        print(f"Testing stations with ambiguous restricted status")
        stn_remove = test_restricted(evst_stn_check, node_path=_node_path, auth=_auth, eida_cred=eida_cred)
        # remove stations with 403 errors
        mask = evst_results['station'].isin(stn_remove)
        stc_remove_list = evst_results.loc[mask, 'stcode'].unique()
        evst_results = evst_results[~mask].reset_index(drop=True)
        print(f"    Removed {len(stc_remove_list)} stations from networks {evst_results.loc[~mask, 'network'].unique()}")

    return evst_results

def read_evst(path):
    """
    Given a .csv path, reads the evst csv file into the same dtypes as generation.

    :type path: str
    :param path: path to the evst csv file
    :return: pd.DataFrame with the evst data
    """
    import pandas as pd
    from obspy import UTCDateTime
    import ast
    columns = ['node', 'network', 'station', 'stcode', 'cmt', 'evtime', 'evlat', 'evlon', 'evdep', 'moment_magnitude', \
               'evpath', 'stlat', 'stlon', 'stelev', 'epi_dist', 'bands', 'ptime', 'window_start', 'window_end', \
               'req_start', 'req_end', 'log_path', 'status', 'data_path']
    dtypes = [str] * 6 + [np.float64] * 4 + [str] * 1 + [np.float64] * 4 + [str] * 1 + [np.float64] * 1 + [str] * 5 + [np.int64] * 1 + [str] * 1

    dtype_dict = dict(zip(columns, dtypes))

    evst_import = pd.read_csv(path, dtype=dtype_dict)

    keys = ['evtime', 'window_start', 'window_end']
    for k in keys: evst_import[k] = pd.to_datetime(evst_import[k], format='mixed')

    keys = ['req_start', 'req_end']
    for k in keys: evst_import[k] = [UTCDateTime(t) for t in evst_import[k]]
    evst_import['bands'] = evst_import['bands'].apply(lambda x: ast.literal_eval(x) if (", " in x) \
                                                             else [s.strip("''") for s in x.strip('[]').split(' ')])
    evst_import['bands'] = evst_import['bands'].apply(np.array)

    return evst_import
