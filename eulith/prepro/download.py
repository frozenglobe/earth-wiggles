import os, argparse, signal, shutil, atexit
import multiprocessing as mp
from pathlib import Path
import pandas as pd
from obspy.taup.tau import TauPyModel as TPM
from obspy.clients.fdsn import Client

from eulith.core.fdsn import fdsn_connect
from eulith.core.evstdf import read_evst, process_stnlist, process_cmt, evst_generator
from eulith.prepro.functions import event_sort, request_waveforms, process_chunk, _write_traces, _write_cmt, log_chunk


# initialise variables
lat_min, lat_max = 32, 72
lon_min, lon_max = -12, 48
sr_min, sr_max = 1.0, 100.
gain = 'H'
tt_model = 'ak135'
chunksize = 50
process_window_threshold = 300

# EarthScope / IRIS
user = 'cocky_swanson'
password = '6b8TEAqkjpQCUKz8'

def chunk_generator(chunksize:int, 
                    buffer_in:mp.Queue, buffer_out:mp.Queue,
                    nus:int, nds:int):
    none_count:int = 0
    is_last = False
    chunk_store:list[pd.DataFrame] = []
    to_group:list[list] = []
    chunk_idx = 0

    while True:
        item:pd.DataFrame = buffer_in.get()
        if item is None: 
            none_count += 1
            if none_count == nus: is_last = True

        else:
            chunk_in:pd.DataFrame = item
            chunk_store.append(chunk_in)
            chunk_store_length:int = len(to_group) + \
                                 sum([len(chunk) for chunk in chunk_store 
                                      if isinstance(chunk, pd.DataFrame)])
            if chunk_store_length < 3*chunksize: continue
        
        # if enough rows, process
        if chunk_store: 
            to_group_df:pd.DataFrame = pd.concat(chunk_store, ignore_index=True)
            chunk_store = [] # reset
            to_group.extend(to_group_df[['Index', 'window_start', 'window_end']].values.tolist())
        groups, to_group = event_sort(rows_to_sort=to_group, chunksize=chunksize, last=is_last)
        for group in groups:
            chunk_idx += 1
            chunk_out:pd.DataFrame = to_group_df.loc[to_group_df['Index'].isin(group)].copy()
            out = (chunk_idx, chunk_out)
            buffer_out.put(out)

        if is_last: break

    for _ in range(nds): buffer_out.put(None) # signal end of processing

def load_chunks(chunks:list[pd.DataFrame], band_state:int, load:bool,
                buffer_out:mp.Queue,
                nds:int,
                mlog):
    for chunk_idx, chunk in enumerate(chunks):
        out = (band_state, chunk_idx, chunk)
        buffer_out.put(out)

    if not load: 
        for _ in range(nds): buffer_out.put(None) # signal end of processing

def reload_chunks(chunksize:int,
                  buffer_in:mp.Queue, buffer_out:mp.Queue,
                  nus:int, nds:int,
                  mlog):
    print(f"    RELOAD :: running :: {os.getpid()}")
    mlog.write(f">   {pd.Timestamp.now()}    RELOAD: running    {os.getpid()}\n")
    mlog.flush()

    none_count = 0
    while True: 
        item = buffer_in.get()
        rows_out = []
        if item is None:   
            none_count += 1
            if none_count == nus: break

        else: 
            row, _ = item

            while len(rows_out) < chunksize:
                rows_out.append(row)

            
            out = rows_out
            buffer_out.put(out)
            rows_out = [] # flush once in queue

def query(node_path:str, eida_cred:str,
          buffer_in:mp.Queue, buffer_out:mp.Queue,
          nus: int, nds:int,
          mlog):
    """
    Query FDSN datacentre for waveforms

    :type chunks: list[pd.DataFrame]
    :param chunks: List of chunks to process
    :type node_path: str
    :param node_path: URL to initialise ObsPy Client
    :type eida_cred: str
    :param eida_cred: Path to EIDA credentials file
    :type buffer_out: multiprocessing.Queue
    :param buffer_out: Output queue to store chunks and responses for processing
    :type nds: int
    :param nds: Number of downstream processes (consumers) receiving data from this reader
    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """
    print(f"    QUERY :: running :: {os.getpid()}")
    mlog.write(f">   {pd.Timestamp.now()}    QUERY: running    {os.getpid()}\n")
    mlog.flush()

    count_403 = 0
    client = Client(node_path, eida_token=eida_cred)

    none_count = 0
    while True: 
        if item is None:   
            none_count += 1
            if none_count == nus: break

        else: 
            item = buffer_in.get()
            chunk_info, chunk = item
            chunk_idx, no_chunks = chunk_info
            # build bulk request list
            chunk['bulk_request'] = list(zip(chunk['network'], 
                                            chunk['station'], 
                                            ['*'] * len(chunk),
                                            chunk['req_chan'], 
                                            chunk['req_start'], 
                                            chunk['req_end']
                                            ))
            # query server for waveforms
            resp_tuple = request_waveforms(client=client, request_list=chunk['bulk_request'].tolist())
            resp_error = resp_tuple[1]
            # handle error
            if isinstance(resp_error, int):
                if resp_error == 403: count_403 += 1
                if (count_403 > 2) or (resp_error == 401): 
                    print(f"    QUERY: Reinitialising client.")
                    client = Client(node_path, eida_token=eida_cred) # reinitialise client

            _keep_columns = ['Index', 'network', 'station', 'stcode', 'cmt', 'evtime', 'evpath',
                            'bulk_request', 'log_path', 'data_path', 'status']
            chunk_out = chunk[_keep_columns]
            out = (chunk_info, chunk_out, resp_tuple, _keep_columns)
            buffer_out.put(out)
            print(f"    QUERY: ({chunk_idx}/{no_chunks-1}) Chunk queried from server. Putting in buffer for processing.")

    for _ in range(nds): buffer_out.put(None) # signal end of processing

    print(f"    QUERY :: done :: {os.getpid()}")
    mlog.write(f">   {pd.Timestamp.now()}    QUERY: done    {os.getpid()}\n")
    mlog.flush()

def process(buffer_in:mp.Queue, buffer_outs:mp.Queue, buffer_outf:mp.Queue,
            nus:int, ndss:int, ndsf:int,
            mlog):
    """
    Process server response to chunk bulk request.
    
    :type no_chunks: int
    :param no_chunks: Total number of chunks to process.

    :type buffer_in: multiprocessing.Queue
    :param buffer_in: Input queue containing chunks to process.
    :type buffer_out: multiprocessing.Queue
    :param buffer_out: Output queue to store processed chunk results.

    :type nus: int
    :param nus: Number of upstream processes (producers) sending data to this processor.
    :type nds: int
    :param nds: Number of downstream processes (consumers) receiving data from this processor.

    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress.
    """
    print(f"    PROCESSOR :: running :: {os.getpid()}")
    mlog.write(f">   {pd.Timestamp.now()}    PROCESSOR: running    {os.getpid()}\n")
    mlog.flush()

    none_count = 0
    while True:
        item = buffer_in.get()
        if item is None:
            none_count += 1
            if none_count == nus: break

        else:
            chunk_info, chunk, resp_tuple, _keep_columns = item
            chunk_idx, no_chunks = chunk_info
            resp, resp_error = resp_tuple
            chunk_time_start = pd.Timestamp.today()
            counter = f"{chunk_idx}/{no_chunks-1}"
            print(f"    PROCESSOR: ({counter}) Chunk started processing.")
            # handle request response
            if not resp:
                chunk['channel'] = None
                if resp_error:
                    chunk['error'] = "http_error"
                    if isinstance(resp_error, int): chunk['status'] = resp_error # usually 403
                    else: chunk['status'] = 1 # chunk error but not specific error code
                else: 
                    chunk['status'] = 2 # no traces
                    chunk['error'] = "no_traces"
                print(f"    PROCESSOR: ({counter}) No traces retrieved from server. Finished chunk.")
    
            else: # and if resp is successful
                # process chunk
                try: chunk = process_chunk(chunk=chunk, resp=resp, pwt=process_window_threshold, _keep_columns=_keep_columns) # 2-5 error, 10 success
                except: 
                    chunk['status'] = 6 # processing error
                    chunk['error'] = "process_error"
                chunk_time_end = pd.Timestamp.today()
                print(f"    PROCESSOR: ({counter}) Finished processing chunk. PROCESS TIME: {(chunk_time_end - chunk_time_start).floor('s')}")
            
            out = (chunk_idx, no_chunks, chunk)
            buffer_outs.put(out)

            # to retry
            condition = (~chunk['error'].isna) & \
                        (~chunk['status'].isin([403])) & \
                        (chunk['band_state'] < chunk['bands'].str.len())
            chunk_retry = chunk.loc[condition]
            chunk_retry['req_chan'] = f"{chunk_retry['bands'].str[chunk_retry['band_state']]}?"
            out = chunk_retry
            buffer_outf.put(out)
    
    for _ in range(ndss): buffer_outs.put(None) # signal end of processing
    for _ in range(ndsf): buffer_outf.put(None) # signal end of processing

    print(f"    PROCESSOR :: done :: {os.getpid()}")
    mlog.write(f">   {pd.Timestamp.now()}    PROCESSOR: done    {os.getpid()}\n")
    mlog.flush()

def write(chunk_log_path:str, evst_path:str, evst:pd.DataFrame,
          buffer_in:mp.Queue, buffer_out:mp.Queue,
          nus:int, nds:int,
          mlog):
    """
    
    
    """
    print(f"    WRITER :: running :: {os.getpid()}")
    mlog.write(f">   {pd.Timestamp.now()}    WRITER: running    {os.getpid()}\n")
    mlog.flush()

    none_count = 0
    while True:
        item = buffer_in.get()
        if item is None:
            none_count += 1
            if none_count == nus: break
            
        else:
            chunk_idx, no_chunks, chunk = item
            counter = f"{chunk_idx}/{no_chunks-1}"
            print(f"    WRITER: ({counter}) Writing chunk and logs to disk.")

            mask = (chunk['error'].isna())
            # write traces to disk
            if not chunk.loc[mask].empty: _write_traces(chunk.loc[mask])
            mt_mask = (chunk['error'].str.contains('multiple_traces', na=False))
            if not chunk.loc[mt_mask].empty: _write_traces(chunk.loc[mt_mask], mt=True)

            # write CMT files to disk
            cmt_mask = mask | mt_mask
            cmt_evst = chunk.loc[cmt_mask].drop_duplicates(subset=['evtime'])
            if not cmt_evst.empty: _write_cmt(cmt_evst)

            # log chunk (write to log files)
            log_chunk(chunk_idx, chunk, chunk_log_path)

            # update evst dataframe with chunk results
            evst.loc[chunk['Index'], ['status', 'error', 'channel']] = chunk[['status', 'error', 'channel']]

            out = chunk[~chunk['error'].isna(), ['Index', 'bands', 'window_start', 'window_end', 'status']]
            buffer_out.put(out)

            print(f"    WRITER: ({counter}) Finished writing chunk and logs to disk.")

    print(f"    WRITER: Writing evst results dataframe")
    evst.to_csv(evst_path, mode='w', index=False)
    print(f"    WRITER :: done :: {os.getpid()}")
    mlog.write(f">   {pd.Timestamp.now()}    WRITER: done    {os.getpid()}\n")
    mlog.flush()
    return evst

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(prog='Download',
                                    description='Download seismic data from FDSN datacentres')
    parser.add_argument('-w', '--write_path', type=str, 
                        default='./',
                        help='Path to write downloaded data')
    parser.add_argument('-cp', '--cmt_path', type=str,
                        default='./cmt_catalogue.nds',
                        help='Path to CMT catalog file')
    parser.add_argument('-sp', '--stnlist_path', type=str,
                        default='./stnlist.csv',
                        help='Path to station list file')
    parser.add_argument('-n', '--node', type=str, 
                        help='Node name (e.g., IRIS, NOA, etc.)')
    parser.add_argument('-s', '--starttime', type=str,
                        default="2020-01-01T00:00:00",
                        help='Start time for data download (format: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('-e', '--endtime', type=str,
                        default="2024-12-31T23:59:59",
                        help='End time for data download (format: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('-c', '--cred', type=str,
                        default='.eidatoken',
                        help='Path to EIDA credentials file')
    parser.add_argument('-r', '--read_existing', type=str,
                        default=None,
                        help='Path to existing evst dataframe (csv) to read')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output for writer')
    args = parser.parse_args()

    write_path = args.write_path
    cmt_path = args.cmt_path
    stnlist_path = args.stnlist_path
    node = args.node.lower()
    starttime = pd.to_datetime(args.starttime)
    endtime = pd.to_datetime(args.endtime)
    eida_cred = args.cred # EIDA token
    read_evst_path = args.read_existing
    verbose = args.verbose

    # overrides
    # write_path = ''
    # cmt_path = ''
    # stnlist_path = ''
    # node = ''
    # starttime = pd.to_datetime('2020-01-01T00:00:00')
    # endtime = pd.to_datetime('2024-12-31T23:59:59')
    if (node == 'ingv'): chunksize = 10
    elif (node == 'odc'): chunksize = 50
    elif (node == 'raspishake'): chunksize = 15
    if (node == 'ipgp'): process_window_threshold = 1800
    
    print(f"Starting download process for node: {node.upper()}")
    print(f"Time range: {starttime} to {endtime}")
    print(f"Station list: {stnlist_path}")
    print(f"CMT catalog: {cmt_path}")

    # initialise client
    client, auth, node_path = fdsn_connect(node=node.upper(), cred=eida_cred)
    # auth is a boolean indicating whether access to node was authenticated 
    print(f"    Connected to FDSN node: {node.upper()} (auth={auth})")
    if not auth: eida_cred = None # if not authenticated, set eida_cred to None

    # initialising logs
    os.chdir(write_path)
    print(f"Current working directory: {os.getcwd()}")
    script_time_start = pd.Timestamp.today()
    run_specific = f"{node}_{script_time_start.strftime('%y%m%d')}"
    log_dir = Path(f"logs/{node}/")
    log_dir.mkdir(parents=True, exist_ok=True)
    mlog_path = os.path.join(log_dir, f"download_{run_specific}.log")
    mlog = open(mlog_path, 'w')
    clog_path = os.path.join(log_dir, f"chunk_{run_specific}.log")
    evst_path = os.path.join(log_dir, f"evst_results_{run_specific}.csv")
    old_evst_path = os.path.join(log_dir, f"_evst_results_{run_specific}.csv")
    

    # exit handling
    def exit_handler():
        print("\n[CLEANUP] Shutting down...")

        print(f"[CLEANUP] Saving to {evst_path}...")
        if Path(evst_path).exists(): shutil.copy(evst_path, old_evst_path) # avoid overwriting
        try:
            evst_results.to_csv(evst_path, mode='w', index=False)
            print("[CLEANUP] Save successful. Exiting.")
        except NameError: pass

    atexit.register(exit_handler)
    def handle_external_stop(signum, frame):
        raise SystemExit("[SIGNAL] External stop signal received.")

    signal.signal(signal.SIGINT, handle_external_stop)  # Catch Ctrl+C
    signal.signal(signal.SIGTERM, handle_external_stop) # Catch 'kill' commands

    # generate event-station pairs or read in existing dataframe
    if read_evst_path: 
        evst_results = read_evst(read_evst_path)
        print(f"Read existing evst dataframe from: {read_evst_path}")
        evst_results = evst_results[evst_results['evtime'].between(starttime, endtime)].reset_index(drop=True)
        evst_results.loc[:, 'status'] = 0

    else: 
        print(f"Generating event-station pairs")
        # generate station list
        stnlist, stn_check = process_stnlist(stnlist_path=stnlist_path,
                                             node=node.upper(),
                                             lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                                             sr_min=sr_min, sr_max=sr_max,
                                             bands=['L', 'B', 'H'], gain=gain,
                                             starttime=starttime, endtime=endtime,
                                             screen=True, client=client, verbose=True)
        # generate CMT dataframe
        cmt_df = process_cmt(cmt_path=cmt_path, starttime=starttime, endtime=endtime, verbose=True)
        # generate event-station pairs
        evst_results = evst_generator(cmt_df=cmt_df, stnlist=stnlist,
                                     tt_model=tt_model,
                                     stn_check=stn_check,
                                     _node_path=node_path,
                                     eida_cred=eida_cred)

    # set up multiprocessing
    # define buffers
    buffer_chunk = mp.Queue()
    buffer_resp = mp.Queue(maxsize=3)
    buffer_proc = mp.Queue()

    # define processes
    query_client = mp.Process(target=query, args=(node_path, eida_cred, 
                                                  buffer_chunk, buffer_resp, 
                                                  1, 1, 
                                                  mlog))
    processor = mp.Process(target=process, args=(buffer_resp, buffer_proc,
                                                 1, 1,
                                                 mlog))
    writer = mp.Process(target=write, args=(clog_path, evst_results,
                                            buffer_proc,
                                            1, 
                                            mlog))

    # start all processes
    query_client.start()
    processor.start()
    writer.start()
    print('All processes started.')


    
    band_state = 0
    request_state = True

    
    while request_state == True:
        # generate chunks
        evst_results['req_chan'] = evst_results['bands'].str[band_state]

        evst_results_condensed = evst_results[['network', 'station', 'stcode', 
                                            'cmt', 'evpath',
                                            'window_start', 'window_end',
                                            'req_start', 'req_end', 'req_chan',
                                            'log_path', 'data_path', 
                                            'status']]
        chunks = chunk_generator(evst=evst_results_condensed, chunksize=chunksize)
        no_chunks = len(chunks)

        # add chunks to buffer
        load_chunks(chunks=chunks, band_state=band_state, load=request_state,
                    buffer_out=buffer_chunk,
                    nds=1,
                    mlog=mlog)


    


    # block until all processes are done
    query_client.join()
    processor.join()
    writer.join()

    print('All processes done.')