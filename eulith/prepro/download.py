import os, argparse, signal, shutil, atexit, io, sys
import multiprocessing as mp
from pathlib import Path
import pandas as pd
from obspy import UTCDateTime
from obspy.taup.tau import TauPyModel as TPM
from obspy.clients.fdsn import Client
from traitlets import default

from eulith.core.fdsn import fdsn_connect
from eulith.core.evstdf import read_evst, process_stnlist, process_cmt, evst_generator
from eulith.prepro.functions import _prepare_chunk, _generate_next_evst, request_waveforms, process_chunk, _write_traces, _write_cmt, log_chunk

# start HERE

def chunk_generator(evst:pd.DataFrame, buffer_out:mp.Queue, 
                    nds:int,
                    cglog_path:str, mlog:io.TextIOWrapper,
                    chunksize:int=20,
                    band_state:int=0,
                    min_input_size:int=0):
    """
    Takes an evst and generates chunks which are put into a mp queue.

    :param evst: input evst dataframe
    :param buffer_out: output queue to store chunks    
    :param nds: Number of downstream processes receiving data from this generator
    :param cglog_path: Path to the chunk log file
    :param mlog: Master log file for logging each worker's progress
    """
    mlog.write(f">   {pd.Timestamp.now()}    CG: running    {os.getpid()}\n")
    mlog.write(f"    bs {band_state}\n")
    mlog.flush()
    print('Chunk generator running')

    to_group = evst[['req_start', 'req_end']].reset_index().values.tolist() 
        # resetting the index causes the index column to be formed
    
    chunk_counter = 0 # count the number of chunks
    cglog = open(cglog_path, 'a') # store chunk information

    while len(to_group) > min_input_size: 
        grouped = [to_group[0][0]]
        prev_end:UTCDateTime = to_group[0][2]
        to_group_again = []

        for i in range(1, len(to_group)):
            if len(grouped) >= chunksize:
                to_group_again.extend(to_group[i:])
                break
            if prev_end > to_group[i][1]: # if previous end time is greater than current start time, they overlap
                to_group_again.append(to_group[i])
            else:
                grouped.append(to_group[i][0]) # add index to current group
                prev_end = to_group[i][2] # update previous end time
        
        to_group = to_group_again
        chunk = _prepare_chunk(evst.loc[grouped],
                            band_state=band_state)
        buffer_out.put((chunk_counter, chunk))
        chunk_counter += 1

        # log 
        cglog.write(f"{chunk_counter} {len(grouped)} {','.join([str(x) for x in grouped])}\n")
        cglog.flush()

    cglog.close()
    for _ in range(nds): buffer_out.put(None)

def query(node_path:str, eida_cred:str,
          client:Client,
          buffer_in:mp.Queue, buffer_out:mp.Queue,
          nds:int,
          mlog:io.TextIOWrapper):
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
    :type mlog: TextIOWrapper
    :param mlog: Master log file for logging each worker's progress
    """
    print(f"    QUERY :: running :: {os.getpid()}")
    mlog.write(f">   {pd.Timestamp.now()}    QUERY: running    {os.getpid()}\n")
    mlog.flush()

    count_403 = 0
    if client is None:
        client = Client(node_path, eida_token=eida_cred)
    
    while True: 
        item = buffer_in.get()
        if item is None: break

        else: 
            chunk_idx, chunk = item

            # query server for waveforms
            resp_tuple = request_waveforms(client=client, request_list=chunk['bulk_request'].tolist())
            resp_error = resp_tuple[1]
            # handle error
            if isinstance(resp_error, int):
                if resp_error == 403: count_403 += 1
                if (count_403 > 2) or (resp_error == 401): 
                    print(f"    QUERY: Reinitialising client.")
                    client = Client(node_path, eida_token=eida_cred) # reinitialise client

            _keep_columns = ['index', 'network', 'station', 'stcode', 'cmt', 'evtime', 'evpath',
                             'bulk_request', 'req_band', 'req_start', 'req_end', 'log_path', 'data_path', 'status']
            chunk_out = chunk[_keep_columns]
            out = (chunk_idx, chunk_out, resp_tuple)
            buffer_out.put(out)
            if verbose: print(f"    QUERY: ({chunk_idx}) Chunk queried from server.")

    for _ in range(nds): buffer_out.put(None) # signal end of processing

    mlog.write(f">   {pd.Timestamp.now()}    QUERY: done    {os.getpid()}\n")
    mlog.flush()
    print(f"    QUERY :: done :: {os.getpid()}")

def process(buffer_in:mp.Queue, buffer_out:mp.Queue,
            nus:int, nds:int,
            mlog:io.TextIOWrapper,
            verbose:bool=False):
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

    :type mlog: TextIOWrapper
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
            chunk_idx, chunk, resp_tuple = item
            resp, resp_error = resp_tuple
            chunk_time_start = pd.Timestamp.now()
            if verbose: print(f"    PROCESSOR: ({chunk_idx}) Chunk started processing.")
            # handle request response
            if not resp:
                chunk['band'] = None
                if resp_error:
                    print(resp_error)
                    chunk['error'] = "http_error"
                    if isinstance(resp_error, int): chunk['status'] = resp_error # usually 403
                    else: chunk['status'] = 1 # chunk error but not specific error code
                else: 
                    chunk['status'] = 2 # no traces
                    chunk['error'] = "no_traces"
                if verbose: print(f"    PROCESSOR: ({chunk_idx}) No traces retrieved from server. Finished chunk.")
    
            else: # and if resp is successful
                # process chunk
                try: chunk = process_chunk(chunk=chunk, resp=resp, pwt=process_window_threshold) # 2-5 error, 10 success
                except Exception as e:
                    print(e) 
                    chunk['status'] = 6 # processing error
                    chunk['error'] = "process_error"
                    chunk['band'] = None
                chunk_time_end = pd.Timestamp.now()
                if verbose: print(f"    PROCESSOR: ({chunk_idx}) Finished processing chunk. PROCESS TIME: {(chunk_time_end - chunk_time_start).floor('s')}")
            chunk['status'] = chunk['status'].astype('int64') # ensure status is int64
            out = (chunk_idx, chunk)
            buffer_out.put(out)
    
    for _ in range(nds): buffer_out.put(None) # signal end of processing

    mlog.write(f">   {pd.Timestamp.now()}    PROCESSOR: done    {os.getpid()}\n")
    mlog.flush()
    print(f"    PROCESSOR :: done :: {os.getpid()}")

def write(evst:pd.DataFrame, band_state:int,
          chunk_log_path:str, evst_path:str,
          buffer_in:mp.Queue,
          nus:int,
          mlog:io.TextIOWrapper,
          verbose:bool=False):
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
            chunk_idx, chunk = item
            if verbose: print(f"    WRITER: ({chunk_idx}) Writing chunk and logs to disk.")

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
            log_chunk(band_state, chunk_idx, chunk, chunk_log_path)

            # update evst dataframe with chunk results
            evst.loc[chunk['index'], ['status', 'error', 'band']] = chunk[['status', 'error', 'band']].values

            if verbose: print(f"    WRITER: ({chunk_idx}) Finished writing chunk and logs to disk.")

    print(f"    WRITER: Writing evst results dataframe")
    evst.to_csv(evst_path, mode='w', index=False) # safe option

    mlog.write(f">   {pd.Timestamp.now()}    WRITER: done    {os.getpid()}\n")
    mlog.flush()
    print(f"    WRITER :: done :: {os.getpid()}")

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
                        required=True,
                        help='(REQUIRED) Node name (e.g., IRIS, NOA, etc.)')
    parser.add_argument('-s', '--starttime', type=str,
                        default="2020-01-01T00:00:00",
                        help='Start time for data download (format: YYYY-MM-DDTHH:MM:SS) defaults to 2020-01-01T00:00:00')
    parser.add_argument('-e', '--endtime', type=str,
                        default="2024-12-31T23:59:59",
                        help='End time for data download (format: YYYY-MM-DDTHH:MM:SS) defaults to 2024-12-31T23:59:59')
    parser.add_argument('-cs', '--chunksize', type=int,
                        default=50,
                        help='Number of events to query the server at one time; defaults to 50')
    parser.add_argument('-sr', '--samplerate', type=str,
                        default='1,100',
                        help='Sampling rate for the requested data (format: min,max) defaults to 1,100')
    parser.add_argument('-lat', '--latituderange', type=str,
                        default='32,72',
                        help='Latitude range for station selection (format: min,max) defaults to 32,72')
    parser.add_argument('-lon', '--longituderange', type=str,
                        default='-12,48',
                        help='Longitude range for station selection (format: min,max) defaults to -12,48')
    parser.add_argument('-pwt', '--process_window_threshold', type=int,
                        default=300,
                        help='Threshold for processing window length (in seconds) defaults to 300')
    parser.add_argument('-c', '--cred', type=str,
                        default=None,
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

    if not args.node:
        parser.print_help(); sys.exit(0)
    node = args.node.lower()
    starttime = pd.to_datetime(args.starttime)
    endtime = pd.to_datetime(args.endtime)
    chunksize = args.chunksize # default 50
    process_window_threshold = args.process_window_threshold # default 300
    eida_cred = args.cred # EIDA token
    read_evst_path = args.read_existing
    verbose = args.verbose

    sr_min, sr_max = [float(x) for x in args.samplerate.split(',')]
    lat_min, lat_max = [float(x) for x in args.latituderange.split(',')]
    lon_min, lon_max = [float(x) for x in args.longituderange.split(',')]

    gain = 'H'
    tt_model = 'ak135'

    # overrides
    # write_path = ''
    # cmt_path = ''
    # stnlist_path = ''
    # node = ''
    # starttime = pd.to_datetime('2020-01-01T00:00:00')
    # endtime = pd.to_datetime('2024-12-31T23:59:59')
    # lat_min, lat_max = 44., 46.5
    # lon_min, lon_max = 2.5, 7.5
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
    # master log
    mlog_path = os.path.join(log_dir, f"download_{run_specific}.log")
    mlog = open(mlog_path, 'w')
    # chunk generator log
    cglog_path = os.path.join(log_dir, f"cg_{run_specific}.log")
    with open(cglog_path, 'w') as cglog: 
        cglog.write("chunk_idx chunk_size idxs\n")
    # chunk log
    clog_path = os.path.join(log_dir, f"chunk_{run_specific}.log")
    with open(clog_path, 'w') as clog:
        clog.write("bs,chunk_idx,idx,data_path,band,status,error\n")
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

    # generate `evst_results' dataframe, either from scratch or from reading in existing dataframe
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

        evst_results.to_csv(evst_path, mode='w', index=False)

    request_state = True
    band_state = 0

    cols_condensed = ['network', 'station', 'stcode', 
                      'cmt', 'evtime', 'evpath', 'bands',
                      'window_start', 'window_end',
                      'req_start', 'req_end',
                      'log_path', 'data_path', 
                      'status']
    input_evst = evst_results[cols_condensed].copy()

    while not input_evst.empty:
        print(f"Starting download for band state: {band_state}")

        # set up multiprocessing
        # define buffers
        buffer_chunk = mp.Queue()
        buffer_resp = mp.Queue()
        buffer_proc = mp.Queue()

        # define processes
        generate_chunks = mp.Process(target=chunk_generator, args=(input_evst, buffer_chunk,
                                                                   1,
                                                                   cglog_path, mlog,
                                                                   chunksize,
                                                                   band_state,
                                                                   0))
        query_client = mp.Process(target=query, args=(node_path, eida_cred,
                                                      client,
                                                      buffer_chunk, buffer_resp, 
                                                      1, 
                                                      mlog))
        processor = mp.Process(target=process, args=(buffer_resp, buffer_proc,
                                                     1, 1,
                                                     mlog,
                                                     True))
        writer = mp.Process(target=write, args=(evst_results, band_state,
                                                clog_path, evst_path,
                                                buffer_proc,
                                                1, 
                                                mlog,
                                                True))

        # start all processes
        generate_chunks.start()
        query_client.start()
        processor.start()
        writer.start()
        print('> All processes started.')

        # block until all processes are done
        generate_chunks.join()
        print('< Chunk joined')
        query_client.join()
        print('< Query client joined')
        processor.join()
        print('< Processor joined')
        writer.join()
        print('< Writer joined')
        print('< All processes done.')

        # generate the next input evst
        input_evst = read_evst(evst_path, new=True)[cols_condensed]
        input_evst = _generate_next_evst(input_evst, band_state=band_state)
        band_state += 1
        print(f'increasing band state to {band_state}')
    
    print('No more events to process. Exiting.')