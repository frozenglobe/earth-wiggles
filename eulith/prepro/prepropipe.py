master_path = '/space/jhyl3/data_retrieval/europe'
starttime = '2022-01-01T00:00:00'
endtime = '2024-12-31T23:59:59'

njob_rr = 8
njob_rot = 1
njob_res = 1
njob_taup = 1

no_evs = 8 # how many days to keep in the queue

import os, argparse, signal

from scipy import stats
os.environ['OPENBLAS_NUM_THREADS'] = '5'
import pandas as pd
from obspy import read
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client
import multiprocessing as mp
from pathlib import Path
import fortranformat as ff

from eulith.core import read_evst, fdsn_registry_dict
from functions import prefilt, remove_resp, rotate_rt

os.chdir(master_path)
print(f"CWD: ./{master_path.split('/')[-1]}")

parser = argparse.ArgumentParser(prog='Pre-processing pipeline',
                                 description='Remove response, rotate, resample,' \
                                             'calculate firstP')
parser.add_argument('-n', '--node', type=str, 
                    help='Node name (e.g., IRIS, NOA, etc.)')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print verbose output for writer')
args = parser.parse_args()
node = args.node.lower()

# client initialisation
if node == 'ign': node_path = 'https://fdsnws.sismologia.ign.es'
else: node_path = fdsn_registry_dict().get(node.upper())
try: client = Client(node_path)
except: 
    try: client = Client(node.upper())
    except: raise SystemExit(f"Failed to initialize FDSN client for {node.upper()}")

ttmodel = TauPyModel(model="ak135")

# evst
evst_mpath = f"logs/{node.lower()}/evst_master_{node.lower()}.csv"
new_mpath = f"logs/{node.lower()}/evst_master_{node.lower()}_NEW.csv"
evst = read_evst(evst_mpath)
evst = evst[evst['evtime'].between(pd.to_datetime(starttime), pd.to_datetime(endtime))]
evst = evst[evst['status'] == 11] # complete metadata

script_time_start = pd.Timestamp.today()
print(f"Script started: {script_time_start}")

# exit handling
def handle_external_stop(signum, frame):
    raise SystemExit(f"\nReceived signal {signum}. Stopping processes...")
signal.signal(signal.SIGINT, handle_external_stop) # catch ctrl+c
signal.signal(signal.SIGTERM, handle_external_stop) # catch kill

# LOGGING

# 'rr' = Resp Rm, 'rot' = Rotate, 'res' = Resample
prepro_tasks = ('prepro', 'rr', 'rot', 'res', 'taup')
run_specific = f"{node.lower()}_{script_time_start.strftime('%y%m%d')}"
log_paths = [f'logs/{item}/{item}_update_{run_specific}.txt' for item in prepro_tasks]

for log_path in log_paths:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f: f.write(f'{script_time_start}\n')

logs = {task: open(log_path, 'a') for task, log_path in zip(prepro_tasks, log_paths)}

master_log_path = f'logs/prepro/master_{run_specific}.txt'
mlog = open(master_log_path, 'w')
mlog.write(f'{script_time_start}\n') # write to summary log
mlog.write(f"Main script PID: {os.getpid()}\n")
mlog.flush()

# define read process
def read_traces(evst:pd.DataFrame,
                buffer_out:mp.Queue,
                nds:int,
                f, mlog):
    """ 
    Reads traces from evst['data_path'] and puts them, grouped by day (helps with
    response removal), into the buffer for processing

    :type buffer_out: multiprocessing.queues.Queue
    :param buffer_out: Output buffer containing groups of read traces
    :type evst: pandas.DataFrame
    :param evst: Event-station master dataframe for metadata

    :type nds: int
    :param nds: Number of None values required to signal completion to downstream processes

    :type f: <class '_io.TextIOWrapper'>
    :param f: Log file
    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """

    print(f'READ_TRACES: running :: {os.getpid()}')
    mlog.write(f">   {pd.Timestamp.now()}    READ_TRACES: running         {os.getpid()}\n")
    mlog.flush()

    groups = evst.groupby(evst['evtime'].dt.to_period('D'))

    for group_name, group in groups:
        group_entries = []
        for row in group.itertuples():
            try: 
                st = read(f"{row.data_path}/*.?h?")
                group_entries.append((row.Index, st))
            except Exception as e:
                f.write(f"{(1, row.Index, row.data_path, 'read', str(e))}\n")
                f.flush()
        
        buffer_out.put((group_name, group, group_entries))
        # print(f'    >> READ_TRACES read in {group_name}', flush=True)

    for _ in range(nds): buffer_out.put(None) # Signal that process is done

    print(f'    >> READ_TRACES: complete :: {os.getpid()}', flush=True)
    mlog.write(f">>  {pd.Timestamp.now()}    READ_TRACES: complete        {os.getpid()}\n")
    mlog.flush()
    
    return buffer_out

# define write process
def write_traces(evst:pd.DataFrame,
                 buffer_in:mp.Queue, buffer_out:mp.Queue,  
                 nus:int, nds:int, 
                 f, mlog):
    """ 
    Writes traces to disk, dumps status to write into evst

    :type evst: pandas.core.frame.DataFrame
    :param evst: Event-station master dataframe for metadata

    :type buffer_in: multiprocessing.queues.Queue
    :param buffer_in: Input buffer containing traces to be written
    :type buffer_out: multiprocessing.queues.Queue
    :param buffer_out: Output buffer containing evst status updates

    :type nus: int
    :param nus: Number of upstream processes to wait for before exiting
    :type nds: int
    :param nds: Number of None values required to signal completion to downstream processes

    :type f: <class '_io.TextIOWrapper'>
    :param f: Log file
    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """

    print(f'WRITE_TRACES: running :: {os.getpid()}')
    mlog.write(f">   {pd.Timestamp.now()}    WRITE_TRACES: running        {os.getpid()}\n")
    mlog.flush()

    none_count = 0
    while True:
        # get item from buffer
        item = buffer_in.get()
        if item is None: 
            none_count += 1
            if none_count == nus: break
        
        else: 
            #-vvvvvv- CODE HERE -vvvvvv-#
            row_idx, st, status = item
            if args.verbose: print(f'   >> WRITER received row {row_idx}', flush=True)
        
            # get row from evst for metadata
            row = evst.loc[row_idx]

            # write traces
            write_state = 0
            for tr in st:
                # again i am really sorry
                try: kevnm = stats.sac.kevnm
                except: kevnm = row.data_path.split('/')[3]
                finally: file_name = f"{kevnm}_{row.station.lower()}.d{tr.stats.channel[-1].lower()}"
                try: 
                    tr.write(f"{row.data_path}/{file_name}", format='SAC')
                    f.write(f"{(0, row_idx, file_name, 'write', 'success')}\n")
                    
                except Exception as e:
                    if args.verbose: print(f'   >>> {row_idx}, {tr.stats.channel[-1].lower()}: failed to write {file_name}')
                    f.write(f"{(1, row_idx, file_name, 'write', str(e))}\n")
                    write_state += 1
            
            if write_state == 0: buffer_out.put((row_idx, status))
            else: buffer_out.put((row_idx, 19)) # 19 = write failure
            
            f.flush()
            #-^^^^^^- CODE HERE -^^^^^^-#
    
    for _ in range(nds): buffer_out.put(None) # Signal that process is done

    print(f'    >> WRITE_TRACES: complete :: {os.getpid()}', flush=True)
    mlog.write(f">>  {pd.Timestamp.now()}    WRITE_TRACES: complete       {os.getpid()}\n")
    mlog.flush()

# define middle processes
def process_rr(client:Client,
               buffer_in:mp.Queue, buffer_out:mp.Queue, 
               nus:int, nds:int,
               f, mlog):
    """ 
    Response removal + metadata update

    :type client: obspy.clients.fdsn.Client
    :param client: FDSN client for fetching station inventories

    :type buffer_in: multiprocessing.queues.Queue
    :param buffer_in: Input buffer containing groups of traces to process
    :type buffer_out: multiprocessing.queues.Queue
    :param buffer_out: Output buffer containing groups of response-removed traces

    :type nus: int
    :param nus: Number of upstream processes to wait for before exiting
    :type nds: int
    :param nds: Number of None values required to signal completion to downstream processes

    :type f: <class '_io.TextIOWrapper'>
    :param f: Log file
    :param mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """

    print(f'RESPONSE_REMOVAL: running :: {os.getpid()}')
    mlog.write(f">   {pd.Timestamp.now()}    RESPONSE_REMOVAL: running    {os.getpid()}\n")
    mlog.flush()

    none_count = 0
    while True:
        # get item from buffer
        item = buffer_in.get()
        if item is None: 
            none_count += 1
            if none_count == nus: break
        
        else: 
            #-vvvvvv- CODE HERE -vvvvvv-#
            group_name, group, group_entries = item
            print(f'> PROCESSOR received {group_name}')
            
            # remove response
            try: 
                group_inv = client.get_stations(network=','.join(group['network'].unique()), 
                                            station=','.join(group['station'].unique()),
                                            channel=','.join([f"{band}H?" for band in group['channel'].unique()]),
                                            level='response',
                                            startbefore=group['req_start'].max(),
                                            endafter=group['req_end'].min())
            except Exception as e:
                print(f'    >>> {group_name}: failed to get inventory')
                for ev in group_entries: 
                    row_idx, _ = ev
                    row = group.loc[row_idx]
                    f.write(f"{(1, row_idx, row.data_path, 'inv', str(e))}\n")
                f.flush()
                continue

            for ev in group_entries: 
                row_idx, st_in = ev
                row = group.loc[row_idx]
                st_out, row_log = remove_resp(row, st_in, group_inv, prefilt())
                for line in row_log: f.write(f"{line}\n")
                if len(st_out) > 0: buffer_out.put((row_idx, st_out, 15)) # only put response removed traces into the buffer
            f.flush()
            #-^^^^^^- CODE HERE -^^^^^^-#

    for _ in range(nds): buffer_out.put(None) # Signal that process is done
    
    print(f'    >> Response removal done :: {os.getpid()}')
    mlog.write(f">>  {pd.Timestamp.now()}    RESPONSE_REMOVAL: complete   {os.getpid()}\n")
    mlog.flush()

def process_rot(buffer_in:mp.Queue, buffer_outs:mp.Queue, buffer_outf:mp.Queue,
                nus:int, ndss:int, ndsf:int, 
                f, mlog):
    """
    Rotates traces in RT, but if rotation fails, put the unrotated traces into the write buffer 
    to be written out as is. The item in should be a three-component stream.
    
    :type buffer_in: multiprocessing.queues.Queue
    :param buffer_in: Input buffer containing streams of traces to rotate
    :type buffer_outs: multiprocessing.queues.Queue
    :param buffer_outs: Output buffer containing streams of rotated traces to resample
    :type buffer_outf: multiprocessing.queues.Queue
    :param buffer_outf: Output buffer containing streams of unrotated traces to write

    :type nus: int
    :param nus: Number of upstream processes to wait for before exiting
    :type ndss: int
    :param ndss: Number of None values required to signal completion to downstream processes
    :type ndsf: int
    :param ndsf: Number of None values required to signal completion to downstream processes
    
    :type f: <class '_io.TextIOWrapper'>
    :param f: Log file
    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """
    
    print('Rotation: running')
    mlog.write(f">   {pd.Timestamp.now()}    ROTATION: running            {os.getpid()}\n")
    mlog.flush()

    none_count = 0
    while True:
        item = buffer_in.get()
        if item is None:
            none_count += 1
            if none_count == nus: break
        
        else: 
            #-vvvvvv- CODE HERE -vvvvvv-#
            row_idx, st_in, _ = item
        
            try: 
                st_out = rotate_rt(st_in) # rotate
                # log success
                log = (0, row_idx, 'rotate', 'success')
                f.write(f"{log}\n")
                f.flush()
                # put rotated traces into the next buffer to pass in to resampling
                buffer_outs.put((row_idx, st_out, 16)) # 16 = rotated

            except Exception as e:
                # log failure
                log = (1, row_idx, 'rotate', str(e))
                f.write(f"{log}\n")
                f.flush()
                # put unrotated traces to write out
                buffer_outf.put((row_idx, st_in, 15)) # 15 = response removed

            finally: f.flush()
            #-^^^^^^- CODE HERE -^^^^^^-#

    # Signal that process is done
    for _ in range(ndss): buffer_outs.put(None)
    for _ in range(ndsf): buffer_outf.put(None)

    print(f'    >> Rotation done :: {os.getpid()}')
    mlog.write(f">>  {pd.Timestamp.now()}    ROTATION: complete           {os.getpid()}\n")
    mlog.flush()

def process_res(buffer_in:mp.Queue, buffer_out:mp.Queue, 
                nus:int, nds:int, 
                f, mlog):
    """
    Resamples traces to 1 Hz and writes to write buffer. The item in should
    be a three-component stream.

    :type buffer_in: multiprocessing.queues.Queue
    :param buffer_in: Input buffer containing streams of traces to resample
    :type buffer_out: multiprocessing.queues.Queue
    :param buffer_out: Output buffer containing streams of traces to compute first P arrival

    :type nus: int
    :param nus: Number of upstream processes to wait for before exiting
    :type nds: int
    :param nds: Number of None values required to signal completion to downstream processes

    :type f: <class '_io.TextIOWrapper'>
    :param f: Log file
    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """

    print(f'Resampling: running :: {os.getpid()}')
    mlog.write(f">   {pd.Timestamp.now()}    RESAMPLING: running          {os.getpid()}\n")
    mlog.flush()

    none_count = 0
    while True:
        item = buffer_in.get()
        if item is None:
            none_count += 1
            if none_count == nus: break
        
        else:
            #-vvvvvv- CODE HERE -vvvvvv-#
            row_idx, st_in, _ = item

            sr = st_in[0].stats.sampling_rate
            try: 
                if sr not in (1, 1.0): st_in.resample(1) # resample to 1 Hz
                # log success
                log = (0, row_idx, 'resample', 'success')
                f.write(f"{log}\n")
                buffer_out.put((row_idx, st_in, 17)) # 17 = resampled
            
            except Exception as e:
                # log failure
                log = (1, row_idx, 'resample', str(e))
                f.write(f"{log}\n")
                buffer_out.put((row_idx, st_in, 16)) # 16 = rotated
            
            finally: f.flush()
            #-^^^^^^- CODE HERE -^^^^^^-#

    for _ in range(nds): buffer_out.put(None) # Signal that process is done

    print(f'    >> Resampling done :: {os.getpid()}')
    mlog.write(f">>  {pd.Timestamp.now()}    RESAMPLING: complete         {os.getpid()}\n")
    mlog.flush()

def process_taup(ttmodel:TauPyModel, evst:pd.DataFrame,
                 buffer_in:mp.Queue, buffer_out:mp.Queue,
                 nus:int, nds:int,
                 f, mlog):
    """
    Computes first P arrival for each trace. The item in should be a three-component stream. 
    
    :type ttmodel: obspy.taup.TauPyModel
    :param ttmodel: TauPyModel object for computing travel times, e.g. ak135 or iasp91
    :type evst: pandas.DataFrame
    :param evst: Event-station master dataframe for metadata

    :type buffer_in: multiprocessing.queues.Queue
    :param buffer_in: Input buffer containing streams of traces to compute first P arrival
    :type buffer_out: multiprocessing.queues.Queue
    :param buffer_out: Output buffer containing streams of preprocessed traces to write

    :type nus: int
    :param nus: Number of upstream processes to wait for before exiting
    :type nds: int
    :param nds: Number of None values required to signal completion to downstream processes

    :type f: <class '_io.TextIOWrapper'>
    :param f: Log file
    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """

    print(f'TauP: running :: {os.getpid()}')
    mlog.write(f">   {pd.Timestamp.now()}    TAUP: running                {os.getpid()}\n")
    mlog.flush()

    none_count = 0
    while True:
        item = buffer_in.get()
        if item is None:
            none_count += 1
            if none_count == nus: break
        
        else:
            #-vvvvvv- CODE HERE -vvvvvv-#
            row_idx, st_in, status = item
            data_path = evst.loc[row_idx, 'data_path']
            
            try:
                evdp = st_in[0].stats.sac.evdp # event depth in km
                epideg = st_in[0].stats.sac.gcarc # epicentral distance in degrees

                try:
                    tt = ttmodel.get_travel_times(source_depth_in_km=evdp, 
                                                  distance_in_degree=epideg, 
                                                  phase_list=['P'])
                    
                    if len(tt) == 0: log = (1, row_idx, 'taup', 'no_solution')
                    
                    else: 
                        out_text = ff.FortranRecordWriter('F8.2').write([round(tt[0].time, 2)]) + '  \n'
                        with open(f"{data_path}/firstP", 'w') as g: g.write(out_text)
                        log = (0, row_idx, 'taup', 'success')
                        buffer_out.put((row_idx, st_in, status))
                
                except Exception as e: log = (1, row_idx, 'taup', str(e))

            except Exception as e: log = (1, row_idx, 'metadata', str(e))
            
            finally: 
                f.write(f"{log}\n")
                f.flush()
                if log[0] == 1: buffer_out.put((row_idx, st_in, 14)) # 14 = taup failure
            #-^^^^^^- CODE HERE -^^^^^^-#

    for _ in range(nds): buffer_out.put(None) # Signal that process is done

    print(f'    >> TauP done :: {os.getpid()}')
    mlog.write(f">>  {pd.Timestamp.now()}    TAUP: complete               {os.getpid()}\n")
    mlog.flush()

# define end of script functions
def dump_status(buffer_in:mp.Queue,
                nus:int,
                mlog):
    """


    :type buffer_in: multiprocessing.queues.Queue
    :param buffer_in: Input buffer containing status updates for each row

    :type nus: int
    :param nus: Number of upstream processes to wait for before exiting

    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """

    print('Dumping status...')
    mlog.write(f">   {pd.Timestamp.now()}    DUMP_STATUS: running\n")
    mlog.flush()

    status_tuples = []
    
    none_count = 0    
    while True:
        item = buffer_in.get()
        if item is None:
            none_count += 1
            if none_count == nus: break
        else:
            #-vvvvvv- CODE HERE -vvvvvv-#
            row_idx, status = item
            status_tuples.append((row_idx, status))
            #-^^^^^^- CODE HERE -^^^^^^-#

    print('Dumping complete')
    mlog.write(f">>  {pd.Timestamp.now()}    DUMP_STATUS: complete\n")
    mlog.flush()
    
    return status_tuples

def update_evst(status_tuples:list, evst:pd.DataFrame,
                mlog):
    """ 
    Updates the evst dataframe with the processing status from each step

    :type status_tuples: list
    :param status_tuples: List containing tuples of (row.Index, status)
    :type evst: pandas.DataFrame
    :param evst: Event-station master dataframe
    
    :type mlog: <class '_io.TextIOWrapper'>
    :param mlog: Master log file for logging each worker's progress
    """

    print('Updating evst...')
    mlog.write(f">   {pd.Timestamp.now()}    UPDATE_EVST: running\n")
    mlog.flush()

    status_df = pd.DataFrame(status_tuples, columns=['row_idx', 'status'])
    taup_indices = status_df[status_df['status'] == 14].index.tolist() # get indices of rows with taup failure
    # keep rows that are not taup failures
    mask = ~status_df['row_idx'].isin(status_df.loc[taup_indices, 'row_idx'].tolist())
    keep = status_df.loc[mask]
    # if there are duplicates in row_idx (there shouldn't), choose the highest
    if keep.duplicated(subset='row_idx').sum() != 0: 
        keep = keep.groupby('row_idx', as_index=False).agg({'status': 'max'})
    # now these are the row indices to update
    update_idx = pd.concat([status_df.loc[taup_indices], keep]).sort_values('row_idx').reset_index(drop=True)
    
    evst.loc[update_idx['row_idx'], 'status'] = update_idx['status'].values

    print('evst update complete')
    mlog.write(f">>  {pd.Timestamp.now()}    UPDATE_EVST: complete\n")
    mlog.flush()

    return evst

# main script
if __name__ == '__main__':
    
    # define buffers
    buffer_in = mp.Queue(maxsize=no_evs) # for raw traces to be response removed
    buffer_rr = mp.Queue() # for response removed traces to be rotated
    buffer_rot = mp.Queue() # for rotated traces to be resampled
    buffer_res = mp.Queue() # to calculate first arrivals for resampled traces
    buffer_write = mp.Queue() # for traces to be written out
    buffer_status = mp.Queue() # for status updates from each step

    # define processes
    reader = mp.Process(target=read_traces, args=(evst,                                 # process-specific arguments
                                                  buffer_in,                            # output queue
                                                  njob_rr,                              # no. downstream processes
                                                  logs['prepro'], mlog))                # logging
    
    processor_rr = [mp.Process(target=process_rr, args=(client,                         # process-specific arguments
                                                        buffer_in, buffer_rr,           # input and output queues
                                                        1, njob_rot,                    # no. up/downstream processes
                                                        logs['rr'], mlog))              # logging
                    for _ in range(njob_rr)]
    
    processor_rot = [mp.Process(target=process_rot, args=(buffer_rr, buffer_rot, buffer_write,  # input and output queues
                                                          njob_rr, njob_res, 1,                 # no. up/downstream processes
                                                          logs['rot'], mlog))                   # logging
                     for _ in range(njob_rot)]
    
    processor_res = [mp.Process(target=process_res, args=(buffer_rot, buffer_res,       # input and output queues
                                                          njob_rot, njob_taup,          # no. up/downstream processes
                                                          logs['res'], mlog))           # logging
                     for _ in range(njob_res)]
    
    processor_taup = [mp.Process(target=process_taup, args=(ttmodel, evst,              # process-specific arguments
                                                            buffer_res, buffer_write,   # input and output queues
                                                            njob_res, 1,                # no. up/downstream processes
                                                            logs['taup'], mlog))        # logging
                      for _ in range(njob_taup)]
    
    writer = mp.Process(target=write_traces, args=(evst,                                # process-specific arguments
                                                   buffer_write, buffer_status,         # input and output queues
                                                   njob_rot+njob_taup, 1,               # no. up/downstream processes
                                                   logs['prepro'], mlog))               # logging
    
    # start all processes
    reader.start()
    for p in processor_rr: p.start()
    for p in processor_rot: p.start()
    for p in processor_res: p.start()
    for p in processor_taup: p.start()
    writer.start()

    print('All processes started.')

    status_tuples = dump_status(buffer_in=buffer_status, nus=1, mlog=mlog)                                            
    
    # block until all processes are done
    reader.join()
    for p in processor_rr: p.join()
    for p in processor_rot: p.join()
    for p in processor_res: p.join()
    for p in processor_taup: p.join()
    writer.join()

    print('All processes done. Updating evst...')

    # update evst with processing status
    if status_tuples: 
        updated_evst = update_evst(status_tuples, evst, mlog)
        updated_evst.to_csv(new_mpath, index=False)

    script_time_end = pd.Timestamp.today()
    print(f"Finished at {script_time_end}, duration: {(script_time_end - script_time_start).floor('s')}")
