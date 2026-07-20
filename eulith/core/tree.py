# A series of tools to help with the directory structure of the data, including evst dataframes

import os, ast
from glob import glob
from random import choice
import numpy as np
import pandas as pd
from obspy import UTCDateTime

master_path = '/space/jhyl3/data_retrieval/europe'


def file_to_path(file_name:str, 
                 absolute:bool=False,
                 include_file:bool=True,
                 master_path=master_path):
    """
    Given a file name in the format 'evid_stn.chn', returns the path to the file in the data directory.

    :type file_name: str
    :param file_name: file name in the format 'evid_stn.chn'
    :type absolute: bool
    :param absolute: if True, returns the absolute path to the file, otherwise returns the relative path from the master_path
    :type include_file: bool
    :param include_file: if True, includes the file name in the returned path, otherwise returns the path to the directory containing the file
    :type master_path: str
    :param master_path: path to the master data directory
    :return: path to the file in the data directory
    """
    evid, stn_chn = file_name.rsplit('_', 1)
    stn = stn_chn.split('.')[0]
    year_short = evid[:2]
    month = evid[2:4]
    if int(year_short) < 70: year = '20' + year_short
    else: year = '19' + year_short
    evpath = f"{master_path}/data/{year}/{month}/{evid}/{stn[0]}"
    stcode = [s for s in os.listdir(evpath) if stn in s][0]
    path = f"{evpath}/{stcode}"
    if not absolute: path = f'data/{year}/{month}/{evid}/{stn[0]}/{stcode}'
    if include_file: path += f"/{file_name}"
    return path

def path_to_file(path, cmp='z', absolute=False):
    """
    Given a path to a evst directory return the file name in the format 'evid_stn.chn'
    for post-preprocessed files (hence .d{cmp} extension). If absolute is True, 
    returns the full path to the file.

    :type path: str
    :param path: path to the evst directory, e.g. 'data/2020/01/200101_000000/a/ab.abc'
    :type cmp: str
    :param cmp: component of the file, e.g. 'z'
    :type absolute: bool
    :param absolute: whether to return the absolute path
    :return: path to the file, e.g. '200101_000000_abc.dz'
    """
    split = path.split('/')
    filename = f"{split[-3]}_{split[-1].split('.')[-1]}.d{cmp}"
    return_path = filename if not absolute else f'{path}/{filename}'
    return return_path

def random_trace(path:str):
    """
    Returns a random trace file from the specified path (usually a data path).
    Please specify year and month otherwise it will be too slow.
    :type path: str
    :param path: path to the data directory
    :return: random trace file path
    """
    return choice(glob(f"{path}/**/**/**/*.?h?"))

### logging tools

def read_chunk_error(path):
    """
    Reads a chunk error log file and returns a list of tuples containing the station code and error message for each line in the log.
    
    :type path: str
    :param path: path to the chunk error log file
    :return: list of tuples containing the station code and error message
    """
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        return [ast.literal_eval(line.rsplit(', ', 2)[0] + ')') for line in lines]

### evst dataframe tools

def read_evst(path):
    """
    Given a .csv path, reads the evst csv file into the same dtypes as generation.

    :type path: str

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
    for k in keys: evst_import.loc[:, k] = [UTCDateTime(t) for t in evst_import[k]]

    evst_import.loc[:, 'bands'] = evst_import['bands'].apply(lambda x: ast.literal_eval(x) if (", " in x) \
                                                             else [s.strip("''") for s in x.strip('[]').split(' ')])
    evst_import.loc[:, 'bands'] = evst_import['bands'].apply(np.array)

    return evst_import

def merge_evsts(evst_list:list, status:str='max'):
    """
    Merges two evst dataframes, keeping the highest (if 'max') status code 

    :type evst_list: list
    :param evst_list: List of evst dataframes to merge
    :type status: str
    :param status: 'max' or 'min', determines whether to keep the highest or lowest status code when merging
    :return: Merged evst dataframe
    """
    columns = evst_list[0].columns
    agg_dict = {col: 'first' for col in columns}
    agg_dict['status'] = status
    evst_all = pd.concat(evst_list).groupby('data_path', as_index=False).agg(agg_dict)
    return evst_all

def generate_error_dict(df=None, node=None, year=None):
    """
    For a given evst dataframe, or a node, displays the unique error codes each station records
    NEEDS UPDATING. CONSIDER CHANGING STATUS CODES.
    :type df: pd.DataFrame
    :param df: evst dataframe
    :type node: str
    :param node: node name
    :type year: int
    :param year: year to filter errors by
    :return: dictionary mapping station codes to sets of error codes
    """
    if df: error_paths = [s.replace('.log', '_error.log') for s in df.loc[df['status'] != 0, 'log_path'].unique()]
    elif node: 
        error_paths = [os.path.join(f'logs/{node.lower()}', f) for f in os.listdir(f'logs/{node.lower()}') if f.endswith('_error.log')]
    error_dict = {}
    for path in error_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            if year: lines = [l for l in lines if f"UTCDateTime({year}" in l]
            if not lines: continue 
            stcode = lines[0].split(', ', 5)[1]
            error_dict[stcode] = tuple(set([s.split(', ', 5)[4] for s in lines]))
    return error_dict

def check_ami_status(path:str):
    """
    For a given path to an event-station directory, checks the status of the AMI processing\n
    Usage: evst.loc[mask, 'status'] = evst['data_path'].apply(check_ami_status)

    :param path: str, path to the event-station directory
    :return: status code (20, 21, or 22)
    """
    files = os.listdir(path)
    if 'paramp_all' in files:
        return 22
    elif len([s for s in files if s.startswith('NOD')]) == 0:
        return 21
    else: return 20