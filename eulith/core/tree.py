# A series of tools to help with the directory structure of the data, including evst dataframes

import os, ast, shutil
from glob import glob
from pathlib import Path
from random import choice
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from eulith.core.utils import _copy_file, _create_symlink


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

def _copy_cmt(evst:pd.DataFrame, src:str, dst:str):
    """
    Minimal checks

    :type evst: pd.DataFrame
    :param evst: evst dataframe
    :type src: str
    :param src: source directory
    :type dst: str
    :param dst: destination directory
    :return: log of errors
    """
    log = []
    for path in evst['evpath'].unique():
        cmt_rel_path = os.path.join(path, f"cmt{path.strip('/').split('/')[-1]}")
        old_cmt_path = os.path.join(src, cmt_rel_path)
        new_cmt_path = os.path.join(dst, cmt_rel_path)
        if os.path.exists(new_cmt_path): continue
        Path(new_cmt_path).parent.mkdir(parents=True, exist_ok=True)
        try: shutil.copy2(old_cmt_path, new_cmt_path)
        except Exception as e: log.append((1, path, 'copy-cmt', str(e)))
    return log

def copy_tree(cmplst:list[str], 
              evst:pd.DataFrame,
              src:str, dst:str,
              symlink:bool=False):
    """
    Copies files from source to destination directory, maintaining directory structure.
    
    :type cmplst: list[str]
    :param cmplst: list of components, e.g. ['z', 'r', 't']
    :type evst: pd.DataFrame
    :param evst: event-station dataframe
    :type src: str
    :param src: str, source directory
    :type dst: str
    :param dst: str, destination directory
    :type symlink: bool
    :param symlink: if True, creates symlinks instead of copying files
    :return: tuple: list of new file paths and log of errors
    """
    log = []

    # copy files from src to dst, maintaining the same directory structure
    state = "symlink" if symlink else "copy"
    cmplst = [c.lower() for c in cmplst]
    new_file_paths = []

    for path in evst['data_path']:
        old_path = os.path.join(src, path)

        for cmp in cmplst: 
            file_path = glob(f"{old_path}/*.d{cmp}")

            if len(file_path) == 1:
                # compute new path
                old_file_path = file_path[0]
                rel_path = os.path.relpath(old_file_path, src)
                filename = os.path.relpath(rel_path, path)
                filename = filename.split('_')[-1]
                rel_path = os.path.join(path, filename)
                new_file_path = os.path.join(dst, rel_path)

                try:
                    if symlink: _create_symlink(old_file_path, new_file_path)
                    else: _copy_file(old_file_path, new_file_path)
                    new_file_paths.append(new_file_path)

                except Exception as e: log.append((1, path, f'{state}-{cmp}', str(e)))

            elif len(file_path) == 0:
                log.append((1, path, f'{state}-{cmp}', "file doesn't exist"))

            else: raise ValueError(f"Multiple files found for {path} with component {cmp}: {file_path}")

    # copy CMTs
    log.extend(_copy_cmt(evst=evst, src=src, dst=dst))
    
    return new_file_paths, log