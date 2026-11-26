import os
from glob import glob
from pathlib import Path
from datetime import datetime as dt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eulith.core import replace_60

def pairs_stats(pairs_list_path:str,
                condensed:bool=True) -> pd.DataFrame:
    """
    Run quick statistics on the pairs.list file. Counts how many events are in each pair
    returning as a dataframe.

    :type pairs_list_path: str
    :param pairs_list_path: path to the pairs.list file
    :type condensed: bool
    :param condensed: if True, return a dataframe with columns 'pair' and 'n_ev' (max of n_Ra and n_Lo).
    If False, return all columns
    :rtype: pd.DataFrame
    :return: dataframe...
    """
    with open(pairs_list_path) as f:
        lines = f.readlines()
    print(f"Number of pairs: {len(lines)}")
    pair_df = pd.DataFrame([line.split()[:6] + line.split()[7::2] for line in lines], 
                           columns=['pair', 's1lon', 's1lat', 's2lon', 's2lat', 'intrstn_dist', 'n_Ra', 'n_Lo'])
    pair_df['n_Ra'] = pair_df['n_Ra'].astype(int)
    pair_df['n_Lo'] = pair_df['n_Lo'].astype(int)
    pair_df['s1lon'] = pair_df['s1lon'].astype(float)
    pair_df['s1lat'] = pair_df['s1lat'].astype(float)
    pair_df['s2lon'] = pair_df['s2lon'].astype(float)
    pair_df['s2lat'] = pair_df['s2lat'].astype(float)

    pair_df['n_ev'] = pair_df[['n_Ra', 'n_Lo']].max(axis=1) # to fix

    if condensed: 
        pair_df = pair_df[['pair', 'n_ev']]

    return pair_df

def generate_pvt_input(pairs_dir:str, 
                       wt:list=['R', 'L'],
                       out_dir:str=None,
                       pairs_list:list[str]=None,
                       list_name:str=None,
                       debug:bool=False):
    """
    Generate PVT input files for each wave type.

    >>> generate_pvt_input(pairs_dir="/space/jhyl3/script_tests/cc/cc_testing/pairs",
    ...                    wt=['R', 'L'],
    ...                    out_dir="/space/jhyl3/script_tests/cc/cc_testing/lists",
    ...                    list_name="test")
    
    :type pairs_dir: str
    :param pairs_dir: path to directory containing the data files
    :type wt: list[str]
    :param wt: list of component names, e.g. ['R', 'L'] for Rayleigh and Love
    :type out_dir: str
    :param out_dir: path to output directory
    :type pairs_list: list[str]
    :param pairs_list: list of pairs to include in the input file. If None, all pairs will be included
    :type list_name: str
    :param list_name: name of the output list file
    :type debug: bool
    :param debug: debug modeeeee (won't write to disk). note will only output love because stooooopid
    :raise ValueError: if wave type is not one of 'R' or 'L'
    """
    for type in wt:
        type = type.upper()
        if type == 'R': cmp = 'Ra'
        elif type == 'L': cmp = 'Lo'
        else: raise ValueError(f"Wave type {type} not recognised. Must be 'R' or 'L'")
        
        if not pairs_list:
            # glob to find all lists in pairs directory
            type_list = [f"{pairs_dir}/{path}" for path in glob(f"*/*/*{cmp}.lst")]
        else: 
            type_list:list[str] = []
            for pair in pairs_list:
                type_list.append(glob(f"{pairs_dir}/*/{pair}/*{cmp}.lst")[0])

        if not debug:
            # prevents FileNotFoundErrors
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            # control the output file name based on whether a list name is provided
            out_path = f"{out_dir}/pvtain_{cmp}.list"
            if list_name: out_path = f"{out_dir}/pvtain_{cmp}_{list_name}.list"

            with open(out_path, "w") as f:
                for path in type_list: f.write(f"{path}\n")
            print(f"Written {len(type_list)} {cmp} input files to {out_path}")
    if debug: return type_list
    return

def read_velfiles(path:str="vellove", df:bool=False):
    """
    Read in velocity files from vellove/ray file and return header, frequency, and velocity data.
    
    >>> read_velfiles(path="/space/jhyl3/script_tests/cc/test_dir_2/vellove")
    
    :type path: str
    :param path: path to vellove/ray file
    :type df: bool
    :param df: if True, return velocity blocks as a pandas DataFrame with frequency as index
    :rtype: tuple
    :return: (header_dict, frequency_list, velocity_array) or (header_dict, velocity_dataframe) if df=True
    :raise FileNotFoundError: if the file does not exist
    """
    if not os.path.exists(path): raise FileNotFoundError(f"File {path} does not exist")
    with open(path) as f: lines = f.readlines()
    zero_starting = [i for i, p 
                     in enumerate([s[:4].count(" ") for s in lines])
                     if p == 3]
    # parse blocks
    header = lines[0].strip().split()
    freq_block = [line.strip().split() for line in lines[1:zero_starting[0]]]
    # get velocity blocks - identify first item of chains of consecutive numbers
    idx_st = [] # index which marks start of each block
    first = True
    for p in zero_starting:
        if first: 
            last_value = p; idx_st.append(p)
            first = False
            continue
        if (p - last_value > 1): 
            last_value = p; idx_st.append(p)
        elif (p - last_value == 1):
            last_value = p
            continue
    vel_blocks = [lines[idx_st[i]:idx_st[i+1]] for i in range(len(idx_st)-1)]

    # parse header
    keys = ('fint', 'npts', 'modes_max') # frequency interval, number of points, flag (?)
    header[0] = float(header[0]); header[1] = int(header[1]); header[2] = int(header[2])
    hdict = {k: item for k, item in zip(keys, header)}

    # parse frequency block - collapse into single list and convert to floats
    freq_list = [float(freq) for line in freq_block for freq in line]

    # parse velocity blocks - return list of lists 
    vel_arr = []
    for block in vel_blocks:
        vel_list = []
        for line in block: 
            line = [float(val) for val in line.split()]
            vel_list.extend(line)
        vel_arr.append(vel_list)

    if df:
        vel_df = pd.DataFrame(vel_arr, columns=freq_list).T
        vel_df.index.name = 'freq'
        out = (hdict, vel_df)
    else: 
        out = (hdict, freq_list, vel_arr)
    return out

def _extract_stcodes(path:str) -> list[str]:
    """
    Extract station codes from a PVT cb path.
    >>> _extract_stcodes("pvt_t01/IV.FAGN_Y7.C04.Ra.CBdata.dat")
    ['IV.FAGN', 'Y7.C04']

    :rtype: list[str]
    :return: two-element list of station codes
    """
    return sorted(path.split('/')[-1].rsplit('.', 3)[0].split('_'))

def parse_pvtout(path:str, get_stalst:bool=True,
                 verbose:bool=False, debug:bool=False):
    """
    Parse pvt.dat files found in pairs folders. The start of the event blocks are marked
    by the station pair, e.g. 'LATE FAGN'. Because file is specific to the pair, this is
    either 'LATE FAGN' or 'FAGN LATE'. Known issues with getting correct event ID from
    event time. 

    :type path: str
    :param path: path to rbsel.dat file
    :type get_stalst: bool
    :param get_stalst: if True, return the station pair [sta1, sta2] from each block
    :param verbose: if True, print verbose output
    :type debug: bool
    :param debug: if True, return event blocks
    :raise FileNotFoundError: if the file does not exist
    :raise ValueError: if the file is empty
    """
    if not os.path.exists(path): raise FileNotFoundError(f"File {path} does not exist")
    with open(path) as f: lines = f.readlines()
    lines = [line.strip() for line in lines]
    if len(lines) == 0: raise ValueError(f"File {path} is empty")
    stalst = None
    # slice file into event blocks
    # identify start of event block by one of the stations (appearing in all event blocks)
    identifier = lines[0].split(' ')[0]
    ev_split_idx = [i for i, line in enumerate(lines) if identifier in line] + [len(lines)]
    ev_blocks = [lines[ev_split_idx[i]:ev_split_idx[i+1]] for i in range(len(ev_split_idx)-1)]

    pvt_lookup:list[dict] = []
    for ev_block in ev_blocks: 
        if get_stalst: 
            stalst = sorted(ev_block[0].split(' '))
            get_stalst = False # only needs to be retrieved once
        # parse event time to get event ID
        ev_time_str, seconds = ev_block[2].strip().rsplit(' ', 1)
        ev_dt_split = ev_block[2].strip().split(' ')
        ev_date_str = '-'.join(ev_dt_split[:3])
        seconds = str(round(float(seconds))) # doesn't work always
        ev_time_split = ev_dt_split[3:5] + [seconds]
        ev_time_str = ':'.join(ev_time_split)
        ev_dt_str = f"{ev_date_str}T{ev_time_str}"
        ev_dt_str = replace_60(ev_dt_str)
        ev_dt = dt.strptime(ev_dt_str, '%Y-%m-%dT%H:%M:%S')
        evid = dt.strftime(ev_dt, '%y%m%d_%H%M%S')
        npts = int(ev_block[11].split()[-1]) # number of data points
        data = [float(value) for value in [line.strip() for line in ev_block[12:]]]
        freq = data[:npts]
        vel = data[npts:]
        data_out = [freq, vel]
        ev_lookup = {'evid': evid, 'data': data_out}
        pvt_lookup.append(ev_lookup)    
        if verbose: print(f"Event {evid}: parsed {npts} data points from {path.split('/')[-1]}")
    out:list[dict] = pvt_lookup
    if stalst: out = (stalst, pvt_lookup)
    if debug: out = tuple(list(out) + [ev_blocks])
    return out

def parse_rbsel(path:str, get_stalst:bool=True,
                verbose:bool=False, debug:bool=False):
    """
    Parse rbsel.dat files. The start of event blocks are marked by
    '> [evid] [sta1] [sta2] dist12:[dist]'
    and the end of event blocks are marked by '> > >'.
    
    :type path: str
    :param path: path to rbsel.dat file
    :type get_stalst: bool
    :param get_stalst: if True, return the station pair [sta1, sta2] from each block
    :param verbose: if True, print verbose output
    :type debug: bool
    :param debug: if True, return event blocks
    :raise FileNotFoundError: if the file does not exist
    """
    if not os.path.exists(path): raise FileNotFoundError(f"File {path} does not exist")
    with open(path) as f: lines = f.readlines()
    lines = [line.strip() for line in lines]
    stalst = None

    # slice the file into event blocks
    ev_start:list[int] = [0] # list of indices which mark the start of each event block
    for i, line in enumerate(lines):
        if (i < len(lines)-1): 
            if (line.startswith('> > >')) and (lines[i+1].startswith('>')): ev_start.append(i+1)
    ev_start.append(len(lines)) # end of last event block

    ev_blocks = [lines[ev_start[i]:ev_start[i+1]] for i in range(len(ev_start)-1)]

    rbsel_lookup:list[dict] = []
    for ev_block in ev_blocks:
        ev_lookup:dict = {}
        # ev_lookup dict will contain the event ID and the data array for each event block
        ev_info:list[list[str]] = [line.split() for line in ev_block[:4]]
        # parse ev_info: get the event ID and station list
        evid = ev_info[0][1]
        # print(ev_info)
        if get_stalst: # write stalst only once. station list, but also station lat/lon and ev lat/lon
            stalst:dict = {}
            stalst['stns'] = sorted(ev_info[0][2:4])
            stalst['s1lon'], stalst['s1lat'] = [float(x) for x in [ev_info[1][1].split(":")[1]] + [ev_info[1][2]]]
            stalst['s2lon'], stalst['s2lat'] = [float(x) for x in [ev_info[2][1].split(":")[1]] + [ev_info[2][2]]]
            # stalst['evlon'], stalst['evlat'] = [float(x) for x in [ev_info[3][1].split(":")[1]] + [ev_info[3][2]]]
            get_stalst = False # only needs to be retrieved once
        # data array for each event block. each line is '[freq / Hz] [vel / km/s]'
        ev_data = [line for line in ev_block if not line.startswith('>')]
        data_out = np.array([[float(s) for s in line.split()] for line in ev_data]).T
        # shape of data_out is (2, npts) where npts is the number of data points for each event block
        # add to rbsel_lookup list of dicts
        ev_lookup = {'evid': evid, 'data': data_out}
        rbsel_lookup.append(ev_lookup)
        if verbose: print(f"Event {evid}: parsed {data_out.shape[1]} data points from {path.split('/')[-1]}")

    # return results
    out:list[dict] = rbsel_lookup
    if stalst: out = (stalst, rbsel_lookup)
    if debug: out = tuple(list(out) + [ev_blocks])
    return out

def parse_cbdata(path:str, 
                 verbose:bool=False, debug:bool=False):
    """
    Parse c(b)data files

    :type path: str
    :param path: path to c(b)data file
    :type verbose: bool
    :param verbose: if True, print verbose output
    :type debug: bool
    :param debug: if True, return blocks
    """
    with open(path) as f: lines = f.readlines()
    lines = [line.strip() for line in lines]
    ev_start = [0] + [i+1 for i, line in enumerate(lines) if line.startswith('> > > > >')]
    ev_blocks = [lines[ev_start[i]:ev_start[i+1]] for i in range(len(ev_start)-1)]

    cb_lookup:list[dict] = []
    for i, ev_block in enumerate(ev_blocks):
        ev_info = ev_block[-1].split('>')[-1].strip()
        ev_id = ev_info.split()[0]
        ev_block_data = ev_block[1:-1]
        # separate into branches
        branch_start = [0] + [i for i, line in enumerate(ev_block_data) if line.startswith('> > >')] 
        branch_blocks = [ev_block_data[branch_start[i]+1:branch_start[i+1]] for i in range(len(branch_start)-1)]
        n_br = len(branch_blocks) # number of branches calculated
        data_blocks = []
        # put data into numpy arrays for each branch for easier reading
        for branch_block in branch_blocks:
            if not branch_block:
                data_blocks.append(np.empty((2, 0)))
                continue
            block_str = '\n'.join(branch_block)
            # Parse entire block text directly into a 2D float array
            branch_data = np.fromstring(block_str, dtype=float, sep=' ').reshape(-1, 2).T
            data_blocks.append(branch_data)
        ev_lookup = {'evid': ev_id, 'n_br': n_br, 'data': data_blocks}
        cb_lookup.append(ev_lookup)
        if verbose: print(f"{i:<4}Event {ev_id}: {n_br} branches")

    out:list[dict] = cb_lookup
    if debug: out = (cb_lookup, ev_blocks)
    return out

def parse_pdata(path:str) -> list[dict]:
    """
    
    :type path: str
    :param path: path to pdata file
    """
    with open(path) as f: lines = f.readlines()
    block_indices = [0] + [(i+1) for i, s in enumerate(lines) if '> > > > >' in s]
    blocks = [lines[block_indices[i]:block_indices[i+1]] for i in range(len(block_indices)-1)]
    blocks_list:list[dict] = []
    for block in blocks:
        block_dict:dict = {}
        block_dict['evid'] = block[-1].strip('> > > > >').split(' ')[0]
        block_data = block[1:-1]
        branch_indices = [0] + [(i+1) for i, s in enumerate(block_data) if '> > >' in s]
        branches = [block_data[branch_indices[i]:branch_indices[i+1]] for i in range(len(branch_indices)-1)]
        block_dict['n_br'] = len(branches)
        block_dict_list = []
        for branch in branches:
            branch_data = np.array([[float(x) for x in s.strip().split(' ')] for s in branch[0:-1]]).T
            block_dict_list.append(branch_data)
        block_dict['data'] = block_dict_list
        blocks_list.append(block_dict)

    return blocks_list

def plot_branches(cb_lookup:list[dict], evidx:int=None, evid:str=None,
                  plot_select:bool=False, 
                  pvt_lookup:list[dict]=None, rbsel_lookup:list[dict]=None,
                  verbose:bool=False):
    """
    Plot phase velocity 2npi branches for a given event in the cb_lookup.

    :type cb_lookup: list[dict]
    :param cb_lookup: list of dictionaries containing C matrix from CB.dat file
    :type evidx: int
    :param evidx: index of event in cb_lookup to plot
    :type evid: str
    :param evid: event ID of event in cb_lookup to plot
    :type plot_select: bool
    :param plot_select: if True, plot the selection from pvt_lookup or rbsel_lookup
    :type pvt_lookup: list[dict]
    :param pvt_lookup: list of dictionaries containing selection PV data from the pvt file
    :type rbsel_lookup: list[dict]
    :param rbsel_lookup: list of dictionaries containing selection PV data from the rbsel file
    :type verbose: bool
    :param verbose: if True, print verbose output

    :raise ValueError: if none of cb, pvt or rbsel provided
    """
    plot_label = False
    cmap = plt.get_cmap('turbo')

    cb_evids = [ev['evid'] for ev in cb_lookup] # events in cb_lookup

    if (not evid and not evidx) & (plot_select):
        if (not pvt_lookup and not rbsel_lookup): raise ValueError("pvt_lookup or rbsel_lookup must be provided if plot_select is True")
        if pvt_lookup: sel_lookup = pvt_lookup
        if rbsel_lookup: sel_lookup = rbsel_lookup[1]
        sel_evids = [pev['evid'] for pev in sel_lookup]
        common_evids = list(set(sel_evids) & set(cb_evids))
        n_ev = len(common_evids)
    elif (evid is not None) | (evidx is not None): n_ev = 1

    for j in range(n_ev):
        if evid is not None: ev = cb_lookup[cb_evids.index(evid)]
        elif evidx is not None: ev = cb_lookup[evidx]
        else: ev = cb_lookup[cb_evids.index(common_evids[j])]

        if verbose: print(f"Plotting event {ev['evid']}")

        n_br = ev['n_br']
        if verbose: print(f"Event {ev['evid']} has {ev['n_br']} branches")
        data_blocks = ev['data']
        
        for i in range(n_br):
            pl = f"{i}" if plot_label else None
            plt.scatter(*data_blocks[i], marker='+', color=cmap(i/15), alpha=0.4, label=pl)

        if plot_select: 
            pvt_ev_lst = [pev for pev in sel_lookup if pev['evid'] == ev['evid']]
            if len(pvt_ev_lst) == 0: raise ValueError(f"Event {ev['evid']} not found in sel_lookup")
            pvt_ev = pvt_ev_lst[0]
            plt.scatter(*pvt_ev['data'], marker='x', color='blue', s=25, alpha=0.8, label='selection')

        plt.legend()
        plt.xscale('log'); plt.ylim(2, 7); plt.xlim(5e-3, 2e-1); plt.show()
    return

def plot_rbsel(parse_obj:list, plot_label:bool=False, 
               save_fig_dir:str=None):
    """
    Plot phase velocity curves for each event in the rbsel lookup.
    
    Usage: 
    >>> path = glob("pvt_t01/*rbsel.dat")[0]
    >>> plot_rbsel(parse_rbsel(path), plot_label=True)

    :type parse_obj: tuple
    :param parse_obj: output from parse_rbsel function, (stalst, rbsel_lookup). Must pass debug=False
    :type plot_label: bool
    :param plot_label: if True, add legend to plot
    :type save_fig_dir: str
    :param save_fig_dir: if provided, save the figure to this directory. Must exist
    :raise ValueError: if save_fig_dir does not exist
    """
    stalst, rbsel_lookup = parse_obj
    cmap = plt.get_cmap('turbo')
    n_ev = len(rbsel_lookup)
    for i, ev in enumerate(rbsel_lookup): 
        period = 1/ev['data'][0]
        plt.scatter(period, ev['data'][1], marker='+', 
                    color=cmap(i/n_ev), alpha=0.4, label=ev['evid'])
    plt.xscale('log'); plt.ylim(2, 7)
    # plt.xlim(5e-3, 2e-1)
    plt.xlim(5, 200)
    plt.xlabel(r'$T$ / s'); plt.ylabel(r'$v_p$ / $\mathrm{km s}^{-1}$'); 
    if plot_label: plt.legend()
    plt.title(f"Pair {', '.join(stalst)}, {n_ev} curves")
    if save_fig_dir: 
        if not os.path.exists(save_fig_dir): 
            raise ValueError(f"Directory {save_fig_dir} does not exist")
        plt.savefig(os.path.join(save_fig_dir, f"{'_'.join(stalst)}_pvt.png"), dpi=300, bbox_inches='tight')
    plt.show()
    return

def pvtrbs_stats(master_path:str,
                 plot:bool=False,
                 legend:bool=False,
                 verbose:bool=False):
    """
    Generate statistics for rbsel files in a given directory.

    :type master_path: str
    :param master_path: path to directory containing rbsel files
    :type plot: bool
    :param plot: if True, plot the dispersion curve for the station pair with the most events
    :type verbose: bool
    :param verbose: if True, print verbose output
    :rtype: pd.DataFrame
    :return: df with columns 'path' and 'n_events'
    """
    paths = glob(f"{master_path}/*rbsel.dat")
    if verbose: print(f"Found {len(paths)} rbsel files in {master_path}")
    track = []
    for i, path in enumerate(paths):
        _, rbsel_lookup = parse_rbsel(path, verbose=verbose)
        if verbose: print(f"{i:<4} {path.split('/')[-1]}: {len(rbsel_lookup)} events")
        track.append((path, len(rbsel_lookup)))

    nev_df = pd.DataFrame(track, columns=['path', 'n_events'])
    if verbose: print(nev_df['n_events'].value_counts())

    if plot:
        path = nev_df.loc[nev_df['n_events'].idxmax(), 'path']
        plot_rbsel(parse_rbsel(path, verbose=verbose), plot_label=legend)
    return nev_df

def generate_pvt_path(cb_path:str, base_path:str="/space/jhyl3/script_tests/cc/cc_testing"):
    """
    Generate the path to a PVT file based on a CB file path. Assumes that they are located
    in different directories - the CB in a custom directory and the PVT in a "pairs" directory. The station codes are extracted from the CB file name and used to construct the search path for the corresponding PVT file.
    >>> generate_pvt_path("pvt_t01/IV.FAGN_Y7.C04.Ra.CBdata.dat")
    '/space/jhyl3/script_tests/cc/cc_testing/pairs/*/IV.FAGN_Y7.C04/*PVT*'

    :type cb_path: str
    :param cb_path: path to CB file
    :type base_path: str
    :param base_path: base path for the search
    :rtype: str
    :return: path to the corresponding PVT file
    :raises ValueError: if the search does not return exactly one result
    """
    stcodes = _extract_stcodes(cb_path)
    pvt_search_path = f"pairs/*/{'_'.join(stcodes)}/*PVT*"
    search_path = f"{base_path}/{pvt_search_path}"
    search_results = glob(f"{base_path}/{pvt_search_path}")
    if len(search_results) != 1: raise ValueError(f"Expected 1 result for {search_path}, found {len(search_results)}")
    else: pvt_path = search_results[0]

    return pvt_path

def generate_rbsel_path(cb_path:str):
    """
    Generate the path to a rbsel file based on a CB file path. Assumes that they are located
    in the same directory. Probably not useful.
    """
    if "CBdata" in cb_path: out = cb_path.replace('CBdata', 'rbsel')
    elif "Cdata" in cb_path: out = cb_path.replace('Cdata', 'rbsel')
    return out

def build_c_matrix(c_data:list[dict], evid:str) -> tuple[str, pd.DataFrame]:
    """
    
    :return: tuple of event ID and C matrix as a pandas DataFrame
    """
    event = [s for s in c_data if s['evid'] == evid][0]
    # temporary hacking together a solution
    all_freq = []
    for dataset in event['data']: all_freq += list(dataset[0])
    freq = pd.Series(sorted(all_freq)).drop_duplicates()
    n_freq = len(freq)
    n_br = event['n_br']

    # build the C matrix
    c_matrix = pd.DataFrame(data=np.zeros((n_freq, n_br)), index=freq, columns=range(1,n_br+1))
    br_no = 0
    for branch in event['data']:
        br_no += 1
        br_f, br_c = branch
        for i, f in enumerate(br_f): c_matrix.loc[f, br_no] = br_c[i]
    c_matrix.replace(0, math.nan, inplace=True)
    return (evid, c_matrix)

def prelim_filter(c_matrix:pd.DataFrame, p_matrix:pd.DataFrame,
                  c_min:float=1., c_max:float=8.,
                  f_min:float=1e-3, f_max:float=0.2) -> tuple[pd.DataFrame]:
    """
    takes in c_matrix and p_matrix and filters them based on the provided min/max values for c and f.
    
    """
    c_matrix_ffilt = c_matrix[(c_matrix.index > f_min) & (c_matrix.index < f_max)]
    c_matrix_compare = c_matrix_ffilt.iloc[:,:-1]
    n_br = len(c_matrix_compare.columns)
    c_matrix_cfilt = c_matrix_ffilt.copy()
    c_matrix_cfilt.iloc[:,:-1] = c_matrix_compare[(c_matrix_compare > c_min) & (c_matrix_compare < c_max)]
    
    col_to_rm = []
    for colname in range(1,n_br+1):
        condition = (~np.isnan(c_matrix_cfilt.loc[:,colname])).sum() # if all nan then zero
        if condition == 0: col_to_rm.append(colname)
    c_matrix_filt = c_matrix_cfilt.drop(columns=col_to_rm)

    p_matrix_ffilt = p_matrix[(p_matrix.index > f_min) & (p_matrix.index < f_max)]
    p_matrix_filt = p_matrix_ffilt.drop(columns=col_to_rm)
    freq_filt = c_matrix_filt.index.to_numpy()
    
    return freq_filt, c_matrix_filt, p_matrix_filt

def plot_cp_matrix(evid:str, matrix:pd.DataFrame,
                   phase:bool=False) -> None:
    freq = matrix.index
    to_plot_matrix = matrix.drop(columns=['ref'])
    n_br = len(to_plot_matrix.columns)
    for i in range(n_br): plt.scatter(freq, to_plot_matrix.iloc[:,i], marker='x', s=5)
    plt.plot(freq, matrix['ref'], c='black', ls=':')
    plt.xscale('log')
    if not phase: plt.ylim(2,7); plt.xlim(0.005, 0.2)
    plt.title(f"{evid}")
    plt.show()

def parse_akref(path:str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read and parse ak135 reference file

    :type path: str
    :param path: path to ak135 reference file
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    :return: tuple of frequency, velocity (m/s) and velocity (km/s) numpy arrays
    """
    # read and parse reference file
    with open(path, 'r') as f: lines = f.readlines()
    lines = [[float(x) for x in line.split()] for line in lines]
    ref_data = np.array(lines)
    ref_freq = ref_data[:, 0]
    ref_c_k = ref_data[:, 1]
    ref_c = ref_c_k * 1000  # convert to m/s

    return ref_freq, ref_c, ref_c_k