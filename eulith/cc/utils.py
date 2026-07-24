import os
from glob import glob
from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_velfiles(path:str="vellove", df:bool=False):
    """
    Read in velocity files from vellove/ray file and return header, frequency, and velocity data.
    
    >>> read_velfiles(path="/space/jhyl3/script_tests/cc/test_dir_2/vellove")
    
    :type path: str
    :param path: path to vellove/ray file
    :type df: bool
    :param df: if True, return velocity blocks as a pandas DataFrame with frequency as index
    :return: tuple of (header_dict, frequency_list, velocity_array) or (header_dict, velocity_dataframe) if df=True
    """
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

def _extract_stcodes(path:str):
    return sorted(path.split('/')[-1].rsplit('.', 3)[0].split('_'))

def generate_pvt_path(cb_path:str, base_path:str="/space/jhyl3/script_tests/cc/cc_testing"):
    stcodes = _extract_stcodes(cb_path)
    pvt_search_path = f"pairs/*/{'_'.join(stcodes)}/*PVT*"
    search_path = f"{base_path}/{pvt_search_path}"
    search_results = glob(f"{base_path}/{pvt_search_path}")
    if len(search_results) != 1: raise ValueError(f"Expected 1 result for {search_path}, found {len(search_results)}")
    else: pvt_path = search_results[0]
    return pvt_path

def generate_rbsel_path(cb_path:str, base_path:str="/space/jhyl3/script_tests/cc/cc_testing"):
    if "CBdata" in cb_path: out = cb_path.replace('CBdata', 'rbsel')
    elif "Cdata" in cb_path: out = cb_path.replace('Cdata', 'rbsel')
    return out

def parse_pvtout(path:str,
                 verbose:bool=False, debug:bool=False):
    with open(path) as f: lines = f.readlines()
    lines = [line.strip() for line in lines]
    if len(lines) == 0: raise ValueError(f"File {path} is empty")
    identifier = lines[0].split(' ')[0]
    ev_split_idx = [i for i, line in enumerate(lines) if identifier in line] + [len(lines)]
    ev_split = [lines[ev_split_idx[i]:ev_split_idx[i+1]] for i in range(len(ev_split_idx)-1)]

    pvt_lookup = []
    for ev in ev_split: 
        ev_time_str, seconds = ev[2].strip().rsplit(' ', 1)
        ev_time_str += f" {round(float(seconds))}"
        ev_time = dt.strptime(ev_time_str, '%Y %m %d %H %M %S')
        evid = dt.strftime(ev_time, '%y%m%d_%H%M%S')
        npts = int(ev[11].split()[-1]) # number of data points
        data = [float(value) for value in [line.strip() for line in ev[12:]]]
        freq = data[:npts]
        vel = data[npts:]
        data_out = [freq, vel]
        ev_lookup = {'evid': evid, 'data': data_out}
        pvt_lookup.append(ev_lookup)    
        if verbose: print(f"Event {evid}: parsed {npts} data points from {path.split('/')[-1]}")
    out = pvt_lookup
    if debug: out = (pvt_lookup, ev_split)
    return out

def parse_rbsel(path:str, 
                verbose:bool=False, debug:bool=False):
    with open(path) as f: lines = f.readlines()
    lines = [line.strip() for line in lines]
    ev_start = [0]
    for i, line in enumerate(lines):
        if (i < len(lines)-1): 
            if (line.startswith('> > >')) and (lines[i+1].startswith('>')): ev_start.append(i+1)
    ev_start.append(len(lines))

    ev_blocks = [lines[ev_start[i]:ev_start[i+1]] for i in range(len(ev_start)-1)]

    rbsel_lookup = []
    get_stalst = True
    for ev_block in ev_blocks:
        ev_lookup = {}
        ev_info = ev_block[0].split()
        if get_stalst: 
            stalst = sorted(ev_info[2:4])
            get_stalst = False
        evid = ev_info[1]

        ev_data = [line for line in ev_block if not line.startswith('>')]
        data_out = np.array([[float(s) for s in line.split()] for line in ev_data]).T
        ev_lookup = {'evid': evid, 'data': data_out}
        rbsel_lookup.append(ev_lookup)
        if verbose: print(f"Event {evid}: parsed {data_out.shape[1]} data points from {path.split('/')[-1]}")
    out = (stalst, rbsel_lookup)
    if debug: out = (stalst, rbsel_lookup, ev_blocks)
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

    cb_lookup = []
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
            branch_data = np.array([[float(s) for s in line.split()] for line in branch_block]).T
            data_blocks.append(branch_data)
        ev_lookup = {'evid': ev_id, 'n_br': n_br, 'data': data_blocks}
        cb_lookup.append(ev_lookup)
        if verbose: print(f"{i:<4}Event {ev_id}: {n_br} branches")

    out = cb_lookup
    if debug: out = (cb_lookup, ev_blocks)
    return out

# plot phase velocity curves

def plot_branches(cb_lookup:list, evidx:int=None, evid:str=None,
                  plot_select:bool=False, 
                  pvt_lookup:list=None, rbsel_lookup:list=None,
                  verbose:bool=False, debug:bool=True):
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
            plt.scatter(*data_blocks[i], marker='+', c=cmap(i/15), alpha=0.4, label=pl)

        if plot_select: 
            pvt_ev_lst = [pev for pev in sel_lookup if pev['evid'] == ev['evid']]
            if len(pvt_ev_lst) == 0: raise ValueError(f"Event {ev['evid']} not found in sel_lookup")
            pvt_ev = pvt_ev_lst[0]
            plt.scatter(*pvt_ev['data'], marker='x', c='blue', s=25, alpha=0.8, label='selection')

        plt.legend()
        plt.xscale('log'); plt.ylim(2, 7); plt.xlim(5e-3, 2e-1); plt.show()
    return

def plot_rbsel(parse_obj:list, plot_label:bool=False):
    """
    Plot phase velocity curves for each event in the rbsel lookup.
    
    Usage: 
    >>> path = glob(pvt_t01/*rbsel.dat)[0]
    >>> plot_rbsel(parse_rbsel(path), plot_label=True)

    :type parse_obj: tuple
    :param parse_obj: output from parse_rbsel function, (stalst, rbsel_lookup). Must pass debug=False
    :type plot_label: bool
    :param plot_label: if True, add legend to plot
    """
    stalst, rbsel_lookup = parse_obj
    cmap = plt.get_cmap('turbo')
    n_ev = len(rbsel_lookup)
    for i, ev in enumerate(rbsel_lookup): 
        plt.scatter(*ev['data'], marker='+', 
                    c=cmap(i/n_ev), alpha=0.4, label=ev['evid'])
    plt.xscale('log'); plt.ylim(2, 7); plt.xlim(5e-3, 2e-1)
    plt.xlabel(r'$f$ / Hz'); plt.ylabel(r'$v_p$ / $\mathrm{km s}^{-1}$'); 
    if plot_label: plt.legend()
    plt.title(f"Pair {', '.join(stalst)}")
    plt.show()
    return