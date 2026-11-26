import time, h5py
import numpy as np
import pandas as pd
from eulith.cc.xcorr_functions import *

def ev_xcorr(ev:pd.Series,
             intrstn_dist_km:float,
             maxlag_ref:tuple[float, float], maxlag_ref_idkm:tuple[float, float],
             fmax:float, nfmax:int,
             gamma:int,
             pvref_read:np.ndarray,
             guide_type:int,
             verbose):
    if verbose: print(f"    >>> Processing event: {ev.evid}")
    if ev.direction == 2: path1, path2 = ev.path2, ev.path1
    else: path1, path2 = ev.path1, ev.path2
    
    try: trs, npts, _, dt = preprocess_traces(path1, path2)
    except ValueError as e:
        raise ValueError(f"{e}")
    
    maxlag = calculate_maxlag(npts, intrstn_dist_km, maxlag_ref, maxlag_ref_idkm)

    xc, nxc, zl_ts = xc_hann(trs, npts, maxlag, dt)
    f_list, nf, dfmft = generate_xcf(nxc, dt, fmax, nfmax)
    omega_col, xc_conv_grid = xc_mft(f_list, xc, nxc, dt, gamma)
    
    reference, bounds = generate_pvref(f_list, pvref_read)
    if guide_type == 1: 
        idx_xc_max = (intrstn_dist_km / reference / dt).astype(int)
    elif guide_type == 2:
        idx_fw = calculate_idx_fw(zl_ts, dt, intrstn_dist_km, nxc, bounds)
        idx_xc_max = generate_sliding_window(xc_conv_grid, idx_fw)
    
    arg_list = xc_extract(omega_col, idx_xc_max, xc_conv_grid, nxc, dt, zl_ts, dfmft, nf, gammaw)

    return arg_list

def pair_xcorr(pair:str,
               wt:str,
               pairs_dir:str,
               write_dir:str,
               gamma:float, 
               gammaw_ref:tuple[float, float], gammaw_ref_idkm:tuple[float, float],
               maxlag_ref:tuple[float, float], maxlag_ref_idkm:tuple[float, float],
               fmax:float, nfmax:int,
               mineos_outpath:str,
               guide_type:int,
               verbose:bool=False,
               timeit:bool=False) -> int:

    if timeit: 
        pair_start_time = time.time()
    
    pair_evs_path = generate_paev_path(pairs_dir, pair, wt)
    pair_evs, intrstn_dist_km = read_pair_evs(pair_evs_path)
    pair_evs = pair_evs.iloc[:20]
    n_ev = len(pair_evs)
    write_dir = write_dir.rstrip("/")
    write_h5 = f"{write_dir}/{pair}.h5"

    gammaw_calc = lambda id_km: (gammaw_ref[1] - gammaw_ref[0]) / (gammaw_ref_idkm[1] - gammaw_ref_idkm[0]) * (id_km - gammaw_ref_idkm[0]) + gammaw_ref[0]
    gammaw = gammaw_calc(intrstn_dist_km)

    # reference
    pvref_read = np.loadtxt(mineos_outpath)

    if verbose:
        print(f"Pair: {pair};")
        print(f"Number of events: {len(pair_evs)};")
        print(f"Interstation distance: {intrstn_dist_km:.2f} km;")
        print(f"Gamma windowing factor: {gammaw:.2f};")
        print(f"Writing output to: {write_h5}.")

    with h5py.File(write_h5, "w") as hfile:

        for ev in pair_evs.itertuples():
            try: 
                f_list, arg_list, reference = ev_xcorr(ev, intrstn_dist_km, 
                                                       maxlag_ref, maxlag_ref_idkm,
                                                       fmax, nfmax,
                                                       gamma,
                                                       pvref_read,
                                                       guide_type,
                                                       verbose)
            except ValueError as e:
                print(f"    >>> {ev.evid}: {e}")
                continue
        
            hfile.create_dataset(f"{wt}/{ev.evid}", (3, len(f_list)), 
                                data=np.array([f_list, arg_list, reference]))
            
    if timeit: 
        pair_end_time = time.time()
        print(f">>> {pair}: Real time {pair_end_time - pair_start_time:.2f} s for {n_ev} events.")
    return n_ev