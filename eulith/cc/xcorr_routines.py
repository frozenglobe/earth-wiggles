import time, h5py
import numpy as np
import pandas as pd
from eulith.cc.xcorr_functions import *

def ev_xcorr(ev:pd.Series,
             intrstn_dist_km:float,
             pvref_read:np.ndarray,
             gamma:int, gammaw:float,
             maxlag_ref:tuple[float, float], maxlag_ref_idkm:tuple[float, float],
             fmax:float, nfmax:int,
             guide_type:int,
             verbose:bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the cross-correlation for a single event

    :param ev: row from pair_evs dataframe
    :type ev: pd.Series
    :param intrstn_dist_km: interstation distance in km
    :type intrstn_dist_km: float
    :param pvref_read: reference phase velocity as array, period (s) in first column, velocity (m/s) in second column
    :type pvref_read: np.ndarray

    :param maxlag_ref: reference maxlag values
    :type maxlag_ref: tuple[float, float]
    :param maxlag_ref_idkm: reference interstation distance values for maxlag
    :type maxlag_ref_idkm: tuple[float, float]
    :param fmax: maximum frequency for cross-correlation
    :type fmax: float
    :param nfmax: number of frequency points for cross-correlation
    :type nfmax: int
    :param gamma: gamma parameter for frequency gaussian window
    :type gamma: int
    :param guide_type: type of guidance for cross-correlation, 1 for reference, 2 for sliding window
    :type guide_type: int

    :return: f_list, arg_list, reference
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    
    if verbose: print(f"    >>> Processing event: {ev.evid}")
    if ev.direction == 2: path1, path2 = ev.path2, ev.path1
    else: path1, path2 = ev.path1, ev.path2
    
    try: trs, npts, _, dt = preprocess_traces(path1, path2)
    except ValueError as e: raise ValueError(f"{e}")
    
    maxlag = calculate_maxlag(npts, intrstn_dist_km, maxlag_ref, maxlag_ref_idkm)

    xc, nxc, zl_ts = xc_hann(trs, npts, maxlag, dt)
    f_list, nf, dfmft = generate_xcf(nxc, dt, fmax, nfmax)
    omega_col, xc_conv_grid = xc_mft(f_list, xc, nxc, dt, gamma)
    
    reference, bounds = generate_pvref(f_list, pvref_read)
    if guide_type == 1: 
        idx_xc_max = (zl_ts / dt + intrstn_dist_km / reference / dt).astype(int)
    elif guide_type == 2:
        idx_fw = calculate_idx_fw(zl_ts, dt, intrstn_dist_km, nxc, bounds)
        idx_xc_max = generate_sliding_window(xc_conv_grid, idx_fw)
    
    arg_list = xc_extract(omega_col, idx_xc_max, xc_conv_grid, nxc, dt, zl_ts, dfmft, nf, gammaw)

    return f_list, arg_list, reference

def pair_xcorr(pair:str,
               wt:str,
               pairs_dir:str,
               write_dir:str,
               mineos_outpath:str,
               gamma:float, 
               gammaw_ref:tuple[float, float], gammaw_ref_idkm:tuple[float, float],
               maxlag_ref:tuple[float, float], maxlag_ref_idkm:tuple[float, float],
               fmax:float, nfmax:int,
               guide_type:int,
               verbose:bool=False,
               timeit:bool=False) -> int:

    """
    Perform the cross-correlation for all events in a station pair and write the results to an h5 file

    :param pair: station pair, e.g. "HL.ITM_IU.GNI"
    :type pair: str
    :param wt: wave type, either "Ra" or "Lo"
    :type wt: str
    :param pairs_dir: directory containing the pair event files
    :type pairs_dir: str
    :param write_dir: directory to write the output h5 file
    :type write_dir: str
    :param mineos_outpath: path to the reference phase velocity file
    :type mineos_outpath: str
    
    :param gamma: gamma parameter for frequency gaussian window
    :type gamma: int
    :param gammaw_ref: reference gamma windowing factor values
    :type gammaw_ref: tuple[float, float]
    :param maxlag_ref: reference maxlag values
    :type maxlag_ref: tuple[float, float]
    :param maxlag_ref_idkm: reference interstation distance values for maxlag
    :type maxlag_ref_idkm: tuple[float, float]
    :param fmax: maximum frequency for cross-correlation
    :type fmax: float
    :param nfmax: number of frequency points for cross-correlation
    :type nfmax: int
    :param guide_type: type of guidance for cross-correlation, 1 for reference, 2 for sliding window
    :type guide_type: int

    :param timeit: whether to time the function execution
    :type timeit: bool

    :return: number of events processed for the station pair
    :rtype: int
    """

    if timeit: 
        pair_start_time = time.time()
    
    pair_evs_path = generate_paev_path(pairs_dir, pair, wt)
    pair_evs, intrstn_dist_km, epidist_km = read_pair_evs(pair_evs_path)
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

    with h5py.File(write_h5, "a") as hfile:
        hfile.attrs['pair'] = pair
        hfile.attrs['intrstn_dist_km'] = intrstn_dist_km
        hfile.attrs['epidist_km'] = epidist_km

        for ev in pair_evs.itertuples():
            try: 
                f_list, arg_list, reference = ev_xcorr(ev=ev,
                                                       intrstn_dist_km=intrstn_dist_km, 
                                                       pvref_read=pvref_read,
                                                       gamma=gamma, gammaw=gammaw,
                                                       maxlag_ref=maxlag_ref, maxlag_ref_idkm=maxlag_ref_idkm,
                                                       fmax=fmax, nfmax=nfmax,
                                                       guide_type=guide_type,
                                                       verbose=verbose)
            except ValueError as e:
                print(f"    >>> {ev.evid}: {e}")
                continue
        
            hfile.create_dataset(f"{wt}/{ev.evid}", (3, len(f_list)), 
                                data=np.array([f_list, arg_list, reference]))
            dset = hfile[f"{wt}/{ev.evid}"]
            dset.attrs['flim'] = (f_list[0], f_list[-1])
            dset.attrs['nf'] = len(f_list)
            dset.attrs['gamma'] = gamma

            
    if timeit: 
        pair_end_time = time.time()
        print(f">>> {pair}: Real time {pair_end_time - pair_start_time:.2f} s for {n_ev} events.")
    return n_ev