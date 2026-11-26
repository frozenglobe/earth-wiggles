import time
import numpy as np
import pandas as pd
from scipy.signal import windows
from scipy.signal import butter, filtfilt, detrend, correlate, fftconvolve
from scipy.fft import fft, fftfreq
from obspy import read, Trace, Stream
import h5py

wrap_phase = lambda phi: (phi + np.pi) % (2 * np.pi) - np.pi

def generate_paev_path(master_dir:str,
                       pair:str,
                       wt:str='Ra') -> str:
    """
    >>> master_dir = "/space/jhyl3/prod/cc/pairs_z10/"
    """
    master_dir = master_dir.rstrip('/')
    out = f"{master_dir}/{pair.split('_')[0]}/{pair}/{pair}.{wt}.lst"
    return out

def read_pair_evs(pair_evs_path:str) -> tuple[pd.DataFrame, float]:
    header_names = ['s1', 'nw1', 's2', 'nw2', 
                    'evid', 
                    's1_s2', 's2_s1', 's1_s2_dist', 
                    's1_ev', 's2_ev', 's1_ev_dist', 's2_ev_dist', 
                    's1_lat', 's1_lon', 
                    's2_lat', 's2_lon', 
                    'ev_lat', 'ev_lon', 
                    'direction', 'cmp', 'path1', 'path2']
    full = pd.read_csv(pair_evs_path, sep='\s+', header=None, names=header_names)
    intrstn_dist_km = full['s1_s2_dist'].iloc[0] * np.pi / 180 * 6371 # convert degrees to km
    condensed = full[['evid', 'direction', 'path1', 'path2']]

    return condensed, intrstn_dist_km

def butter3(lcp:float=800., 
            hcp:float=4., 
            dt:float=1.) -> tuple:
    """
    Returns the coefficients of a 3rd order Butterworth bandpass filter
    given the low and high corner periods (lcp, hcp) and the sampling interval (dt).
    Uses scipy.signal.butter
    """
    f_ny = 1/(2*dt) # Nyquist frequency
    b, a = butter(3, [f/f_ny for f in [1/lcp, 1/hcp]], btype='bandpass')
    return (b, a)

def _preprocess_trace(tr:np.ndarray, butter_filt:tuple) -> np.ndarray:
    """
    scipy routines - demean, bandpass filter, detrend
    
    :type tr: np.ndarray
    :param tr: input trace to be preprocessed; usually call Trace.data
    :type butter_filt: tuple
    :param butter_filt: tuple of (b, a) coefficients for the bandpass; generate using butter3()
    """
    tr -= np.mean(tr)
    tr = filtfilt(*butter_filt, tr)
    tr = detrend(tr, type='linear')
    return tr

def _preprocess_traces(st: Stream,
                       butter_filt:tuple,
                       dt:float=1.,
                       new_b:float=-120.) -> tuple[list, int, float]:
    """
    pass in a Stream of (two) traces. resample to dt. trim traces to common 
    start time and end time (determined by the shortest trace) and filter & 
    detrend using _preprocess_trace(). outputs a list of processed traces, the 
    trace length, and common start time (new_b; relative to event origin time)

    :type st: obspy.Stream
    :param st: input Stream of traces
    :type butter_filt: tuple
    :param butter_filt: tuple of (b, a) coefficients for the bandpass; generate using butter3()
    :type dt: float
    :param dt: sampling interval
    :type new_b: float
    :param new_b: common start time (relative to event origin time)

    :return tr_data: list of processed traces (np.ndarray)
    :return tr_len: length of the processed traces (all traces are the same length)
    :return new_b: common start time of the processed traces (relative to event origin time)
    """
    tr_data = []
    for tr in st:
        if tr.stats.delta not in [int(dt), float(dt)]: tr.resample(dt)
        b_loc = tr.stats.sac.b
        tr_data.append(tr.data[int((new_b - b_loc)/dt):])
    tr_len = min([len(tr) for tr in tr_data])
    tr_data = [tr[:tr_len] for tr in tr_data]
    tr_data = [_preprocess_trace(tr, butter_filt) for tr in tr_data]

    if tr_len < 1: 
        raise ValueError(f"Processed traces have length < 1.")

    return tr_data, tr_len, new_b

def preprocess_traces(path1:str, path2:str, 
                      dt:float=1., 
                      new_b:float=-120.) -> tuple[list, int, float, float]:
    """
    wrapper for _process_traces(). Reads in two SAC files and returns the 
    processed traces. 

    :type path1: str
    :param path1: path to first SAC file
    :type path2: str
    :param path2: path to second SAC file
    :type dt: float
    :param dt: sampling interval
    :type new_b: float
    :param new_b: common start time (relative to event origin time)
    """
    st = read(path1) + read(path2)
    butter_filt = butter3(dt=dt)
    trs, npts, b = _preprocess_traces(st, butter_filt, dt=dt, new_b=new_b)
    return trs, npts, b, dt

def calculate_maxlag(npts:int, 
                     intrstn_dist_km:float, 
                     maxlag_ref:list[float], 
                     maxlag_ref_idkm:list[float],
                     maxlag_min:int=500) -> int:
    """
    Calculate the maximum lag for cross-correlation based on the distance between the stations and reference values.

    :param npts: number of points in the traces
    :type npts: int
    :param intrstn_dist_km: distance between the stations in kilometers
    :type intrstn_dist_km: float
    :param maxlag_ref: reference maximum lag values
    :type maxlag_ref: list[float]
    :param maxlag_ref_idkm: reference distances in kilometers
    :type maxlag_ref_idkm: list[float]
    :param maxlag_min: minimum allowed maximum lag
    :type maxlag_min: int
    :return: calculated maximum lag
    :rtype: int
    """

    maxlag_calc = lambda id_km: round((maxlag_ref[1] - maxlag_ref[0]) / (maxlag_ref_idkm[1] - maxlag_ref_idkm[0]) * (id_km - maxlag_ref_idkm[0]) + maxlag_ref[0])
    maxlag = maxlag_calc(intrstn_dist_km)
    maxlag = np.ceil(np.max([maxlag, maxlag_min]))
    maxlag = int(np.min([maxlag, npts-1]))

    return maxlag

def xc_hann(trs:list|tuple, 
               npts:int,
               maxlag:int,
               dt:float=1.) -> tuple[np.ndarray, int, float, np.ndarray]:
    """
    Cross-correlate two traces and apply a Hann window; scipy routines

    :type trs: list | tuple
    :param trs: list or tuple of two traces to be cross-correlated
    :type npts: int
    :param npts: number of points in the traces
    :type dt: float
    :param dt: sampling interval in seconds
    :type maxlag: int
    :param maxlag: maximum lag for the cross-correlation

    :return xc: CCF
    :return nxc: number of points
    :return zl_ts: time of zero lag
    """
    tr1, tr2 = trs
    # perform the cross correlation and truncate to desired lag range
    xc = correlate(tr2, tr1)
    zl_pos = npts - 1 # position of zero lag
    start_idx = zl_pos - maxlag
    end_idx = zl_pos + maxlag + 1
    xc = xc[start_idx:end_idx]
    xc -= np.mean(xc) # remove mean
    nxc = len(xc) # length of CCF
    zl_ts = maxlag * dt # time of zero lag in seconds

    # xcorr temporal hann windowing - pre-frequency filter
    hw_n = int(np.min([maxlag/4, 300/dt]))
    xc[:hw_n] *= windows.hann(2*hw_n)[:hw_n]
    xc[-hw_n:] *= windows.hann(2*hw_n)[-hw_n:]

    return xc, nxc, zl_ts

def generate_xcf(nxc:int, dt:float, fmax:float, nfmax:int):
    """
    Generate the frequency list to sample the CCF. Returns the frequency spacing for the CCF/FFT

    :type nxc: int
    :param nxc: number of points in (length of) the cross-correlation function
    :type dt: float
    :param dt: sampling interval in seconds
    :type fmax: float
    :param fmax: maximum frequency to consider in Hz
    :type nfmax: int
    :param nfmax: maximum number of frequencies to consider
    """
    df = 1 / (nxc * dt)
    fmax = np.min([fmax, df*(nxc/2 + 1)])

    dfmft = int(np.ceil(fmax/df/nfmax))
    nf = int(fmax/df/dfmft)
    f_list = np.arange(1,nf+1) * df * dfmft

    return f_list, nf, dfmft

def generate_pvref(f_list:np.ndarray,
                   pvref_read:np.ndarray):
    """
    Usage: 
    >>> mineos_outpath = "/space/jhyl3/script_tests/cc/mineos/test/pv_AK135_50"
    # Generate a reference curve using MINEOS 

    :type mineos_outpath: str
    :param mineos_outpath: path to the output file from Mineos (AK135 model)
    :type f_list: np.ndarray
    :param f_list: array of frequencies for which to generate the reference bounds
    
    """
    # gpvr_start = time.process_time_ns()
    pvfreq = 1/pvref_read[::-1, 0]
    pvref = pvref_read[::-1, 1]/1000
    # pvref = np.column_stack((pvfreq, pvref_read[::-1, 1]/1000))

    f_intp = f_list[(f_list >= pvfreq[0]) & (f_list <= pvfreq[-1])]
    pvref_intp = np.interp(f_intp, pvfreq, pvref)
    
    # to extrapolate
    f_left = f_list[f_list < f_intp[0]]
    f_right = f_list[f_list > f_intp[-1]]

    buffer = 0.5 # in km/s
    to_concat_upper = [pvref_intp+buffer]
    to_concat = [pvref_intp]
    to_concat_lower = [pvref_intp-buffer]

    if f_right.size > 0: 
    # right linear
        fref_r = f_intp[-2:]
        pvref_r = pvref_intp[-2:]
        pv_right = []
        for inc in [buffer, 0, -buffer]:
            if inc > 0: grad_modifier = 1.4
            elif inc < 0: grad_modifier = 0.6
            else: grad_modifier = 1.0
            pvref_r_loc = pvref_r + inc
            right_grad_ref = (pvref_r_loc[1] - pvref_r_loc[0]) / (fref_r[1] - fref_r[0])
            pv_right.append(right_grad_ref*grad_modifier * (f_right - fref_r[0]) + pvref_r_loc[0])
        to_concat_upper.append(pv_right[0])
        to_concat.append(pv_right[1])
        to_concat_lower.append(pv_right[2])
        
    if f_left.size > 0:
        fref_l = f_intp[:2]
        pvref_l = pvref_intp[:2]
        pv_left = []
        for inc in [buffer, 0, -buffer]:
            if inc > 0: grad_modifier = 1.2
            elif inc < 0: grad_modifier = 0.4
            else: grad_modifier = 1.0
            pvref_l_loc = pvref_l + inc
            left_grad_ref = (pvref_l_loc[1] - pvref_l_loc[0]) / (np.log10(fref_l[1]) - np.log10(fref_l[0]))
            pv_left.append(left_grad_ref*grad_modifier * (np.log10(f_left) - np.log10(fref_l[0])) + pvref_l_loc[0])

        to_concat_upper.insert(0, pv_left[0])
        to_concat.insert(0, pv_left[1])
        to_concat_lower.insert(0, pv_left[2])

    upper_bound = np.concatenate(to_concat_upper)
    reference = np.concatenate(to_concat)
    lower_bound = np.concatenate(to_concat_lower)
    bounds = np.vstack((lower_bound, upper_bound))
    # print(f"refcurve cpu time: {(time.process_time_ns() - gpvr_start):} ns")
    return reference, bounds

def calculate_idx_fw(zl_ts:float, 
                     dt:float, 
                     intrstn_dist_km:float, 
                     nxc:int, 
                     bounds:tuple|np.ndarray) -> tuple[int, int]:
    """
    Usage:
    >>> bounds = (1.5, 99999.) # lower and upper bounds for the velocity in km/s
    >>> bounds = generate_pvref(f_list, mineos_outpath)[1] # use computed reference bounds
    
    """
    idx_fw_0 = int(np.floor(zl_ts/dt))
    mod_1 = np.floor(intrstn_dist_km / bounds[1] / dt)
    mod_2 = np.ceil(intrstn_dist_km / bounds[0] / dt)
    if isinstance(bounds, tuple):
        idx_fw_1 = idx_fw_0 + int(mod_1)
        idx_fw_2 = idx_fw_0 + int(mod_2)
    elif isinstance(bounds, np.ndarray):
        idx_fw_1 = idx_fw_0 + mod_1.astype(int)
        idx_fw_2 = idx_fw_0 + mod_2.astype(int)
    idx_fw_2 = [np.min([i, nxc]) for i in idx_fw_2] # make sure trace length not exceeded
    idx_fw = (idx_fw_1, idx_fw_2)

    return idx_fw

# def generate_sliding_window(xc_conv_grid:np.ndarray,
#                             idx_fw:tuple[int, int]) -> np.ndarray:
#     """
#     return list of window centres, same length as number of frequencies
#     """
#     idx_fw_1, idx_fw_2 = idx_fw
#     idx_xc_max = idx_fw_1 + np.argmax(np.abs(xc_conv_grid[:, idx_fw_1:idx_fw_2]), axis=1)
#     return idx_xc_max

def generate_sliding_window(xc_conv_grid:np.ndarray,
                            idx_fw_lists:tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    idx_xc_max = []
    for i, (idx_fw_1, idx_fw_2) in enumerate(zip(*idx_fw_lists)):
        idx_xc_max.append(idx_fw_1 + np.argmax(np.abs(xc_conv_grid[i, idx_fw_1:idx_fw_2])))
    return np.array(idx_xc_max)

def xc_mft(f_list:np.ndarray, 
           xc:np.ndarray, 
           nxc:int, 
           dt:float, 
           gamma:float) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform the multiple filter technique (MFT) on the CCF

    :type f_list: np.ndarray
    :param f_list: array of frequencies for which to perform the MFT
    :type xc: np.ndarray
    :param xc: cross-correlation function (CCF)
    :type nxc: int
    :param nxc: number of points in the CCF
    :type dt: float
    :param dt: sampling interval in seconds
    :type gamma: float
    :param gamma: parameter for the Gaussian filter in the MFT
    :return omega_col: array of angular frequencies corresponding to f_list
    :return xc_conv_grid: 2D array of convolved CCFs for each frequency in f_list. size is (len(f_list), nxc)
    """
    # mft_start = time.process_time_ns()
    
    f_col = f_list[:, None]
    # each row is now a frequency
    omega_col = 2 * np.pi * f_col
    alpha_col = gamma**2 * omega_col
    emin = 7
    t0_col = np.sqrt(emin * np.log(10) * 4 * alpha_col) / omega_col
    it0_col = (t0_col / dt / 2).astype(int) * 2 # ensures that the number of points in the filter is odd
    nfil_col = it0_col * 2 - 1 # length of filter
    tfil_grid = dt * (np.arange(1, nfil_col.max() + 1) - it0_col)
    yfil_grid = 0.5 * omega_col / np.sqrt(np.pi * alpha_col) * np.exp(-(omega_col * tfil_grid)**2 / (4*alpha_col))
    yfil_grid *= np.cos(omega_col * tfil_grid)

    col_indices = np.arange(nfil_col.max())[None, :]
    mask = col_indices < nfil_col
    yfil_grid = np.where(mask, yfil_grid, 0.0)

    xc_grid = np.broadcast_to(xc, (len(f_list), nxc))
    xc_conv_full = fftconvolve(xc_grid, yfil_grid, mode='full', axes=1)

    row_indices = np.arange(len(f_list))[:, None]
    col_slice_indices = it0_col + np.arange(nxc)[None, :]
    xc_conv_grid = xc_conv_full[row_indices, col_slice_indices]
    xc_conv_grid -= np.mean(xc_conv_grid, axis=1, keepdims=True)

    # print(f"mft cpu time: {(time.process_time_ns() - mft_start)} ns")

    return omega_col, xc_conv_grid

def xc_extract(omega_col:np.ndarray, 
               idx_xc_max:np.ndarray, 
               xc_conv_grid:np.ndarray, 
               nxc:int, 
               dt:float, 
               zl_ts:np.ndarray, 
               dfmft:float, 
               nf:int, 
               gammaw:float) -> np.ndarray:
    """
    Window the convolved CCFs and extract the phase information for each frequency

    :type omega_col: np.ndarray
    :param omega_col: array of angular frequencies corresponding to f_list
    :type idx_xc_max: np.ndarray
    :param idx_xc_max: array of indices of the maximum cross-correlation for each frequency
    :type xc_conv_grid: np.ndarray
    :param xc_conv_grid: 2D array of convolved CCFs for each frequency in f_list. size is (len(f_list), nxc)
    """
    # extract_time = time.process_time_ns()
    new_t_grid = (np.arange(nxc) - idx_xc_max[:, None]) * dt
    alphaw = gammaw**2 * omega_col
    xc_window_grid = xc_conv_grid * np.exp(-(new_t_grid**2 * omega_col**2) / (4 * alphaw))
    i_col = np.arange(1, nf+1)[:, None]
    n_grid = np.broadcast_to(np.arange(nxc), (nf, nxc))
    z_list = (np.cos(omega_col * zl_ts) + 1j * np.sin(omega_col * zl_ts)).T[0]
    fft_list = np.sum(xc_window_grid * np.exp(-1j * 2 * np.pi * (i_col*dfmft) * n_grid/nxc), axis=1)
    z_list *= fft_list
    arg_list = np.arctan2(z_list.imag, z_list.real)

    # print(f"extract cpu time: {(time.process_time_ns() - extract_time)} ns")

    return arg_list