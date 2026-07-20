# Trace/Stream methods

import numpy as np
import pandas as pd
from obspy import Stream, Trace


def trim_trace(st:Stream):
    """
    Trim trace to the overlapping time window of all traces in the stream. 
    
    :type st: obspy.Stream
    :param st: stream to be trimmed
    :return: copy of the stream with the trimmed traces
    """
    stc = st.copy()
    starttime = max([tr.stats.starttime for tr in stc])
    endtime = min([tr.stats.endtime for tr in stc])
    stc.trim(starttime, endtime)
    return stc

def bp(st:Stream, freqmin:float=0.003, freqmax:float=0.1):
    """
    Bandpass filter for stream between freqmin and freqmax. Returns a copy of the stream with the filter applied.

    :param st: obspy Stream object
    :type freqmin: float
    :param freqmin: minimum frequency for bandpass filter
    :type freqmax: float
    :param freqmax: maximum frequency for bandpass filter
    """
    stc = st.copy()
    stc.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    return stc

### INDEV ###

def _check_monotonicity(tr:Trace, window:int):
    """
    Check for monotonic segments in a trace by calculating the difference between consecutive samples and checking for sign changes.

    :type tr: obspy.Trace
    :param tr: Trace object
    :type window: int
    :param window: window size for checking monotonicity
    :return: True if the trace has monotonic segments, False otherwise
    """
    dtr = np.diff(tr)
    sign_changes = np.diff(np.sign(dtr)) != 0
    change_indices = np.where(sign_changes)[0] + 1
    # pad with the start and end indices to capture the first and last segments
    change_indices = np.concatenate(([0], change_indices, [len(dtr)]))
    run_lengths = np.diff(change_indices)

    return np.sum(run_lengths > window) > 0

def _check_flat(tr:Trace, window:int, eps):
    """
    Check for flat segments in a trace by calculating the rolling standard deviation and comparing it to a tolerance.

    :type tr: obspy.Trace
    :param tr: Trace object
    :type window: int
    :param window: window size for rolling standard deviation
    :type eps: float/list
    :param eps: tolerance (per amplitude) for standard deviation. Can be a single value 
    or an array of the same length as tr.
    """
    tr_series = pd.Series(tr[:])
    deviation = tr_series.rolling(window, center=True).std()
    condition = (deviation < eps) & (tr_series.abs() > 0.05 * tr_series.max())

    return np.sum(condition) > 0

def check_clip(tr:Trace):
    """
    Checks for clipping. Wrapper function for _check_flat and _check_monotonicity. 
    Change window/tolerance values here
    
    :type tr: obspy.Trace
    :param tr: Trace object
    :return: Returns True if the trace is clipped, False otherwise.
    """
    eps = 1e-5 * pd.Series(tr[:]).abs()
    f_window = int(2. / np.sqrt(tr.stats.delta))
    flat = _check_flat(tr, f_window, eps)
    m_window = int(100 / tr.stats.delta)
    monotonic = _check_monotonicity(tr, m_window)

    return flat or monotonic