import time, os
import multiprocessing as mp
from pathlib import Path
import pandas as pd
from eulith import pair_xcorr


for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

njob_xc = 8

pairs_df = pd.read_csv("/space/jhyl3/prod/cc/pairs_df_50ev.csv").iloc[:16]
wt = "Ra"
pairs_dir = "/space/jhyl3/prod/cc/pairs_z10"
write_dir = "/space/jhyl3/script_tests/cc/rewrite/pvtas_prod_out_t4"
Path(write_dir).mkdir(parents=True, exist_ok=True)

# global
mineos_outpath = "/space/jhyl3/script_tests/cc/mineos/test/pv_AK135_50"

fmax = 0.5
nfmax = 2400

# calculate the maxlag subject to the interstation distance
maxlag_ref = (1000, 2000)
maxlag_ref_idkm = (400, 3000)

# calculate the gamma windowing factor
gammaw_ref = (20, 70)
gammaw_ref_idkm = (400, 3000)

gamma = 6

def read_pairs(buffer_out:mp.Queue, 
               njob_xc, 
               pairs_df:pd.DataFrame):
    """
    read pairs from pairs_df and put them into the buffer_out queue
    
    """
    print(f"    >> reader start :: {os.getpid()}")
    for pair in pairs_df.pair:
        buffer_out.put(pair)
    for _ in range(njob_xc):
        buffer_out.put(None)
    print(f"    >> reader done :: {os.getpid()}")
    return

def pair_xcorr_mp(buffer_in:mp.Queue,
                  nus:int,
                  wt:str, pairs_dir:str, write_dir:str,
                  gamma:int, 
                  gammaw_ref:tuple[float, float], gammaw_ref_idkm:tuple[float, float],
                  maxlag_ref:tuple[float, float], maxlag_ref_idkm:tuple[float, float],
                  fmax:float, nfmax:int,
                  mineos_outpath:str,
                  guide_type:int,
                  verbose:bool):
    """
    multiprocessing wrapper function for pair_xcorr
    
    """
    print(f"    >> xcorr start :: {os.getpid()}")
    none_count = 0
    pairs_count = 0
    ev_count = 0
    start_time_cpu = time.process_time()
    while True:
        item = buffer_in.get()
        if item is None:
            none_count += 1
            if none_count == nus: break
        
        else: 
            pair = item
            n_ev = pair_xcorr(pair, wt, pairs_dir, write_dir, 
                              gamma, gammaw_ref, gammaw_ref_idkm,
                              maxlag_ref, maxlag_ref_idkm, 
                              fmax, nfmax, 
                              mineos_outpath,
                              guide_type,
                              verbose,
                              timeit=True)
            ev_count += n_ev
            pairs_count += 1
    elapsed_time_cpu = time.process_time() - start_time_cpu
    print(f"    >> xcorr done :: {os.getpid()} :: processed {pairs_count} pairs with {ev_count} events in {elapsed_time_cpu:.2f} s")
        


buffer_pairs = mp.Queue(maxsize=10)

reader = mp.Process(target=read_pairs, args=(buffer_pairs, njob_xc, pairs_df))
processor_pair = [mp.Process(target=pair_xcorr_mp, args=(buffer_pairs,
                                                         1,
                                                         wt, pairs_dir, write_dir,
                                                         gamma, gammaw_ref, gammaw_ref_idkm,
                                                         maxlag_ref, maxlag_ref_idkm,
                                                         fmax, nfmax,
                                                         mineos_outpath,
                                                         1,
                                                         False))
                for _ in range(njob_xc)]

reader.start()
for p in processor_pair: p.start()

for p in processor_pair: p.join()
reader.join()