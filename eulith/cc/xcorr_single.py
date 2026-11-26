import h5py
import numpy as np
from eulith.cc.xcorr_routines import ev_xcorr, generate_paev_path, read_pair_evs

if __name__ == "__main__":
    wt = "Ra"
    pairs_dir = "/space/jhyl3/prod/cc/pairs_z10"

    # global
    mineos_outpath = "/space/jhyl3/script_tests/cc/mineos/test/pv_AK135_50"
    pvref_read = np.loadtxt(mineos_outpath)

    fmax = 0.5
    nfmax = 2400

    # calculate the maxlag subject to the interstation distance
    maxlag_ref = (1000, 2000)
    maxlag_ref_idkm = (400, 3000)

    # calculate the gamma windowing factor
    gammaw_ref = (20, 70)
    gammaw_ref_idkm = (400, 3000)

    gamma = 6

    pair = "FR.LABF_FR.VERF"

    pair_evs_path = generate_paev_path(pairs_dir, pair, wt)
    pair_evs, intrstn_dist_km, epidist_km = read_pair_evs(pair_evs_path)

    gammaw_calc = lambda id_km: (gammaw_ref[1] - gammaw_ref[0]) / (gammaw_ref_idkm[1] - gammaw_ref_idkm[0]) * (id_km - gammaw_ref_idkm[0]) + gammaw_ref[0]
    gammaw = gammaw_calc(intrstn_dist_km)

    evid = "240923_231420"
    ev = pair_evs[pair_evs.evid == evid].iloc[0]

    f_list, arg_list, reference = ev_xcorr(ev, 
                                           intrstn_dist_km,
                                           pvref_read,
                                           gamma, gammaw,
                                           maxlag_ref, maxlag_ref_idkm,
                                           fmax, nfmax,
                                           guide_type=1,
                                           verbose=False)

    hfile = f"/space/jhyl3/script_tests/cc/rewrite/pvtas_prod_out_t5/{pair}.h5"
    with h5py.File(hfile, "w") as h:
        h.create_dataset(f"{wt}/{ev.evid}", (3, len(f_list)), 
                            data=np.array([f_list, arg_list, reference]))
        dset = h[f"{wt}/{ev.evid}"]
        dset.attrs['flim'] = (f_list[0], f_list[-1])
        dset.attrs['nf'] = len(f_list)
        dset.attrs['gamma'] = gamma
        dset.attrs['gammaw'] = gammaw
        h.attrs['pair'] = pair
        h.attrs['intrstn_dist_km'] = intrstn_dist_km
        h.attrs['epidist_km'] = epidist_km
        