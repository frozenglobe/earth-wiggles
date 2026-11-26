import os, argparse
from glob import glob
from pathlib import Path
from pandas import Timestamp

from eulith.core.utils import azbaz, azdif


def get_options():
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(prog='generate_pairs',
                                     description='make pairs from dataset')
    # paths, directories, etc. 
    parser.add_argument("-d", "--data_path", type=str,
                        default=f"{cwd}/data",
                        help="path to data directory")
    parser.add_argument("-o", "--output_path", type=str,
                        default=f"{cwd}/data.list",
                        help="path to output file")
    parser.add_argument("-p", "--pairs_dir", type=str,
                        default=f"{cwd}/pairs",
                        help="path to pairs directory")
    # pair options
    parser.add_argument("-z", "--azimuth", type=float,
                        default=7.,
                        help="maximum azimuth separation of gcps (of sta pair vs sta-ev)")
    parser.add_argument("-s", "--separation", type=str,
                        default="1,9999999",
                        help="min and max separation of stations in format [min,max]")
    # other
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=False,
                        help="enable verbose output")
    parser.add_argument("-r", "--read", type=str,
                        default=None,
                        help="read data list from file instead of querying data directory")
    args = parser.parse_args()
    # error handling
    if len(args.separation.split(",")) != 2: raise ValueError("Separation must be in format [min,max]")

    return args

def process_event(event:lt.Event, 
                  pairdist_min:float, pairdist_max:float, azimuth:float,
                  pairs_dict:dict,
                  pairs_dir:str,
                  verbose:bool=False) -> dict:
    """
    

    """
    nstns = event.nstations # number of stations
    evlat = event.lat # event latitude
    evlon = event.lon # event longitude
    unsorted_stns = event.stations # list of stations
    if nstns < 2: return pairs_dict
    if verbose: print(f"Event: {event.name}")

    # sort stations alphabetically
    sorted_stns = sorted(unsorted_stns, key=lambda x: (x.net, x.name))

    for i, s1 in enumerate(sorted_stns):
        s1_ev= azbaz(s1.lat, s1.lon, evlat, evlon)[0] # forward azimuth
        for s2 in sorted_stns[i+1:]:
            pair = f"{s1.stcode}_{s2.stcode}" # FIX
            # check for existing Rayleigh and Love
            both_ray = s1.Rayl & s2.Rayl
            both_love = s1.Love & s2.Love
            existing_fail = (not both_ray) & (not both_love)
            if existing_fail: continue
            # distance condition
            intrstn_dist = s1.sep(s2)
            dist_fail = (intrstn_dist == 0) | (intrstn_dist < pairdist_min) | (intrstn_dist > pairdist_max)
            if dist_fail: continue
            # azimuth condition
            s2_ev = azbaz(s2.lat, s2.lon, evlat, evlon)[0] # forward azimuth
            s1_s2, s2_s1 = azbaz(s1.lat, s1.lon, s2.lat, s2.lon)[:2]
            az_try = min(azdif(s1_ev, s1_s2, deg=True),
                         azdif(s2_ev, s2_s1, deg=True))
            az_accept = (az_try <= azimuth) # accept if azimuth difference is less than threshold
            if not az_accept: continue
            # compute station-event distances and directions
            s1_s2_dist = delta(s1.lat, s1.lon, s2.lat, s2.lon)
            s1_ev_dist = delta(s1.lat, s1.lon, evlat, evlon)
            s2_ev_dist = delta(s2.lat, s2.lon, evlat, evlon)
            if (s1_ev_dist > s2_ev_dist): direction = 2
            else: direction = 1

            # make pair directory
            Path(pairs_dir).mkdir(parents=True, exist_ok=True)
            pair_dir = _generate_pair_path(pairs_dir=pairs_dir, s1=s1, s2=s2)
            # write station information to 'stations' file
            if not os.path.exists(f"{pair_dir}/stations"):
                with open(f"{pair_dir}/stations", "w") as f:
                    f.write("{0:9.5f} {1:10.5f} {2:9.5f} {3:10.5f} {4:7.2f}\n"
                            .format(s1.lat, s1.lon, 
                                    s2.lat, s2.lon, 
                                    s1_s2_dist))

            if both_ray:
                wt_dir = f"{pair_dir}/Rayleigh" # e.g. pairs/S1/S1_S2/Rayleigh
                Path(wt_dir).mkdir(parents=True, exist_ok=True)
                with open(f"{wt_dir}/xseed.lst", "a") as f:
                    for sta in [s1, s2]:
                        f.write("{0:s}/{1:s}\n".format(sta.path.strip(), sta.dz.strip()))

                list_path = f"{pair_dir}/{pair}.Ra.lst" # e.g. pairs/S1/S1_S2/S1_S2.Ra.lst
                with open(list_path, "a") as f:
                    f.write("{0:5s} {1:2s} {2:5s} {3:2s}  {4:13s}  {5:7.3f} {6:7.3f} {7:7.3f} {8:7.3f}  {9:7.3f} {10:7.3f} {11:7.3f} {12:7.3f} {13:8.3f} {14:7.3f} {15:8.3f} {16:7.3f} {17:8.3f} {18:d}  R  {19:s}/{20:s}  {21:s}/{22:s}\n".format(
                        s1.name, s1.net, 
                        s2.name, s2.net,
                        event.name,
                        s1_s2, s2_s1, s1_s2_dist, 
                        s1_ev, s2_ev, s1_ev_dist, s2_ev_dist,
                        s1.lat, s1.lon, 
                        s2.lat, s2.lon, 
                        evlat, evlon, direction, 
                        s1.path.strip(), s1.dz.strip(),
                        s2.path.strip(), s2.dz.strip()))
                # add to pairs dict
                if pair not in pairs_dict:
                    pairs_dict[pair] = [s1.lon, s1.lat,
                                        s2.lon, s2.lat,
                                        intrstn_dist,
                                        1, 0] # number of Rayleigh and Love pairs
                else: pairs_dict[pair][5] += 1 # increment number of Rayleigh pairs

            if both_love: # duplicate of above with z -> t and Rayleigh -> Love
                wt_dir = f"{pair_dir}/Love" # e.g. pairs/S1/S1_S2/Love
                Path(wt_dir).mkdir(parents=True, exist_ok=True)
                with open(f"{wt_dir}/xseed.lst", "a") as f:
                    for sta in [s1, s2]:
                        f.write("{0:s}/{1:s}\n".format(sta.path.strip(), sta.dt.strip()))

                list_path = f"{pair_dir}/{pair}.Lo.lst" # e.g. pairs/S1/S1_S2/S1_S2.Lo.lst
                with open(list_path, "a") as f:
                    f.write("{0:5s} {1:2s} {2:5s} {3:2s}  {4:13s}  {5:7.3f} {6:7.3f} {7:7.3f} {8:7.3f}  {9:7.3f} {10:7.3f} {11:7.3f} {12:7.3f} {13:8.3f} {14:7.3f} {15:8.3f} {16:7.3f} {17:8.3f} {18:d}  T  {19:s}/{20:s}  {21:s}/{22:s}\n"
                            .format(s1.name, s1.net, 
                                    s2.name, s2.net,
                                    event.name,
                                    s1_s2, s2_s1, s1_s2_dist, 
                                    s1_ev, s2_ev, s1_ev_dist, s2_ev_dist,
                                    s1.lat, s1.lon, 
                                    s2.lat, s2.lon, 
                                    evlat, evlon, direction, 
                                    s1.path.strip(), s1.dt.strip(),
                                    s2.path.strip(), s2.dt.strip()))
                # add to pairs dict
                if pair not in pairs_dict:
                    pairs_dict[pair] = [s1.lon, s1.lat,
                                        s2.lon, s2.lat,
                                        intrstn_dist,
                                        0, 1] # number of Rayleigh and Love pairs
                else: pairs_dict[pair][6] += 1 # increment number of Love pairs

    return pairs_dict


if __name__ == "__main__": 
    starttime = Timestamp.now()
    args = get_options()
    pairdist_min, pairdist_max = [float(x) for x in args.separation.split(",")]
    Path(args.pairs_dir).mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(f"Reading from {args.data_path}")
        print(f"Writing to... ")
        print(f"    Data list: {args.output_path}")
        print(f"    Pairs directory: {args.pairs_dir}")

    if not args.read: 
        data_dict = lt.query_data_directory(datadir=args.data_path,
                                            dirtype=1,
                                            verb=args.verbose)

        lt.write_stn_ev_dict(evdict=data_dict, 
                             outfile=args.output_path, 
                             gz=False,
                             verb=args.verbose)
    
    else: 
        if args.verbose: print(f"Reading data list from {args.read}")
        data_dict = lt.read_stn_ev_dict(infile=args.read)

    events = sorted(data_dict.keys())

    # initialise pairs dict
    pairs_dict:dict = {}

    for ev in events:
        pairs_dict = process_event(event=data_dict[ev],
                                   pairdist_min=pairdist_min, pairdist_max=pairdist_max, azimuth=args.azimuth,
                                   pairs_dict=pairs_dict,
                                   pairs_dir=args.pairs_dir,
                                   verbose=args.verbose)

    if pairs_dict:
        # sort by number of pairs
        sorted_pairs_list = sorted(pairs_dict.items(), key=lambda x: (x[1][5] + x[1][6]), reverse=True)
        pairs_list_path = f"{args.pairs_dir}/pairs.list"
        with open(pairs_list_path, "w") as f:
            for pair in sorted_pairs_list: # needs checking
                f.write("{0:17s} {1:-10.5f} {2:-9.5f}  {3:-10.5f} {4:-9.5f}  {5:-7.2f}  R:{6:5d} L:{7:5d}\n"
                        .format(pair[0], pair[1][0], pair[1][1], pair[1][2], pair[1][3], pair[1][4], pair[1][5], pair[1][6]))
    
    elapsedtime = Timestamp.now() - starttime
    print('Elapsed time: {}'.format(elapsedtime))