import argparse
from eulith.cc.utils import pairs_stats, generate_pvt_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--master_path", 
                        default="/space/jhyl3/script_tests/cc/cc_all",
                        help="Path to CC directory in which pairs directory is stored")
    parser.add_argument("-p", "--pairs_dir",
                        default="pairs_z10",
                        help="Name of pairs directory within master_path")
    parser.add_argument("-e", "--min_events", type=int,
                        default=50,
                        help="Minimum number of events for a pair to be included in the output list")
    parser.add_argument("-lat", "--latrange", type=str,
                        default=None,
                        help="Latitude range for filtering events, in format 'min,max'")
    parser.add_argument("-lon", "--lonrange", type=str,
                        default=None,
                        help="Longitude range for filtering events, in format 'min,max'")
    parser.add_argument("-ln", "--list_name", type=str,
                        default='',
                        help="Optional name for the output list file")
    args = parser.parse_args()

    pairs_dir = f"{args.master_path}/{args.pairs_dir}"
    if args.latrange or args.lonrange: condensed = False
    else: condensed = True
    nev_df = pairs_stats(f"{pairs_dir}/pairs.list", condensed=condensed)

    print(f"Filtering pairs based on minimum number of events: {args.min_events}")
    condition = nev_df['n_ev'] > args.min_events
    if args.latrange: 
        lat_min, lat_max = map(float, args.latrange.split(","))
        # print(type(lat_min), type(lat_max))
        # raise SystemExit
        print(f"Filtering events by latitude range: {lat_min} to {lat_max}")
        lat_condition = nev_df['s1lat'].between(lat_min, lat_max) &\
                        nev_df['s2lat'].between(lat_min, lat_max)
        condition = condition & lat_condition
    if args.lonrange:
        lon_min, lon_max = map(float, args.lonrange.split(","))
        print(f"Filtering events by longitude range: {lon_min} to {lon_max}")
        lon_condition = nev_df['s1lon'].between(lon_min, lon_max) &\
                        nev_df['s2lon'].between(lon_min, lon_max)
        condition = condition & lon_condition

    print(f"Number of filtered pairs: {condition.sum()}")
    pairs_list = nev_df.loc[condition, 'pair'].tolist()

    generate_pvt_input(pairs_dir=pairs_dir,
                       wt=['R', 'L'],
                       pairs_list=pairs_list,
                       out_dir=f"{args.master_path}/lists",
                       list_name=args.list_name)