from eulith.cc.utils import pairs_stats, generate_pvt_input

if __name__ == "__main__":
    master_path = "/space/jhyl3/script_tests/cc/cc_all"
    pairs_dir = f"{master_path}/pairs_z10"
    nev_df = pairs_stats(f"{pairs_dir}/pairs.list", condensed=True)
    condition = nev_df['n_ev'] > 50
    print(f"Number of filtered pairs: {condition.sum()}")
    pairs_list = nev_df.loc[condition, 'pair'].tolist()

    generate_pvt_input(pairs_dir=pairs_dir,
                       wt=['R', 'L'],
                       pairs_list=pairs_list,
                       out_dir=f"{master_path}/lists",
                       list_name='ev50')