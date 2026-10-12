import pandas as pd
from obspy import UTCDateTime
from obspy.taup.tau import TauPyModel as TPM


from eulith.core.evstdf import read_evst, evst_generator


class evst:
    def __init__(self, path:str=None, df:pd.DataFrame=None) -> None | pd.DataFrame:

        if df: 
            # do some checks
            self.df = df
            self.index()
            return self.df 

        if path: 
            self.read(path)

 
    def read(self, in_path:str):
        if in_path:
            self.path = in_path
            self.df = read_evst(self.path)
            self.index()
            return self.df

    def generate(self,
                 cmt_df:pd.DataFrame,
                 stnlist_df:pd.DataFrame,
                 tt_model:str='ak135',
                 stn_check:list=[],
                 _node_path:str=None,
                 eida_cred:str=None,
                 vg_min:float=2.8, 
                 dist_min:int=400, dist_max:int=18000, 
                 ADIS:float=4.9, BDIS:float=6.67e-5
    ) -> pd.DataFrame: 


        """
        Generate station-event pairs based on the provided data and criteria.
        
        :type vg_min: float
        :type dist_min: int
        :type dist_max: int
        :type ADIS: float
        :type BDIS: float
        :param vg_min: Minimum group velocity in km/s.
        :param dist_min: Minimum epicentral distance in km.
        :param dist_max: Maximum epicentral distance in km.
        :param ADIS: Constant for calculating the magnitude threshold.
        :param BDIS: Constant for calculating the magnitude threshold.

        
        :return: DataFrame containing the generated station-event pairs.

        """

        try:
            self.df = evst_generator(cmt_df=cmt_df, 
                                     stnlist=stnlist_df,
                                     tt_model=tt_model,
                                     stn_check=stn_check,
                                     _node_path=_node_path,
                                     eida_cred=eida_cred,
                                     vg_min=vg_min, 
                                     dist_min=dist_min, dist_max=dist_max,
                                     ADIS=ADIS, BDIS=BDIS)
            self.index()
            print("Access by calling `evst.df'.")
        except ValueError: raise SystemExit("No station-event pairs available after generating evst dataframe. Exiting.")
        
        return self.df

    def write(self, out_path:str) -> None:

        if (not out_path) and (not self.path): 
            raise ValueError("Output path must be provided to write the evst dataframe.")
        if not out_path: out_path = self.path
        elif self.path: 
            print("Overwriting existing path")
            self.path = out_path

        self.df.to_csv(out_path, mode='w', index=False)
        print(f"Wrote evst dataframe to {out_path}.")

        return

    def merge(self, other_evst:pd.DataFrame, 
              status_sel:str='max',
              inplace:bool=False) -> pd.DataFrame:
        if not isinstance(other_evst, pd.DataFrame):
            if not isinstance(other_evst, evst):
                raise TypeError("The supplied evst must be a pandas DataFrame or an instance of the evst class.")
            else: other_evst = other_evst.df

        # aggregate dictionary
        agg_dict = {col: 'first' for col in self.columns}
        agg_dict['status'] = status_sel

        evst_merged = pd.concat([self.df, other_evst], ignore_index=True)\
                        .groupby('data_path', as_index=False)\
                        .agg(agg_dict)
        if inplace: 
            print("Overwriting existing evst...")
            self.df = evst_merged
            return self.df
        else:
            return evst(df=evst_merged)

    def index(self):
        if self.df is None:
            raise ValueError("need dataframe to index")
        self.status = self.df['status'].value_counts()
        self.stcodes = self.df['stcode'].unique().tolist()
        self.networks = self.df['network'].unique().tolist()
        self.nodes = self.df['node'].unique().tolist()
        self.evpaths = self.df['evpath'].unique().tolist() # indexes unique events
        self.count = self.df.shape[0]
        self.cols = self.df.columns.tolist()


def merge_evsts(evst_list:list[evst], status_sel:str='max'):

    columns = evst_list[0].df.columns
    agg_dict = {col: 'first' for col in columns}
    agg_dict['status'] = status_sel

    evst_merged = pd.concat([evst.df for evst in evst_list], ignore_index=True)\
                    .groupby('data_path', as_index=False)\
                    .agg(agg_dict)

    return evst_merged