 
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import os
import preprocessing.read_data as reda
import preprocessing.evaluation as preval
import visualization.visual_evol as viseval

PATH = os.path.join(os.path.dirname(__file__), 'data/processing/')

if __name__ == '__main__':
    df = reda.df_outlet(PATH + 'discharge/')
    forcing = reda.read_forcmat(PATH + 'forcing/rainfall_0101-0701-2015.mat')
    viseval.outlet_vis(df, forcing, PATH + 'output_0101-0107-2015.svg')
    df_sumtable = preval.sum_table(df, forcing, PATH)
    df_runtime, df_speedup = reda.read_log(PATH +'run_time.log')
    viseval.rmse_speedup(df_sumtable['RMSE'], df_speedup, PATH +'rmse_speedup.pdf')