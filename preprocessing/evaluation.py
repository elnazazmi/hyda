
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import numpy as np
import pandas as pd
from hydroeval import *
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp

# Compute the root-mean-square error between each method and original run
def rmse(data):
    df_rmse = pd.DataFrame()
    for i in range(data.shape[1]):
        df_rmse = df_rmse.append({'method':data.columns.values[i], 'rmse':np.sqrt(mean_squared_error(
            data['Original'], data.iloc[:, i]))}, ignore_index=True, sort=False)
    df_rmse = df_rmse.set_index('method')
    df_rmse = df_rmse.sort_values(by=['rmse'])
    
    return df_rmse

# Compute the pearson correlation between each method and original run
def pear_corr(data):
    df_corr = pd.DataFrame()
    for i in range(data.shape[1]):
        df_corr = df_corr.append({'method':data.columns.values[i], 'corr':data.iloc[:, i].corr(
            data['Original'], method='pearson')}, ignore_index=True, sort=False)
    df_corr = df_corr.set_index('method')
    df_corr = df_corr.sort_values(by=['corr'], ascending=False)
    
    return df_corr

# Compute the correlation of percentage change between each method and original run
def pctch_corr(data):
    df_pctch = data.pct_change()
    df_pct_corr = pd.DataFrame()
    for i in range(data.shape[1]):
        df_pct_corr = df_pct_corr.append({'method':df_pctch.columns.values[i], 'pct_corr':df_pctch.iloc[:, i].corr(
            df_pctch['Original'])}, ignore_index=True, sort=False)
    df_pct_corr = df_pct_corr.set_index('method')
    df_pct_corr = df_pct_corr.sort_values(by=['pct_corr'], ascending=False)
    
    return df_pct_corr

# Compute the Kolmogorov-Smirnov statistic between each method and original run
def kolmo_smir(data):
    df_ks = pd.DataFrame()
    for i in range(data.shape[1]):
        s, p = ks_2samp(data.iloc[:, i], data['Original'])
        df_ks = df_ks.append({'method':data.columns.values[i], 'statistic':s, 'pvalue':p},
                             ignore_index=True, sort=False)
    df_ks = df_ks.set_index('method')
    df_ks = df_ks.sort_values(by=['pvalue'], ascending=False)
    
    return df_ks

# an evaluation summary table
def sum_table(data, forcing, outputpath):
    df_forcing = pd.DataFrame(forcing, columns=['forcing'])
    df_forcing = df_forcing.set_index(data.index.values)
    df_woforc = data.loc[df_forcing['forcing'] == 0.00]
    df_wforc = data.loc[df_forcing['forcing'] != 0.00]
    df_table = pd.DataFrame(index=data.columns.values)
    df_table['RMSE'] = rmse(data).round(4)
    df_table['RMSE-WOF'] = rmse(df_woforc).round(4)
    df_table['RMSE-WF'] = rmse(df_wforc).round(4)
    df_table['PCC'] = pear_corr(data).round(3)
    df_table['PCC-WOF'] = pear_corr(df_woforc).round(3)
    df_table['PCC-WF'] = pear_corr(df_wforc).round(3)
    df_table['PCT-PCC'] = pctch_corr(data).round(3)
    df_table['PCT-PCC-WOF'] = pctch_corr(df_woforc).round(3)
    df_table['PCT-PCC-WF'] = pctch_corr(df_wforc).round(3)
    # Nash-Sutcliffe Efficiency (Nash and Sutcliffe 1970 - https://doi.org/10.1016/0022-1694(70)90255-6)
    # my_nse = evaluator(nse, simulated_flow, observed_flow)
    df_table['NSE'] = evaluator(nse, np.array(data), np.array(data['Original'])).round(3)
    # Original Kling-Gupta Efficiency (Gupta et al. 2009 - https://doi.org/10.1016/j.jhydrol.2009.08.003)
    # my_kge = evaluator(kge, simulated_flow, observed_flow)
    df_table['KGE'] = evaluator(kge, np.array(data), np.array(data['Original']))[0].round(3)
    df_table['PVALUE'] = kolmo_smir(data)['pvalue']
    df_table['STATISTIC'] = kolmo_smir(data)['statistic'].round(3)
    
    df_table.to_csv(outputpath+'evaluation.csv')
              
    return df_table
