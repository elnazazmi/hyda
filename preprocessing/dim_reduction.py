# -*- coding: utf-8 -*-

__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import numpy as np
import pandas as pd


INPUTFILE = '../data/total_storage.csv'

# read input data
def read_data(path):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    
    return data

# calculate active storage (initial state-current state) * 100 / initial state and total active storage and time to equilibrium
def calc_active_storage(data):
    df_AS = pd.DataFrame()
    df_AS['date'] = data['date']
    AS = TtE = np.empty([0])

    for i in range(1, data.shape[1]):
        df_AS[data.columns[i]] = (data.iloc[0, i] - data.iloc[:, i]) * 100 / data.iloc[0, i]
        AS = np.append(AS, df_AS.iloc[:, i].loc[~df_AS.iloc[:, i].isnull()].iloc[-1])
        TtE = np.append(TtE, df_AS.iloc[:, i].loc[~df_AS.iloc[:, i].isnull()].shape[0]-1)
    
    return [df_AS, AS, TtE]

# extract features
def extr_features(df_AS, AS, TtE):
    df_Feature = pd.DataFrame()
    df_Feature['Mean'] = df_AS.iloc[:, 1:].mean() # mean - moment1
    df_Feature['Variance'] = df_AS.iloc[:, 1:].var() # variance - moment2
    df_Feature['Skewness'] = df_AS.iloc[:, 1:].skew() # measure of the asymmetry  - moment3
    df_Feature['Kurtosis'] = df_AS.iloc[:, 1:].kurtosis() # descriptor of the shape - moment4

    td = df_AS.loc[~df_AS.isnull().any(axis=1)].iloc[-1]['date'] - df_AS.iloc[0,0]
    Tn = td.total_seconds()
    df_Feature['1st_Gradient'] = np.array(((df_AS.loc[~df_AS.isnull().any(axis=1)].iloc[-1][1:]) / Tn).astype(np.float)) # delta AS at nan / delta Time

    df_Feature['Active_Storage'] = AS # active storage
    df_Feature['Time_to_Equilibrium'] = TtE # time to equilibrium
    
    # replace nan values manually
    null_values = df_Feature.loc[df_Feature['Kurtosis'].isnull()].index.values
    for hname in null_values:
        N = df_AS[hname].loc[~df_AS[hname].isnull()].size
        sigma = 0.0
        
        for i in range(N):
            sigma = sigma + (((df_AS[hname][i] - df_AS[hname].mean()) ** 4) / N)
        df_Feature['Kurtosis'].loc[hname] = (sigma / (df_AS[hname].std()** 4)) - 3
        
        df_Feature['Skewness'].loc[hname] = (3 * (df_AS[hname].mean() - df_AS[hname].median())) / df_AS[hname].std()
    
    return df_Feature

# normalize features with (x-mean(x))/std(x)
def norm_features(df_Feature):
    df_Feat_Norm = pd.DataFrame()
    for i in range(df_Feature.shape[1]):
        df_Feat_Norm[df_Feature.columns[i]] = (df_Feature.iloc[:, i] - df_Feature.iloc[:,i].mean()) / df_Feature.iloc[:, i].std()
    
    return df_Feat_Norm

# calculate correlation of features
def calc_correlation(df_Feat_Norm):
    df_Corr_pearson = df_Feat_Norm.corr(method='pearson')
    
    return df_Corr_pearson

# filter features
def filt_features(df_Feat_Norm, df_Corr_pearson):
    high_corr = df_Corr_pearson[df_Corr_pearson.mask(np.triu(np.ones(df_Corr_pearson.shape)).astype(bool)) >= 0.9].dropna(axis=1, thresh=1).columns.tolist()
    df_Feat_Norm = df_Feat_Norm.drop(high_corr, axis=1)
    
    return df_Feat_Norm

# main
if __name__ == '__main__':
    data = read_data(INPUTFILE)
    df_AS, AS, TtE = calc_active_storage(data)
    df_Feature = extr_features(df_AS, AS, TtE)
    df_Feat_Norm = norm_features(df_Feature)
    df_Corr_pearson = calc_correlation(df_Feat_Norm)
    df_Feat_Norm = filt_features(df_Feat_Norm, df_Corr_pearson)