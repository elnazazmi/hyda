
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import os
import numpy as np
import pandas as pd
import preprocessing.dim_reduction as predim
import preprocessing.read_data as reda
import clust_forc as cf
import clustering.matcluster as clusmat
from sklearn.decomposition import PCA

# INPUTFILE = os.path.join(os.path.dirname(__file__), 'data/icon-art-NWP_GASPHASE-chem-1file_DOM01_ML_0001.nc')
INPUTFILE = '/home/elnaz/workspace/fortran_proj/iconart/icon-art-NWP_GASPHASE-chem-1file_DOM01_ML_0001.nc'

def struct_features():
    data = reda.read_nc(INPUTFILE)
    df_Feature = pd.DataFrame()
    for k in data.variables.keys():
        if k not in ['clon', 'clon_bnds', 'clat', 'clat_bnds', 'height', 'height_bnds', 'time', 'cosmu0']:
            var = reda.read_varnc(data, k, 0)
            sf = predim.extr_feat(var)
            sf = sf.rename(columns=lambda s: s + '-' + k)
            df_Feature = pd.concat([df_Feature, sf], axis=1)

    df_Feat_Norm = predim.norm_features(df_Feature)
    # zero values are NAN after normalization with (x-mean(x))/std(x)
    # sf.columns[sf.isnull().all()].tolist()  92 features - 28 nan - 3 inf
    df_Feat_Norm = df_Feat_Norm.replace([np.inf, -np.inf], np.nan)
    df_Feat_Norm = df_Feat_Norm.dropna(axis=1, how='all')
    df_Corr_pearson = predim.calc_correlation(df_Feat_Norm)
    s_feat = predim.filt_features(df_Feat_Norm, df_Corr_pearson)

    pca = PCA(n_components=4)
    s_feat_p = pca.fit_transform(s_feat)
    s_feat_pca = pd.DataFrame(s_feat_p, index=s_feat.index.values, columns=['F1', 'F2', 'F3', 'F4'])

    return s_feat_pca

def init_clus():
    features = struct_features()
    k = cf.apply_elbow(features)
    w, x, y, z = clusmat.mat_kmeans(features, k, 'output')

    return w, x, y, z

if __name__ == '__main__':
    import hyda.clust_art as ac
    w, x, y, z = ac.init_clus()
