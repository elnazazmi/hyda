
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import preprocessing.dim_reduction as predim
import clustering.clustering as clusclus
import visualization.visualization as visvis

INPUTFILE = 'data/total_storage.csv'
INPUTCOMPTIME = 'data/hillslopes_computation_time.csv'
OUTPUTPATH = 'data/'

if __name__ == '__main__':
    data = predim.read_data(INPUTFILE)
    df_AS, AS, TtE = predim.calc_active_storage(data)
    df_Feature = predim.extr_features(df_AS, AS, TtE)
    df_Feat_Norm = predim.norm_features(df_Feature)
    df_Corr_pearson = predim.calc_correlation(df_Feat_Norm)
    df_Feat_Norm = predim.filt_features(df_Feat_Norm, df_Corr_pearson)
    
    visvis.plot_timeseries(data, 'total_storage')
    visvis.plot_timeseries(df_AS, 'active_storage')
    
    df_HS_runtime = clusclus.read_data(INPUTCOMPTIME)
    df_RmseSum_Kmeans = clusclus.apply_kmeans(df_AS, df_Feat_Norm)
    df_RmseSum_DBSCAN = clusclus.apply_DBSCAN(df_AS, df_Feat_Norm)
    df_RmseSum_Kmedoids = clusclus.apply_kmedoids(df_AS, df_Feat_Norm)
