
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
    print('Reading the input data ...')
    data = predim.read_data(INPUTFILE)
    
    vis_indata = input('Visualize input data? yes/no ')
    if vis_indata == 'yes':
        visvis.plot_timeseries(data, 'total_storage')
    else:
        print('Preprocessing ...')
    df_AS, AS, TtE = predim.calc_active_storage(data)
    vis_ac = input('Visualize active storage data? yes/no ')
    if vis_ac == 'yes':
        visvis.plot_timeseries(df_AS, 'active_storage')
    else:
        print('Preprocessing ...')
    df_Feature = predim.extr_features(df_AS, AS, TtE)
    df_Feat_Norm = predim.norm_features(df_Feature)
    df_Corr_pearson = predim.calc_correlation(df_Feat_Norm)
    df_Feat_Norm = predim.filt_features(df_Feat_Norm, df_Corr_pearson)
    
    clust_method = input('Select one of the clustering methods from [\'kmeans\', \'dbscan\', \'kmedoids\']: ')
    
    df_HS_runtime = clusclus.read_data(INPUTCOMPTIME)
    if clust_method == 'kmeans':
        df_RmseSum_Kmeans = clusclus.apply_kmeans(df_AS, df_Feat_Norm, df_HS_runtime)
        df_Norm_Clust = clusclus.norm_rmse_ctime(df_RmseSum_Kmeans, df_HS_runtime)
        visvis.plot_rmse_ctime(df_Norm_Clust, 'rmse_ctime_Kmeans')
    elif clust_method == 'dbscan':
        df_RmseSum_DBSCAN = clusclus.apply_DBSCAN(df_AS, df_Feat_Norm)
        df_Norm_Clust = clusclus.norm_rmse_ctime(df_RmseSum_DBSCAN, df_HS_runtime)
        visvis.plot_rmse_ctime(df_Norm_Clust, 'rmse_ctime_dbscan')
    elif clust_method == 'kmedoids':
        df_RmseSum_Kmedoids = clusclus.apply_kmedoids(df_AS, df_Feat_Norm)
        df_Norm_Clust = clusclus.norm_rmse_ctime(df_RmseSum_Kmedoids, df_HS_runtime)
        visvis.plot_rmse_ctime(df_Norm_Clust, 'rmse_ctime_Kmedoids')
    else:
        print('Enter a valid clustering method!')
        
    print('The program is completed!')
