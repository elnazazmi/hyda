
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import numpy as np
import pandas as pd
import random
import math
import os
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans

OUTPUTPATH = os.path.join(os.path.dirname(__file__), '../data/')
    
# apply K-means
def mat_kmeans(df_Feat_Norm, k, outputname, init='k-means++'):

    mydata = pd.DataFrame()
    mydata = df_Feat_Norm.copy()
    
    kmeans = KMeans(init=init, n_clusters=k, n_init=10, max_iter=300, random_state=0).fit(mydata)
    pred = kmeans.labels_
   
    represent = pd.DataFrame()

    for j in np.unique(pred):
        df_dist = pd.DataFrame(metrics.pairwise.euclidean_distances(mydata[pred == j], mydata[pred == j]), index=mydata[pred == j].index.values, columns=mydata[pred == j].index.values)
        reptiv = df_dist.sum().idxmin()

        represent = represent.append({'representative':reptiv, 'label':j}, ignore_index=True)
        
    # save hillslope groups as csv file
    mydata['pred'] = pred
    for r in np.unique(pred):
        mydata.loc[mydata['pred'] == r,'rep'] = represent.loc[represent['label'] == r, 'representative'].values[0]
        
    # mydata[['pred', 'rep']].to_csv(OUTPUTPATH + outputname + str(k) + '.csv', header=['label', 'representative'])
    # represent.to_csv(OUTPUTPATH + outputname + str(k) + 'rep.csv', index=False, header=True)
    
    mydata_dict = mydata[['pred', 'rep']].to_dict()
    mydata_list = [mydata.index.values.tolist()] + [mydata['rep'].tolist()] + [mydata['pred'].tolist()]
    
    return [mydata_list, represent['representative'].tolist(), mydata_dict, kmeans.cluster_centers_]

# calculate RMSE
def mat_rmse(o_path, c_path, df_HS_runtime, out_count):

    df_rmse = pd.DataFrame()
    arr_ctime = np.empty([0])
    output_all = pd.read_csv(c_path + 'output_all_.csv')
    output_all = output_all.fillna(0)

    # map representatives outputs into all hillslopes
    for k in range(1, out_count):
        ctime = 0.0
        rmse_ = pd.DataFrame()
        clust_data = pd.read_csv(c_path + 'mat_kmeans' + str(k) + '.csv')
        clust_data.columns = ['hsname', 'label', 'rep']
        # output_data = pd.read_csv(o_path + 'output_' + str(k) + '/output_all_.csv')
        # output_data = output_data.fillna(0)
        output_mapped = pd.read_csv(c_path + 'output_names.csv')

        for i in np.unique(clust_data['rep']):
            for j in clust_data.loc[clust_data['rep'] == i, 'hsname']:
                for m in range(3):
                    # output_mapped[output_mapped.columns[output_mapped.columns.str.contains(j)][m]] = output_data[output_data.columns[output_data.columns.str.contains(i)][m]]
                    output_mapped[output_mapped.columns[output_mapped.columns.str.contains(j)][m]] = output_all[ output_all.columns[output_all.columns.str.contains(i)][m]]
            ctime = ctime + float(df_HS_runtime['C_Time'][df_HS_runtime.index.values == i])

        # calculate RMSE
        for n in output_all.columns:
            rmse = math.sqrt(metrics.mean_squared_error(output_all[n], output_mapped[n]))
            rmse_ = rmse_.append({'outname': n, 'rmse': rmse}, ignore_index=True)

        df_rmse[str(k)] = rmse_['rmse']
        arr_ctime = np.append(arr_ctime, ctime)

        output_mapped.to_csv(c_path + 'mapped_out' + str(k) + '.csv', index=False, header=True, float_format='%.6f')

    df_RmseSum_Kmeans = pd.DataFrame({'rmse_sum': np.sqrt(np.square(df_rmse).sum()), 'ctime': arr_ctime})
    df_RmseSum_Kmeans.to_csv(c_path + 'rmsesum.csv', index=False, header=True, float_format='%.6f')

    return df_RmseSum_Kmeans