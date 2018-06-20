
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# read hillslopes computation time
def read_data(path):
    df_HS_runtime = pd.read_csv(path)
    df_HS_runtime = df_HS_runtime.set_index(['HS_Name'])
    
    return df_HS_runtime

# apply K-means
def apply_kmeans(df_AS, df_Feat_Norm):
    df_rmse = pd.DataFrame()
    arr_ctime = np.empty([0])
    df_fillnan = df_AS.fillna(method='ffill')
    n_hs = df_Feat_Norm.shape[0]+1

    for i in range(1, n_hs): 
        mydata = pd.DataFrame()
        mydata = df_Feat_Norm.copy()
        
        n_clusters = i
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=20, max_iter=600).fit(mydata)
        pred = kmeans.labels_

        ctime = 0.0
        
        # representitive hillslope
        rmse_ = pd.DataFrame()    
        represent = pd.DataFrame()

        for j in np.unique(pred):
            df_dist = pd.DataFrame(metrics.pairwise.euclidean_distances(mydata[pred == j], mydata[pred == j]), index=mydata[pred == j].index.values, columns=mydata[pred == j].index.values)
            reptiv = df_dist.sum().idxmin()
            ctime = ctime + float(df_HS_runtime['C_Time'][df_HS_runtime.index.values == reptiv])
            
            represent = represent.append({'representitive':reptiv, 'label':j}, ignore_index=True)
            
            # Root Mean Square Error
            for k in range(mydata[pred == j].shape[0]):
                rmse = math.sqrt(metrics.mean_squared_error(df_fillnan[mydata[pred == j].iloc[k].name], df_fillnan[reptiv]))
                rmse_ = rmse_.append({'hname':mydata[pred == j].iloc[k].name, 'rmse':rmse}, ignore_index=True)

        rmse_ = rmse_.set_index(['hname'])
        rmse_ = rmse_.sort_index()
        df_rmse[str(i)] = rmse_['rmse']
        arr_ctime = np.append(arr_ctime, ctime)
        
    df_RmseSum_Kmeans = pd.DataFrame({'rmse_sum':np.sqrt(np.square(df_rmse).sum()), 'ctime':arr_ctime})
    
    return df_RmseSum_Kmeans

# apply DBSCAN
def apply_DBSCAN(df_AS, df_Feat_Norm):
    df_fillnan = df_AS.fillna(method='ffill')
    df_rmsesum = pd.DataFrame()

    for e in np.arange(0.1, 2.4, 0.2):
        df_rmse = pd.DataFrame()
        arr_ctime = np.empty(shape=[0, 1])
        count = 0
            
        for i in range(1, 22, 2):
            mydata = pd.DataFrame()
            mydata = df_Feat_Norm.copy()
            eps = e
            min_samples = i
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='auto').fit(mydata)
            pred = dbscan.labels_

            ctime = 0.0

            # representitive hillslope
            rmse_ = pd.DataFrame()

            for j in np.unique(pred):
                if j == -1:
                    for k in range(mydata[pred == j].shape[0]):
                        ctime = ctime + float(df_HS_runtime['C_Time'][df_HS_runtime.index.values == mydata[pred == j].iloc[k].name])
                        rmse = 0.0
                        rmse_ = rmse_.append({'hname':mydata[pred == j].iloc[k].name, 'rmse':rmse}, ignore_index=True)
                else:                
                    df_dist = pd.DataFrame(metrics.pairwise.euclidean_distances(mydata[pred == j], mydata[pred == j]), index=mydata[pred == j].index.values, columns=mydata[pred == j].index.values)
                    reptiv = df_dist.sum().idxmin()
                    ctime = ctime + float(df_HS_runtime['C_Time'][df_HS_runtime.index.values == reptiv])

                    # Root Mean Square Error
                    for k in range(mydata[pred == j].shape[0]):
                        rmse = math.sqrt(metrics.mean_squared_error(df_fillnan[mydata[pred == j].iloc[k].name], df_fillnan[reptiv]))
                        rmse_ = rmse_.append({'hname':mydata[pred == j].iloc[k].name, 'rmse':rmse}, ignore_index=True)

            rmse_ = rmse_.set_index(['hname'])
            rmse_ = rmse_.sort_index()
            df_rmse[str(i)] = rmse_['rmse']
            arr_ctime = np.append(arr_ctime, ctime)
            nr_cls = len(np.unique(pred)) - 1 + mydata[pred == -1].shape[0]
            df_RmseSum_DBSCAN= df_RmseSum_DBSCAN.append({'rmse_sum':np.sqrt(np.square(df_rmse).sum())[count], 'ctime':arr_ctime[count], 'epsilon':e, 'min_samp':i,
                                            'nr_cls':nr_cls, 'silhouettecoef':metrics.silhouette_score(mydata, pred)}, ignore_index=True)
            count +=1

    return df_RmseSum_DBSCAN

# K-medoids clustering
def cluster(distances, k=3):

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)
   
    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point. 
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids

# K-medoids clustering assign points
def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

# K-medoids clustering compute new medoid
def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)

# apply K-medoids
def apply_kmedoids(df_AS, df_Feat_Norm):
    mydata = pd.DataFrame()
    mydata = df_Feat_Norm.copy()
    df_dist = pd.DataFrame(metrics.pairwise.euclidean_distances(mydata), index=mydata.index.values, columns=mydata.index.values)

    dist = np.array(df_dist)
    df_rmse = pd.DataFrame()
    arr_ctime = np.empty([0])
    df_fillnan = df_AS.fillna(method='ffill')
    n_hs = 81

    for i in range(2, n_hs): 
        mydata = pd.DataFrame()
        mydata = df_Feat_Norm.copy()
        n_clusters = i

        kclust, kmedoids = cluster(dist, k=n_clusters)

        ctime = 0.0
        
        # representitive hillslope
        rmse_ = pd.DataFrame()    
        represent = pd.DataFrame()

        for j in (kmedoids):
            reptiv = mydata.iloc[j].name
            ctime = ctime + float(df_HS_runtime['C_Time'][df_HS_runtime.index.values == reptiv])
            
            represent = represent.append({'representitive':reptiv, 'label':j}, ignore_index=True)
            
            # Root Mean Square Error
            for k in range(np.where(kclust == j)[0].shape[0]):
                rmse = math.sqrt(metrics.mean_squared_error(df_fillnan[mydata[kclust == j].iloc[k].name], df_fillnan[reptiv]))
                rmse_ = rmse_.append({'hname':mydata[kclust == j].iloc[k].name, 'rmse':rmse}, ignore_index=True)

        rmse_ = rmse_.set_index(['hname'])
        rmse_ = rmse_.sort_index()
        df_rmse[str(i)] = rmse_['rmse']
        arr_ctime = np.append(arr_ctime, ctime)
        
    df_RmseSum_Kmedoids = pd.DataFrame({'rmse_sum':np.sqrt(np.square(df_rmse).sum()), 'ctime':arr_ctime})
    
    return df_RmseSum_Kmedoids