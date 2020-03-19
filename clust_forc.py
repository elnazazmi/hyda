import os
import math
import numpy as np
import pandas as pd
import numpy.matlib
import hyda.preprocessing.dim_reduction as predim
import hyda.clustering.clustering as clusclus
import hyda.clustering.matcluster as clusmat
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.io import loadmat

INPUTFILE = os.path.join(os.path.dirname(__file__), 'data/total_storage_166.csv')
INPUTCOMPTIME = os.path.join(os.path.dirname(__file__), 'data/hillslopes_computation_time_166.csv')

def clust_forc(matfile, k, ccenter, t):
    mat = loadmat(matfile)
    colname = np.array([])
    for i in range(1, len(mat['clust_data'][0])):
        colname = np.append(colname, mat['clust_data'][0][i][0])
    indx = np.array([])
    for i in range(1, len(mat['clust_data'])):
            indx = np.append(indx, mat['clust_data'][i][0][0])
    cols = np.zeros(shape=(len(mat['clust_data'])-1,len(mat['clust_data'][0])-1))
    for i in range(1, len(mat['clust_data'])):
        for j in range(1, len(mat['clust_data'][i])):
            cols[i-1, j-1] = mat['clust_data'][i][j][0][0]
    df = pd.DataFrame(data=cols, index=indx, columns=colname)
    
    if ccenter.shape[1] > df.shape[1]:
        pca = PCA(n_components=df.shape[1])
        ccenter = pca.fit_transform(ccenter)
        
    df_norm = predim.norm_features(df)
    df_norm = df_norm.fillna(0.00)
    x, y, z, ccenter = clusmat.mat_kmeans(df_norm, k, 'mat_kmeans_t'+str(t), ccenter)
    clusters = pd.DataFrame(z).combine_first(df_norm)
    
    return x, y, z, ccenter, clusters


def struct_features():
    data = predim.read_data(INPUTFILE)
    df_AS, AS, TtE = predim.calc_active_storage(data)
    df_Feature = predim.extr_features(df_AS, AS, TtE)
    df_Feat_Norm = predim.norm_features(df_Feature)
    df_Corr_pearson = predim.calc_correlation(df_Feat_Norm)
    s_feat = predim.filt_features(df_Feat_Norm, df_Corr_pearson)

    return s_feat


def all_features(matfile, s_feat):
    mat = loadmat(matfile)
    colname = np.array([])
    for i in range(1, len(mat['clust_data'][0])):
        colname = np.append(colname, mat['clust_data'][0][i][0])
    indx = np.array([])
    for i in range(1, len(mat['clust_data'])):
        indx = np.append(indx, mat['clust_data'][i][0][0])
    cols = np.zeros(shape=(len(mat['clust_data']) - 1, len(mat['clust_data'][0]) - 1))
    for i in range(1, len(mat['clust_data'])):
        for j in range(1, len(mat['clust_data'][i])):
            cols[i - 1, j - 1] = mat['clust_data'][i][j][0][0]

    df = pd.DataFrame(data=cols, index=indx, columns=colname)
    df_norm = predim.norm_features(df)
    df_norm = df_norm.fillna(0.00)
    a_feat = pd.concat([s_feat, df_norm], axis=1, sort=False)

    return a_feat


def clust_forc_upd(a_feat, k, ccenter, t):
    x, y, z, ccenter = clusmat.mat_kmeans(a_feat, k, 'mat_kmeans_t' + str(t), ccenter)
    clusters = pd.DataFrame(z).combine_first(a_feat)

    return x, y, z, ccenter, clusters


def apply_kmeans(df_Feat_Norm, df_HS_runtime):
    df_rmse = pd.DataFrame()
    arr_ctime = np.empty([0])
    nr_itter = int((df_Feat_Norm.shape[0] + 1) / 3)

    for i in range(1, nr_itter, 4):
        mydata = df_Feat_Norm.copy()

        n_clusters = i
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=300, random_state=0).fit(mydata)
        pred = kmeans.labels_

        ctime = 0.0

        # representitive hillslope
        rmse_ = pd.DataFrame()
        represent = pd.DataFrame()

        for j in np.unique(pred):
            df_dist = pd.DataFrame(metrics.pairwise.euclidean_distances(mydata[pred == j], mydata[pred == j]),
                                   index=mydata[pred == j].index.values, columns=mydata[pred == j].index.values)
            reptiv = df_dist.sum().idxmin()
            ctime = ctime + float(df_HS_runtime['C_Time'][df_HS_runtime.index.values == reptiv])

            represent = represent.append({'representitive': reptiv, 'label': j}, ignore_index=True)

            # Root Mean Square Error
            for k in range(mydata[pred == j].shape[0]):
                rmse = math.sqrt(
                    metrics.mean_squared_error(mydata.loc[mydata[pred == j].iloc[k].name], mydata.loc[reptiv]))
                rmse_ = rmse_.append({'hname': mydata[pred == j].iloc[k].name, 'rmse': rmse}, ignore_index=True)

        rmse_ = rmse_.set_index(['hname'])
        rmse_ = rmse_.sort_index()
        df_rmse[str(i)] = rmse_['rmse']
        arr_ctime = np.append(arr_ctime, ctime)

    df_RmseSum_Kmeans = pd.DataFrame({'rmse_sum': np.sqrt(np.square(df_rmse).sum()), 'ctime': arr_ctime})

    return df_RmseSum_Kmeans


def k_determiner(features):
    df_HS_runtime = clusclus.read_data(INPUTCOMPTIME)
    df_RmseSum_Kmeans = apply_kmeans(features, df_HS_runtime)
    df_Norm_Clust = clusclus.norm_rmse_ctime(df_RmseSum_Kmeans, df_HS_runtime)
    
    # interpolate data points
    f1 = interp1d(df_Norm_Clust['nr_cls'], df_Norm_Clust['ctime'], kind='cubic')
    f2 = interp1d(df_Norm_Clust['nr_cls'], df_Norm_Clust['rmse_sum'], kind='cubic')
    xnew = list(range(1, int(df_Norm_Clust.iloc[-1,2])))

    # detect intersection by change in sign of difference
    intersect = np.array([], dtype=np.int16)
    diff = f2(xnew) - f1(xnew)
    for i in range(len(diff) - 1):
        if diff[i] == 0. or diff[i] * diff[i + 1] < 0.:
            intersect = np.append(intersect, np.int16(i))
    k = min(intersect)

    return k


def apply_elbow(features):
    md = pd.DataFrame(columns=['nr_cls', 'dist'])
    nr_itter = int((features.shape[0] + 1))

    for i in range(1, nr_itter, 5):
        mydata = pd.DataFrame()
        mydata = features.copy()
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10, max_iter=300, random_state=0).fit(mydata)
        md = md.append({'nr_cls':i, 'dist':sum(np.min(cdist(mydata, kmeans.cluster_centers_,
                                                                                      'euclidean'), axis=1))/mydata.shape[0]}, 
                                       ignore_index=True, sort=False)    
        
    f1 = interp1d(md['nr_cls'], md['dist'], kind='cubic')
    xnew = list(range(1, int(md.iloc[-1, 0] + 1)))
    ynew = f1(xnew)

    # find elbow point
    values=list(ynew)
    nPoints = len(values)
    allCoord = np.vstack((range(1, nPoints+1), values)).T
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    k = np.argmax(distToLine) 

    return np.int16(k)
