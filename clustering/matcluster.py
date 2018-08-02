
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
def mat_kmeans(df_AS, df_Feat_Norm, df_HS_runtime, k, outputname):

    mydata = pd.DataFrame()
    mydata = df_Feat_Norm.copy()
    
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=20, max_iter=600).fit(mydata)
    pred = kmeans.labels_
   
    represent = pd.DataFrame()

    for j in np.unique(pred):
        df_dist = pd.DataFrame(metrics.pairwise.euclidean_distances(mydata[pred == j], mydata[pred == j]), index=mydata[pred == j].index.values, columns=mydata[pred == j].index.values)
        reptiv = df_dist.sum().idxmin()

        represent = represent.append({'representitive':reptiv, 'label':j}, ignore_index=True)   
        
    # save hillslope groups as csv file
    mydata['pred'] = pred
    for r in np.unique(pred):
        mydata.loc[mydata['pred'] == r,'rep'] = represent.loc[represent['label'] == r, 'representitive'].values[0]
        
    mydata[['pred', 'rep']].to_csv(OUTPUTPATH + outputname + str(k) + '.csv', header=['label', 'representitive'])
    represent.to_csv(OUTPUTPATH + outputname + str(k) + 'rep.csv', index=False, header=True)
    
    mydata_dict = mydata[['pred', 'rep']].to_dict()
    
    return [mydata['rep'].tolist(), represent['representitive'].tolist(), mydata_dict]
