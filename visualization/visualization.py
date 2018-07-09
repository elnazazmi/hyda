
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from scipy.stats import norm
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import text
from mpl_toolkits.mplot3d import Axes3D
from progress.bar import Bar

OUTPUTPATH = 'data/'

def plot_timeseries(data, outputname):
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b') 

    n = data.shape[1]
    fig, ax = plt.subplots(figsize=(12,8))

    color=iter(cm.rainbow(np.linspace(0,1,n)))
    bar = Bar('Visualizing', max=n-1)
    
    for i in range(1, n):
        c=next(color)
        ax.plot(data['date'], data.iloc[:, i], color=c, linewidth=2)
        bar.next()
    bar.finish()
     
    # format the ticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('time', fontsize=22, labelpad=20)
    plt.ylabel(outputname, fontsize=22, labelpad=20)
    plt.savefig(OUTPUTPATH + outputname + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
def plot_correlations(df_Corr, outputname):
    # Create a mask to display only the lower triangle of the matrix (since it's mirrored around its 
    # top-left to bottom-right diagonal).
    mask = np.zeros_like(df_Corr)
    mask[np.triu_indices_from(mask)] = True
    
    # Create the heatmap using seaborn
    sns.set(font_scale=2)
    fig = plt.figure(figsize=(8,6))
    fig.tight_layout()
    p = sns.heatmap(df_Corr, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
    
    # Show the plot we reorient the labels for each column and row to make them easier to read.
    plt.yticks(rotation=0, fontsize=22) 
    plt.xticks(rotation=90, fontsize=22) 
    plt.savefig(OUTPUTPATH + outputname + '.pdf', bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)
    plt.clf()
    plt.cla()
    plt.close() 
    
def plot_elbow(meandistortions, outputname):    
    # pull out the list
    values=list(meandistortions)

    #get coordinates of all the points
    nPoints = len(values)
    allCoord = np.vstack((range(1, nPoints+1), values)).T
    #np.array([range(nPoints), values])

    # get the first point
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint

    # To calculate the distance to the line, we split vecFromFirst into two 
    # components, one that is parallel to the line and one that is perpendicular 
    # Then, we take the norm of the part that is perpendicular to the line and 
    # get the distance.
    # We find the vector parallel to the line by projecting vecFromFirst onto 
    # the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    # We project vecFromFirst by taking the scalar product of the vector with 
    # the unit vector that points in the direction of the line (this gives us 
    # the length of the projection of vecFromFirst onto the line). If we 
    # multiply the scalar product by the unit vector, we have vecFromFirstParallel
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    
    # find orthogonal point to the line 
    x1 = allCoord[0][0]
    y1 = allCoord[0][1]
    x2 = allCoord[-1][0]
    y2 = allCoord[-1][1]
    x3 = np.float(idxOfBestPoint)
    y3 = values[idxOfBestPoint]
    
    m = (y2 - y1) / (x2 - x1)
    meetPointx = ((m ** 2 * x2) - (m * y2) + x3 + (m * y3)) / (m ** 2 + 1)
    meetPointy = ((-1 / m) * (meetPointx - x3)) + y3
                        
    # plot of the original curve and its corresponding distances
    fig, ax = plt.subplots(figsize=(13,8))
    ax.plot(np.array(range(1, len(meandistortions)+1)), meandistortions,label='Average Distance to Centroid',color='r', linewidth=2)
    ax.plot(idxOfBestPoint, values[idxOfBestPoint], color='g', marker = 'o', label='Knee')
    ax.plot([x1, x2],[y1, y2], color='k', linestyle='dashed')
    text(35.0, 1.3, '* L *', verticalalignment='center', color='k', fontsize=18)
    ax.plot([x3, meetPointx], [y3, meetPointy], color='k', linestyle='dashed')
    text(40.0, 0.6, '* Orthogonal to L *', verticalalignment='center', color='k', fontsize=18)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.set_xlim(0.00) 
    ax.set_ylim(0.00)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.set_xlabel('Number of Clusters (K)', fontsize=22, labelpad=20)
    ax.set_ylabel('Average Distance to Centroid', fontsize=22, labelpad=20)

    plt.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, fancybox=True, shadow=False, fontsize=22)
    fig.savefig(OUTPUTPATH + outputname + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()

def plot_rmse_ctime(df_Norm_Clust, outputname):
    fig, ax = plt.subplots(figsize=(13,8))
    ax_ = ax.twinx()
    
    if not df_Norm_Clust['nr_cls'].duplicated().any():
        ax.scatter(df_Norm_Clust['nr_cls'], df_Norm_Clust['rmse_sum'], label='_nolegend_', color='silver', edgecolor='gray', s=80)
        ax_.scatter(df_Norm_Clust['nr_cls'], df_Norm_Clust['ctime'], label='_nolegend_', color='silver', edgecolor='gray', s=80)
        
        x_smooth = np.linspace(df_Norm_Clust['nr_cls'].min(), df_Norm_Clust['nr_cls'].max(), 20)
        rmse_sum_tck = interpolate.splrep(df_Norm_Clust['nr_cls'], df_Norm_Clust['rmse_sum'], s=0)
        rmse_sum_smooth = interpolate.splev(x_smooth, rmse_sum_tck, der=0)
        ax.plot(x_smooth, rmse_sum_smooth, label='RMSE', color='r', linewidth=2)

        ctime_tck = interpolate.splrep(df_Norm_Clust['nr_cls'], df_Norm_Clust['ctime'], s=0)
        ctime_smooth = interpolate.splev(x_smooth, ctime_tck, der=0)

        ax_.plot(x_smooth, ctime_smooth, label='Representative Hillslopes Computation Time', color='g', linewidth=2)
    else:
        ax.scatter(df_Norm_Clust['nr_cls'], df_Norm_Clust['rmse_sum'], label='RMSE', color='r', edgecolor='black', s=80)
        ax_.scatter(df_Norm_Clust['nr_cls'], df_Norm_Clust['ctime'], label='Representative Hillslopes Computation Time', color='g', edgecolor='black', s=80)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.set_xlim(0.00) 
    ax.set_ylim(0.00, 1.00)
    ax_.set_ylim(0.00, 1.00)

    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax_.yaxis.set_tick_params(labelsize=18)
    ax.set_xlabel('Number of Clusters', fontsize=18, labelpad=20)
    ax.set_ylabel('RMSE Fraction Normalized by Min-Max', fontsize=18, labelpad=20)
    ax_.set_ylabel('Time Fraction Normalized by Min-Max', fontsize=18, labelpad=20)
    
    # ask matplotlib for the plotted objects and their labels
    pl, labels = ax.get_legend_handles_labels()
    pl_, labels_ = ax_.get_legend_handles_labels()
    ax_.legend(pl + pl_, labels + labels_, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, fancybox=True, shadow=False, fontsize=18)

    plt.savefig(OUTPUTPATH + outputname + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
def plot_methods(df_RmseSum_Kmeans, df_RmseSum_DBSCAN, df_RmseSum_Kmedoids, outputname):    
    fig, ax = plt.subplots(figsize=(13,8))

    df_rmsesum_kmeans = df_RmseSum_Kmeans[['ctime', 'rmse_sum']].drop_duplicates(keep='first')
    df_rmsesum_kmedoids = df_RmseSum_Kmedoids[['ctime', 'rmse_sum']].drop_duplicates(keep='first')
    df_rmsesum_dbscan = df_RmseSum_DBSCAN[['ctime', 'rmse_sum']].drop_duplicates(keep='first')

    df_rmsesum_kmeans['ctime'] = df_rmsesum_kmeans['ctime']/(24 * 3600)
    df_rmsesum_kmedoids['ctime'] = df_rmsesum_kmedoids['ctime']/(24 * 3600)
    df_rmsesum_dbscan['ctime'] = df_rmsesum_dbscan['ctime']/(24 * 3600)

    ax.scatter(df_rmsesum_kmeans['ctime'], df_rmsesum_kmeans['rmse_sum'], label='K-Means', color='r', edgecolor='black', s=80)
    ax.scatter(df_rmsesum_kmedoids['ctime'], df_rmsesum_kmedoids['rmse_sum'], label='K-Medoids', color='b', edgecolor='black', s=80)
    ax.scatter(df_rmsesum_dbscan['ctime'], df_rmsesum_dbscan['rmse_sum'], label='DBSCAN', color='g', edgecolor='black', s=80)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_xlim(0.0) 
    ax.set_ylim(0.000000)

    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.set_xlabel('Computation Time (days)', fontsize=22, labelpad=20)
    ax.set_ylabel('RMSE', fontsize=22, labelpad=20)

    ax.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.8, 0.95), ncol=1, fancybox=True, shadow=False, fontsize=22)

    plt.savefig(OUTPUTPATH + outputname + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
        
        