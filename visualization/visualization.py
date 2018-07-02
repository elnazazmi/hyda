
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import numpy as np
import pandas as pd
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
    p = sns.heatmap(df_Corr_pearson, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
    
    # Show the plot we reorient the labels for each column and row to make them easier to read.
    plt.yticks(rotation=0, fontsize=22) 
    plt.xticks(rotation=90, fontsize=22) 
    plt.savefig(OUTPUTPATH + outputname + '.pdf', bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)
    plt.clf()
    plt.cla()
    plt.close() 

def plot_rmse_ctime(df_Norm_Clust, outputname):
    fig, ax = plt.subplots(figsize=(13,8))
    ax_ = ax.twinx()

    ax.scatter(df_Norm_Clust['nr_cls'], df_Norm_Clust['rmse_sum'], label='_nolegend_', color='silver', edgecolor='gray', s=80)
    ax_.scatter(df_Norm_Clust['nr_cls'], df_Norm_Clust['ctime'], label='_nolegend_', color='silver', edgecolor='gray', s=80)

    x_smooth = np.linspace(df_Norm_Clust['nr_cls'].min(), df_Norm_Clust['nr_cls'].max(), 20)
    rmse_sum_tck = interpolate.splrep(df_Norm_Clust['nr_cls'], df_Norm_Clust['rmse_sum'], s=0)
    rmse_sum_smooth = interpolate.splev(x_smooth, rmse_sum_tck, der=0)

    ax.plot(x_smooth, rmse_sum_smooth, label='RMSE', color='r', linewidth=2)

    ctime_tck = interpolate.splrep(df_Norm_Clust['nr_cls'], df_Norm_Clust['ctime'], s=0)
    ctime_smooth = interpolate.splev(x_smooth, ctime_tck, der=0)

    ax_.plot(x_smooth, ctime_smooth, label='Representative Hillslopes Computation Time', color='g', linewidth=2)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.set_xlim(0.0) 
    ax.set_ylim(0.000000)
    ax_.set_ylim(0.000000)

    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax_.yaxis.set_tick_params(labelsize=18)
    ax.set_xlabel('Number of Clusters', fontsize=18, labelpad=20)
    ax.set_ylabel('RMSE Fraction Normalized by Min-Max', fontsize=18, labelpad=20)
    ax_.set_ylabel('Time Fraction Normalized by Min-Max', fontsize=18, labelpad=20)
    
    # ask matplotlib for the plotted objects and their labels
    pl, labels = ax.get_legend_handles_labels()
    pl_, labels_ = ax_.get_legend_handles_labels()
    ax_.legend(pl + pl_, labels + labels_, loc='upper center', bbox_to_anchor=(0.5, 1.2),
            ncol=1, fancybox=True, shadow=False, fontsize=18)

    plt.savefig(OUTPUTPATH + outputname + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    