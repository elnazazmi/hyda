
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

OUTPUTPATH = 'data/'

def plot_timeseries(data, outputname):
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b') 

    n = data.shape[1]
    fig, ax = plt.subplots(figsize=(12,8))

    color=iter(cm.rainbow(np.linspace(0,1,n)))

    for i in range(1, n):
        c=next(color)
        ax.plot(data['date'], data.iloc[:, i], color=c, linewidth=2)

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