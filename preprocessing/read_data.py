
__author__ = 'Elnaz Azmi'
__email__ = 'elnaz.azmi@kit.edu'
__status__ = 'Development'

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import netCDF4

# find input matlab files
def find_matfiles(path, suffix='.mat'):
    filenames = os.listdir(path)
    
    return [filename for filename in filenames if filename.endswith(suffix)]
    
# read catchmentoutlet discharge from matlab files    
def read_catmat(matfile):
    catoutletmat = loadmat(matfile)
    catoutlet = np.array([])
    for i in range(0, len(catoutletmat['catchmentoutlet'])):
        catoutlet = np.append(catoutlet, catoutletmat['catchmentoutlet'][i][0])
    
    return catoutlet

# read hillslopes discharge from matlab files 
def read_dismat(matfile, datetime_start='01.01.2015 00:00:00'):
    dismat = loadmat(matfile)
    indx = np.array([])
    for i in range(0, len(dismat['hsl_discharge'])):
        indx = np.append(indx, dismat['hsl_discharge'][i][0][0])

    colname = pd.date_range(datetime_start, periods=len(dismat['hsl_discharge'][0])-1, freq='5MIN').tolist()

    cols = np.zeros(shape=(len(dismat['hsl_discharge']),len(dismat['hsl_discharge'][0])-1))

    for i in range(0, len(dismat['hsl_discharge'])):
            for j in range(1, len(dismat['hsl_discharge'][i])):
                cols[i, j-1] = dismat['hsl_discharge'][i][j][0][0]
    df = pd.DataFrame(data=cols, index=indx, columns=colname)
    
    return df

# read forcing data from a matlab file
def read_forcmat(matfile):
    forcmat = loadmat(matfile)
    forcing = np.array([])
    for i in range(0, len(forcmat['ans'])):
        forcing = np.append(forcing, forcmat['ans'][i][0])
    
    return forcing

# read run time log data
def read_log(path):
    with open(path, "r") as file:
        df_rtime = pd.DataFrame(columns=['method', 'elapsed-time', 'start-time', 'end-time'])
        for line in file:
            t = line.split(' ; ')
            st = t[0].split('Start-time: ')[1]
            et = t[1].split('End-time: ')[1]
            elt = t[2].split(':', 2)[1] + ' days ' + t[2].split(':', 2)[2]
            met = t[3].split('\n')[0]
            df_rtime = df_rtime.append({'method':met, 'elapsed-time':elt, 'start-time':st, 'end-time':et},
                                       ignore_index=True, sort=False)
    df_rtime['elapsed-time'] = pd.to_timedelta(df_rtime['elapsed-time'])
    df_rtime['start-time'] = pd.to_datetime(df_rtime['start-time'])
    df_rtime['end-time'] = pd.to_datetime(df_rtime['end-time'])
    df_rtime = df_rtime.set_index(['method'])

    # calculate speedup over original
    df_speedup = pd.DataFrame()
    df_speedup['Speedup'] = df_rtime.loc['Original'][0].total_seconds() / df_rtime['elapsed-time'].dt.total_seconds()

    return df_rtime, df_speedup

# create dataframe of all matlab files
def df_outlet(path, datetime_start='01.01.2015 00:00:00'):
    matfiles = find_matfiles(path)
    df_outlet = pd.DataFrame()
    for i in range(0, len(matfiles)):
        outlet = read_catmat(path + matfiles[i])
        df_outlet[matfiles[i].split('.mat')[0]] = outlet
    date = pd.date_range(datetime_start, periods=df_outlet.shape[0], freq='5MIN').tolist()
    df_outlet = df_outlet.set_index(pd.to_datetime(date))
    
    return df_outlet

# read netcdf file
def read_nc(file):
    nf = netCDF4.Dataset(file, 'r')

    return nf

# read variales from netcdf file
def read_varnc(nc_file, var_name, hight):
    ncells = len(nc_file.variables[var_name][0][hight][:])
    ntimesteps = len(nc_file.variables[var_name])
    datetime = pd.date_range('05.11.2013 00:00:00', periods=len(nc_file.variables['time']), freq='12MIN')
    var_data = pd.DataFrame(columns=[str(i) for i in range(ncells)])
    for i in range(ntimesteps):
        var_data.loc[i] = nc_file.variables[var_name][i][hight][:].data
    var_data = var_data.set_index(datetime)

    return var_data