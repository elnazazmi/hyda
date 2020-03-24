# hyda
Hydrological Data Analysis

## Requirements
- python >= 3.0
- packages to install
    - numpy
    - pandas
    - scipy
    - scikit-learn
    - hydroeval
    - matplotlib
    - bokeh
    - seaborn
    - progress
    
## Get started
#### main.py
The script calls the required functions to read the "total_storage" time series, extract features, reduce dimensionality,
 apply the clustering methods and visualize the output.

#### main_evol.py
The script calls the required functions to read the "discharge" and "forcing" data stored in the "data" directory,
 evaluate the evolutionary approach and visualize the output as plots.
 
#### clust_forc.py
The script contains definition of the functions to extract structure features from the input time series as well as
 dynamically added features in a simulation, and determine the number of clusters (K) using the elbow and rmse-ctime methods.

#### preprocessing
The directory contains functions' definitions for data preprocessing and evaluation.

#### clustering
The directory contains clustering functions' definitions.

#### visualization
The directory contains visualization functions' definitions.

#### data
The input and output data of all functions are available in the "data" directory.
 


