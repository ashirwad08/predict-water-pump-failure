import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import seaborn as sbrn
import numpy as np
#import re
#import datetime
import pickle
#import sklearn

def imputeTrain(train):
  """Function takes in a dataset for the "water pump failure" drivendata.org competition 
  and returns a tuple of two items:
  1. A dataframe that contains columns to impute, namely:
      - gps_height
      - population
      - latitude
      - longitude
      - construction_year
      *Note: An exception will be thrown if any one of these columns are missing
      *Note: Columns do not need to contain 'NaN' values. The function will replace 
      zeroes with NaNs as well as erroneous lat, long values
      
  2. A nested dictionary in the following format that contains trained imputed values for
     each variable above, by a heirarchical geography. The intent is to use this nested
     dictionary to inform unseen test observations during prediction.
  """
  
  imputeCols = ['gps_height','population','latitude','longitude','construction_year', 'sub-village','ward','lga','region_code']
  
  assert (imputeCols in train.columns), raise Exception('Missing Columns! Please make sure all of the following columns are in your training frame: \n' + imputeCols)
  
  
  #replace continuous predictor missing values (0s) with NaN
  train.population.replace({0:np.nan}, inplace=True)   
  train.gps_height.replace({0:np.nan}, inplace=True)
  train['construction_year']=train['construction_year'].astype('int64')
  train.loc[train.construction_year==0,['construction_year']]=np.nan
  
  #replace lat/long outliers with NaN; replace in plce won't work for multiple columns
  train.loc[((train.longitude==0)&(train.latitude==-2.000000e-08)),['latitude','longitude']]=train.loc[((train.longitude==0)&(train.latitude==-2.000000e-08)),['latitude','longitude']].replace({'latitude':{-2.000000e-08:np.nan}, 'longitude':{0.0:np.nan}}, regex=False)
  
  
  #now, impute NaNs with the mean of hierarchical geographies going from nearest to farthest:
  #sub-village > ward > lga > region_code
  
  #population
  dat.population.fillna(dat.groupby(['subvillage'])['population'].transform('mean'), inplace=True)
  dat.population.fillna(dat.groupby(['ward'])['population'].transform('mean'), inplace=True)
  dat.population.fillna(dat.groupby(['lga'])['population'].transform('mean'), inplace=True)
  dat.population.fillna(dat.groupby(['region_code'])['population'].transform('mean'), inplace=True)
  
  
  
  

  
