import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import seaborn as sbrn
import numpy as np
#import re
#import trainetime
import pickle
#from collections import OrderedDict
#import sklearn

def imputeTrain(trn):

  """
  Input: Training dataset
  Output: Returns copy of imputed training set; and a reference map (nested dictionary)
  
  Function takes in a trainaset for the "water pump failure" driventraina.org competition 
  and returns a list of two items:
  1. A training dataframe that contains imputed columns, namely:
      - gps_height
      - population
      - latitude
      - longitude
      - construction_year
      *Note: An exception will be thrown if any one of these columns are missing
      *Note: Columns do not need to contain 'NaN' values. The function will replace 
      zeroes with NaNs as well as erroneous lat, long values
      *Note: Uses a heirarchical geographically nearest neighbors mean measure 
      
  2. A nested dictionary in the following format that contains trained imputed values for
     each variable above, by a heirarchical geography. The intent is to use this nested
     dictionary to inform unseen test observations during prediction.
  """
  
  train = trn.copy()
  
  imputeCols = ['gps_height','population','latitude','longitude','construction_year', 'subvillage','ward','lga','region_code']
  
  imputeMap = {'population':{'subvillage':{},'ward':{},'lga':{},'region_code':{}}, 
               'gps_height':{'subvillage':{},'ward':{},'lga':{},'region_code':{}},
               'construction_year':{'subvillage':{},'ward':{},'lga':{},'region_code':{}},
               'latitude':{'subvillage':{},'ward':{},'lga':{},'region_code':{}},
               'longitude':{'subvillage':{},'ward':{},'lga':{},'region_code':{}} }
  
  
  
  exception = 'Missing Columns! Please make sure all of the following columns are in your training frame: \n'+str(imputeCols)
  
  if not set(imputeCols) < set(list(train.columns)):
   raise Exception(exception)
  
  
  #replace continuous predictor missing values (0s) with NaN
  train.population.replace({0:np.nan,1:np.nan,2:np.nan}, inplace=True)   
  train.gps_height.replace({0:np.nan}, inplace=True)
  train['construction_year']=train['construction_year'].astype('int64')
  train.loc[train.construction_year==0,['construction_year']]=np.nan
  
  #replace lat/long outliers with NaN; replace in plce won't work for multiple columns
  train.loc[((train.longitude==0)&(train.latitude==-2.000000e-08)),['latitude','longitude']]=train.loc[((train.longitude==0)&(train.latitude==-2.000000e-08)),['latitude','longitude']].replace({'latitude':{-2.000000e-08:np.nan}, 'longitude':{0.0:np.nan}}, regex=False)
  
  
  #now, impute NaNs with the mean of hierarchical geographies going from nearest to farthest:
  #sub-village > ward > lga > region_code
  
  #population
  #first, store location mean per location unit
  imputeMap=generateMap('subvillage','population',train,imputeMap)
  train.population.fillna(train.groupby(['subvillage'])['population'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('ward','population',train,imputeMap)
  train.population.fillna(train.groupby(['ward'])['population'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('lga','population',train,imputeMap)
  train.population.fillna(train.groupby(['lga'])['population'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('region_code','population',train,imputeMap)
  train.population.fillna(train.groupby(['region_code'])['population'].transform('mean'), inplace=True)
  
  
  
  #gps_height (do the same thing)
  imputeMap=generateMap('subvillage','gps_height',train,imputeMap)
  train.gps_height.fillna(train.groupby(['subvillage'])['gps_height'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('ward','gps_height',train,imputeMap)
  train.gps_height.fillna(train.groupby(['ward'])['gps_height'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('lga','gps_height',train,imputeMap)
  train.gps_height.fillna(train.groupby(['lga'])['gps_height'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('region_code','gps_height',train,imputeMap)
  train.gps_height.fillna(train.groupby(['region_code'])['gps_height'].transform('mean'), inplace=True)
  
  
  #construction_year (same! just set construction year back to int64 at the end)
  imputeMap=generateMap('subvillage','construction_year',train,imputeMap)
  train.construction_year.fillna(train.groupby(['subvillage'])['construction_year'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('ward','construction_year',train,imputeMap)
  train.construction_year.fillna(train.groupby(['ward'])['construction_year'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('lga','construction_year',train,imputeMap)
  train.construction_year.fillna(train.groupby(['lga'])['construction_year'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('region_code','construction_year',train,imputeMap)
  train.construction_year.fillna(train.groupby(['region_code'])['construction_year'].transform('mean'), inplace=True)
  
  train['construction_year']=train.construction_year.astype('int64') #set to int! or we'll have too many
  
  
  #same for lats and longs
  imputeMap=generateMap('subvillage','latitude',train,imputeMap)
  train.latitude.fillna(train.groupby(['subvillage'])['latitude'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('ward','latitude',train,imputeMap)
  train.latitude.fillna(train.groupby(['ward'])['latitude'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('lga','latitude',train,imputeMap)
  train.latitude.fillna(train.groupby(['lga'])['latitude'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('region_code','latitude',train,imputeMap)
  train.latitude.fillna(train.groupby(['region_code'])['latitude'].transform('mean'), inplace=True)
  
  
  #long
  imputeMap=generateMap('subvillage','longitude',train,imputeMap)
  train.longitude.fillna(train.groupby(['subvillage'])['longitude'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('ward','longitude',train,imputeMap)
  train.longitude.fillna(train.groupby(['ward'])['longitude'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('lga','longitude',train,imputeMap)
  train.longitude.fillna(train.groupby(['lga'])['longitude'].transform('mean'), inplace=True)
  
  imputeMap=generateMap('region_code','longitude',train,imputeMap)
  train.longitude.fillna(train.groupby(['region_code'])['longitude'].transform('mean'), inplace=True)
  

  
  return train, imputeMap
  
  
  
  
  
  
  
def generateMap(geog, col, train, imputeMap):
  """helps the imputeTrain function out by storing the means of each location breakdown
  for that column in the nested dictionary"""
  grpdf = train.groupby(train[geog])[col].mean().reset_index()
  grpdf = grpdf.loc[~grpdf[col].isnull()]
  grpdf.set_index(grpdf.iloc[:,0], inplace=True)
  grpdf.drop(geog, inplace=True, axis=1)
  
  #insert into nested dict
  imputeMap[col][geog].update(grpdf.iloc[:,0].to_dict())
  
  return imputeMap
  





def fillTest(tst, imputeMap):
  """
  Inputs: Test dataframe, reference map nested dictionary
  Outputs: Copy of Test dataframe with filled in trained values.
  
  uses a passed in reference map that contains trained means by geographical
  nearness for numerics 
      - gps_height
      - population
      - latitude
      - longitude
      - construction_year. 
      
      Function returns the passed in test dataframe with any missing values filled 
      in according to the reference map.
   
   *Note: if input dataframe is sorted in any order the order will be lost as 
   missing values are removed, filled in, and appended back to the dataframe. 
   Simply re-sort if original order is desired.
  """
   
  test_imp=tst.copy()
  
  imputeCols = ['gps_height','population','latitude','longitude','construction_year', 'subvillage','ward','lga','region_code']
  
  exception = 'Missing Columns! Please make sure all of the following columns are in your test frame: \n'+str(imputeCols)
  numCols = ['gps_height','population','latitude','longitude','construction_year']
   
  if not set(imputeCols) < set(list(test_imp.columns)):
   raise Exception(exception)

  geogHierarch = np.array(['subvillage','ward','lga','region_code'])

  #replace continuous predictor missing values (0s) with NaN
  test_imp.population.replace({0:np.nan, 1:np.nan, 2:np.nan}, inplace=True)   
  test_imp.gps_height.replace({0:np.nan}, inplace=True)
  test_imp['construction_year']=test_imp['construction_year'].astype('int64')
  test_imp.loc[test_imp.construction_year==0,['construction_year']]=np.nan

  #replace lat/long outliers with NaN; replace in plce won't work for multiple columns
  test_imp.loc[((test_imp.longitude==0)&(test_imp.latitude==-2.000000e-08)),['latitude','longitude']]=test_imp.loc[((test_imp.longitude==0)&(test_imp.latitude==-2.000000e-08)),['latitude','longitude']].replace({'latitude':{-2.000000e-08:np.nan}, 'longitude':{0.0:np.nan}}, regex=False)



  #BACKUP IMPUTE STRATEGY: NOT USING REFERENCE MAP
  """
  test.gps_height.fillna(test.groupby(['subvillage'])['gps_height'].transform('mean'), inplace=True)
  test.gps_height.fillna(test.groupby(['ward'])['gps_height'].transform('mean'), inplace=True)
  test.gps_height.fillna(test.groupby(['lga'])['gps_height'].transform('mean'), inplace=True)
  test.gps_height.fillna(test.groupby(['region_code'])['gps_height'].transform('mean'), inplace=True)

  test.population.fillna(test.groupby(['subvillage'])['population'].transform('mean'), inplace=True)
  test.population.fillna(test.groupby(['ward'])['population'].transform('mean'), inplace=True)
  test.population.fillna(test.groupby(['lga'])['population'].transform('mean'), inplace=True)
  test.populationr.fillna(test.groupby(['region_code'])['population'].transform('mean'), inplace=True)
  
  test.construction_year.fillna(test.groupby(['subvillage'])['construction_year'].transform('mean'), inplace=True)
  test.construction_year.fillna(test.groupby(['ward'])['construction_year'].transform('mean'), inplace=True)
  test.construction_year.fillna(test.groupby(['lga'])['construction_year'].transform('mean'), inplace=True)
  test.construction_year.fillna(test.groupby(['region_code'])['construction_year'].transform('mean'), inplace=True)
  
  test.latitude.fillna(test.groupby(['subvillage'])['latitude'].transform('mean'), inplace=True)
  test.latitude.fillna(test.groupby(['ward'])['latitude'].transform('mean'), inplace=True)
  test.latitude.fillna(test.groupby(['lga'])['latitude'].transform('mean'), inplace=True)
  test.latitude.fillna(test.groupby(['region_code'])['latitude'].transform('mean'), inplace=True)
  
  test.longitude.fillna(test.groupby(['subvillage'])['longitude'].transform('mean'), inplace=True)
  test.longitude.fillna(test.groupby(['ward'])['longitude'].transform('mean'), inplace=True)
  test.longitude.fillna(test.groupby(['lga'])['longitude'].transform('mean'), inplace=True)
  test.longitude.fillna(test.groupby(['region_code'])['longitude'].transform('mean'), inplace=True)
  """  

  df_id = test_imp[['id']]
  test = test_imp
  
  for col in numCols:
    if test[col].isnull().sum():
      #subset ad remove from test frame col specific nulls (will append filled values later)
      test_sub = test[test[col].isnull()]
      test = test[~test[col].isnull()]
      
      #fill in missing values by tiered geography
      test_filled = test_sub[~test_sub[col].isnull()] #empty at first
      for geog in geogHierarch:
        #get col and geog specific reference map
        refdf = extractMap(imputeMap, col, geog)
        
        #now merge col and geog missing values in test with ref map
        test_sub=pd.merge(test_sub, refdf, how='left', on=geog)
        test_sub[col+'_x']=test_sub[col+'_y']
        test_sub.drop(col+'_y', axis=1, inplace=True)
        test_sub=test_sub.rename(columns={col+'_x':col}) #remove _x
        
        #get all non NaNs from test_sub
        test_filled = pd.concat([test_filled,test_sub[~test_sub[col].isnull()]], axis=0)
        test_sub = test_sub[test_sub[col].isnull()]
        
        if test_sub.shape[0]==0:
          break
        
      #merge filled set and any remaining (could not fill) back to Test
      test = pd.concat([test, test_filled, test_sub], axis=0, ignore_index=True)
  
  
  
  #make sure construction year is an integer col    
  test['construction_year']=test['construction_year'].astype('int64')
  
  
  df_merge = pd.merge(df_id, test, on='id')
    
  return df_merge





def extractMap(imap, col, geog):
  """
  Extract impute column and geography specific values from trained reference map.
  Returns a reference dataframe, with columns col, geog.
  """
  
  #extract col and geog specific values from reference map as dataframe
  mapdf = pd.DataFrame()
  mapdf = mapdf.from_dict(imap[col][geog],orient='index') 
  mapdf[geog]=mapdf.index
  mapdf.columns=[col,geog]
  
  return mapdf
