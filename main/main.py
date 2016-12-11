# -*- coding: utf-8 -*-
"""
Created by: Victor Peláez  09-12-2016
"""

import numpy as np 
import pandas as pd
#import kagglegym

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
import math



class model_fit():
    '''
    Date: 10/12/2016
    
    ################
    Version: v.0.0.1
    Comments: Initial approximation 
    
    '''
    
    def __init__(self, model, train, columns):

        self.model   = model
        self.columns = columns
        
        y = np.array(train.y)
        X = train[columns] # dataframe
        self.xMeans = X.mean(axis=0) 
        #self.xStd   = X.std(axis=0)  
        #X = np.array(X.fillna( self.xMeans ))
        #X = (X - np.array(self.xMeans))/np.array(self.xStd)
        
        # Observed with histograns: Approach found in kaggle kernels      
        low_y_cut = -0.086093
        high_y_cut = 0.093497
    
        y_is_above_cut = (train.y > high_y_cut)
        y_is_below_cut = (train.y < low_y_cut)
        y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
        
        
        X = np.array(X.fillna( self.xMeans )[y_is_within_cut].values)        
        y = np.array(train[y_is_within_cut].y)
        
        self.model.fit(X, y)
        
        return
    
    def predict(self, features):
        '''
        Prediction function
        '''
        X = features[self.columns]
        X = np.array(X.fillna( self.xMeans ))
        #X = (X - np.array(self.xMeans))/np.array(self.xStd)
        print(self.columns)

        return self.model.predict(X)
        
def shift_ifpossible(df, col, lag):
    '''
    Function shift_ifpossible(): creates an extra feature as a column with a lag or shift for a especified column
    Date: 10/12/2016
    
    ################
    Version: v.0.0.1
    Comments: 
    '''
    
    df_s = df[['id', 'timestamp', col]].sort(["id","timestamp"]) #important
    df_s['id_sh'] = df_s.id.shift(lag)
    df_s['ts_sh'] = df_s.timestamp.shift(lag)
    col_lag = col+'sh_'+str(lag)
    df_s[col_lag] = df_s[col].shift(lag)
    df_s[col_lag] = df_s.apply(lambda x: x[col_lag] if ((x.id == x.id_sh) & (x.timestamp == x.ts_sh + lag)) else np.nan, axis=1)
    return df_s[col_lag]        
        
        
if __name__ == "__main__":        
        
    train = pd.read_hdf(r"..\input\train.h5")  
    
    # preprocessing

    
    
    # feature selection 
    mode = 'tech_20'
    if mode == 'all_columns':
        c = train.columns
        columns = [col for col in c if not col.startswith('y')]
    if mode == 'reduce_columns':
        columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19'] 
    if mode == 'tech_20':
        columns = ['technical_20'] 
    
    # training
    print("Training...")
    
    rng = np.random.RandomState(1)
    models = [LinearRegression(), AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),
                              n_estimators=400, random_state=rng)]
    models_dict = {}
    y_pred = {}
    for m in models: 
        m_str = str(m) # model in string name
        models_dict[m_str] =  model_fit(m, train, columns)
        y_pred[m_str] = models_dict[m_str].predict(train) # predict training set 
        r2 = r2_score(train.y, y_pred[m_str])
        r = np.sign(r2) * math.sqrt(abs(r2))
        print(r)
    
    y_t = np.mean( np.array([ y_pred[str(models[0])], y_pred[str(models[1])]]), axis=0 )
    r2 = r2_score(train.y, y_t)
    r = np.sign(r2) * math.sqrt(abs(r2))
    print(r)
    




