# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:14:30 2016

@author: victor
"""

import kagglegym
import numpy as np
import pandas as pd

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
        
        #y = np.array(train.y)
        X = train[columns] # dataframe
        self.xMeans = X.mean(axis=0) 
        X = X.fillna( self.xMeans )
        #self.xStd   = X.std(axis=0)  
        #X = np.array(X.fillna( self.xMeans ))
        #X = (X - np.array(self.xMeans))/np.array(self.xStd)
        
        # Observed with histograns: Approach found in kaggle kernels      
        low_y_cut = -0.086093
        high_y_cut = 0.093497
    
        y_is_above_cut = (train.y > high_y_cut)
        y_is_below_cut = (train.y < low_y_cut)
        y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
        
        X = np.array(X[y_is_within_cut].values)        
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

        return self.model.predict(X)

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
columns= ['technical_20']
low_y_cut = -0.086093
high_y_cut = 0.093497


print("Training...")
rng = np.random.RandomState(1)
models = [LinearRegression(), AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),
                              n_estimators=100, random_state=rng)]
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

print("Predicting...")
while True:
    
    y_1 = models_dict[str(models[0])].predict(observation.features) 
    y_2 = models_dict[str(models[1])].predict(observation.features) 
    y_t = np.mean( np.array([ y_1, y_2 ]), axis=0 )
    observation.target.y = y_t.clip(low_y_cut, high_y_cut)
    
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)