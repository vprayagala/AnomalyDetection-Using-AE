# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:52:03 2017

@author: vprayagala2

Prediction Class - Predict/Forecast Data points using LSTM Network

"""
#%%
import numpy as np
import pandas as pd
import logging

from sklearn.preprocessing import MinMaxScaler
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,LSTM
#from keras.callbacks import ModelCheckpoint, TensorBoard
#from keras import regularizers

from fbprophet import Prophet
#%%
class TSForecast:
    def __init__(self,logFile):
        self.logFile=logFile
        logging.basicConfig(filename=self.logFile,level=logging.DEBUG)
        
    # convert an array of values into a dataset matrix
    def scale_data(self,data,feature):
        new_feat=feature+"_Encoded"
        scaler=MinMaxScaler(feature_range=(0,1))
        data[new_feat]=scaler.fit_transform(data[feature].astype('float32').values.reshape(-1,1))
        return scaler,data
    
    #Create dataset with loop_back window
    def create_dataset(self,dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset.iloc[i:(i+look_back),]
            dataX.append(a)
            logging.debug(i)
            b=dataset.iloc[i+look_back,]
            logging.debug(b)
            dataY.append(dataset.iloc[i + look_back,])
        return np.array(dataX), np.array(dataY)
    
    def prepare_data_TS(self,data,freq='S'):
        min_date=min(data.index)
        max_date=max(data.index)
        logging.info(min_date)
        logging.info(max_date)
        index = pd.date_range(start=min_date,end=max_date,freq=freq)
        logging.info(index)
        
        ts=data.iloc[:,0].astype('float32')
        ts=ts.reindex(index,fill_value=0)
        #ts=ts.values
        logging.info(ts.shape)
        logging.info(type(ts))
        return ts
    # create and fit the LSTM network
    def build_model(self,layers=[16,8,4,1],look_back=1):
        model = Sequential()
        model.add(LSTM(layers[0], return_sequences=True,input_shape=(None,look_back)))
        model.add(LSTM(layers[1], return_sequences=True))
        model.add(LSTM(layers[2], return_sequences=False))
        model.add(Dense(layers[3]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def Prophet_Forecast(self,data):
        model=Prophet()
        model.fit(data)
        return model
    