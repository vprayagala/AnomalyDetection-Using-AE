# -*- coding: utf-8 -*-
#!/usr/bin/python
#%%
"""
Created on Thu Sep 14 11:32:40 2017

@author: vprayagala2

Purpose:
    This script is to pre-process Data Frame. It extracts the specific columns, converts types
    aggregates data, fills missing values
    
    Input: Pandas Data Frame
    Output : Processed Data Frame
    
Version History
1.0     -   First Version
2.0     -   Created Class and renamed the file
"""
#%%
#Import the packages
import logging
from time import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#%%
class PreProcessData:
    """Class Definition for PreProcessing Json File
        Supply Log file Name for creating Log for debug
        Methods: To Be Described
    """
    def __init__(self,logFile):
        self.logFile=logFile
        logging.basicConfig(filename=self.logFile,level=logging.DEBUG)
        
    def pre_process_data(self,data):
        #Extract the timestamp /Date 
        data['TS']=data.time.str[:20]
        data['TS'] = pd.to_datetime(data['TS'],format='%d/%b/%Y:%H:%M:%S')
        data['Date']=[d.date() for d in data['TS']]
        data['Hour']=[d.time().hour for d in data['TS']]
        data['Minute']=[d.time().minute for d in data['TS']]
        data['Seconds']=[d.time().second for d in data['TS']]
        return data
    
    def encode_feature(self,data,feature):
        le=LabelEncoder()
        
        new_col=str(feature)+"_Encoded"
        data[new_col]=le.fit_transform(data[feature])  
        
        return data,le
    
    def decode_feature(self,data,feature,encoder):
        new_col=str(feature) + "_Decoded"
        data[new_col]=encoder.inverse_transform(data[feature])  
        
        return data
    
    def extract_features(self,data,fields):
        t0=time()
        columns=data.columns
        columns_to_drop=[item for item in columns if item not in fields]
        data.drop(columns_to_drop,axis=1,inplace=True)
        logging.info("Feature Extraction Completed in %0.3fs" % (time() - t0))
        return data
    
    def aggregate_data(self,data,agg_by,agg_over,agg_fun): 
        #Drop columns not required
        t0=time()
        columns=data.columns
        data['DT'] = pd.to_datetime(data.Date) +data.Hour.astype('timedelta64[h]')+data.Minute.astype('timedelta64[m]')
        retain_columns=agg_by.copy()
        retain_columns.append('DT')
        retain_columns.append(agg_over)
        columns_to_drop=[item for item in columns if item not in retain_columns]
        logging.info("Dropping Columns {}".format(columns_to_drop))
        data.drop(columns_to_drop,axis=1,inplace=True)
       
        #Apply the function on the required fields
        logging.debug(data.dtypes)
        agg_columns=agg_by.copy()
        agg_columns.append('DT')
        logging.debug("Agg {} over {} by {}".format(agg_columns,agg_over,agg_fun))
        #dict1={agg_over:agg_fun}
        #logging.debug(dict1)
        processed_grp=data.groupby(by=agg_columns,as_index=False,sort=False)
        processed_agg=processed_grp[agg_over].agg(agg_fun).set_index('DT')
        logging.info("Feature Extraction Completed in %0.3fs" % (time() - t0))
        return processed_agg
    
    def aggregate_data_ts(self,data,fields,agg_on): 
        #Drop columns not required   
        t0=time()
        columns=data.columns
        columns_to_drop=[item for item in columns if item not in fields]
        logging.info("Dropping Columns {}".format(columns_to_drop))
        data.drop(columns_to_drop,axis=1,inplace=True)
        
        if agg_on == 'Date':
            data['DT'] = pd.to_datetime(data.Date)
        if agg_on == 'Hour':
            data['DT'] = pd.to_datetime(data.Date) +data.Hour.astype('timedelta64[h]')
     
        if agg_on == 'Minute':
            data['DT'] = pd.to_datetime(data.Date) +data.Hour.astype('timedelta64[h]')+data.Minute.astype('timedelta64[m]')
        if agg_on == 'Seconds':
            data['DT'] = pd.to_datetime(data.Date) +data.Hour.astype('timedelta64[h]')+data.Minute.astype('timedelta64[m]') +data.Seconds.astype('timedelta64[s]')
    
        print(data.head())
        data.drop(['Date','Hour','Minute','Seconds'],axis=1,inplace=True)
        processed=pd.DataFrame({'Count':data.groupby('DT').size()}).reset_index(level=['DT']).set_index(['DT'])
        logging.info("Aggregate TS completed in %0.3fs" % (time() - t0))    
        return processed
    
    def split_data(self,data,test_size,seed):
        X_train, X_test = train_test_split(data, test_size=test_size, random_state=seed)
        return X_train,X_test
#%%
#Custom Classes/Modules
#import Read_Json_File as RJF
#import pandas as pd
#data=RJF.read_data("C:\\Data\\LogAnalysis\\nginx_json_logs.json")
#RJF.data_summary(data)
##data_source=pd.DataFrame(data._source.values.tolist())
#data=pre_process_data(data)
#data_feat=extract_features(data,fields=['bytes','remote_ip','request','response',
#                                        'Date','Hour','Minute','Seconds'])
#data_agg=aggregate_data(data_feat,agg_on=['Date','Hour','Minute','remote_ip','response'],agg_fun=['size'])
#%%
#Commented Lines for Reference
#processed=pd.DataFrame({'Count':data.groupby(agg_on).size()}).reset_index(level=agg_on)
#processed=pd.DataFrame({'Count':data.groupby(agg_on).size()}).reset_index(level=agg_on).set_index(['Date','Hour','Minute'])
#processed=pd.DataFrame({'Count':data.groupby(agg_by).agg({agg_over:agg_fun})})
#processed=processed_agg.reset_index(level=['DT']).set_index(['DT'])
#processed.sort_values('Count',axis=0,ascending=False,inplace=True)
#processed.drop(['Hour','Minute'],axis=1,inplace=True)

#print('='*80)
#print("Top Rows:")
#print(processed.head())
#print("Pre-Processing Completed in %0.3fs" % (time() - t0))
    
