# -*- coding: utf-8 -*-
#!/usr/bin/python
#%%
"""
Created on Thu Sep 14 10:47:10 2017

@author: vprayagala2

Purpose:
    This script is to read a json file and return pandas data frame
    Input: Json file with absolute path
    Output : Pandas Dataframe

Version History
1.0     -   First Version
2.0     -   Created OO Structure
"""
#%%
#Import the required python packages
import os as os
import math
import pandas as pd
from time import time
import json
from itertools import islice
import logging
#%%
#Define Class/Functions
class ReadJsonFile:
    """Class Definition for Reading Json File
        Supply Log file Name for creating Log for debug
        Methods: read_data
                    input - Json File Name Abolsute Path
                    Output - Data Frame
                get_meta_data
                    input - Json File Name Abolsute Path
                    output - Json Object with Column Headers
    """
    def __init__(self,logFile):
        self.logFile=logFile
        #f=open(self.logFile,'w')
        #f.close()
        logging.basicConfig(filename=self.logFile,level=logging.DEBUG)
      
        
    def read_data(self,file_name):
        """Read Json Using Pandas read_json Method
            input - Json File Name Abolsute Path
            Output - Data Frame
        """
        #Set the directory and file to be read
        if not os.path.exists(file_name):
            logging.error("File Not Found, Please Input Absolute path for the file")
        else:
            file_stats=os.stat(file_name)
            if file_stats.st_size > 2*math.pow(10,9):
                logging.error("Cannot Handle More than 2GB data currently")
        #Time at start of reading file        
        t0=time()
        data=pd.read_json(file_name,lines=True)
        logging.info('='*80)
        logging.info("File was read in %0.3fs" % (time() - t0))
        logging.info('='*80)
        self.__data_summary(data)
        return data
        
    def __data_summary(self,data):
        logging.info('='*80)
        logging.info(data.shape)
        logging.info(data.dtypes)
        logging.info('='*80) 
        
    def get_meta_data(self,file_name):
        
        df=self.get_sample_data(file_name,0,6)
        json_meta_data=json.dumps({"Column_Headers":list(df.columns)})
        return json_meta_data
    
    def get_sample_data(self,file_name,start=0,end=100):
        #result=pd.DataFrame()
        if (end - start) > 1000:
            print("Cannot get Sample size grater than 1000 records\n")
        else:
            f=open(file_name)
            chunk = list(islice(f,start,end))
            #loading the json file content in data list
            if len(chunk) == 0:
                print("Invalid Start/End positions, cannot find data\n")
            else:
                data=[]
                for line in chunk:
                    data.append(json.loads(line))
                df=pd.DataFrame.from_dict(data,orient='columns')
                #result=df.copy()
                return df
#%%
#Test Functions
#in_file="C:\\Data\\LogAnalysis\\nginx_json_logs.json"
#data=read_data("C:\\Data\\LogAnalysis\\nginx_json_logs.json")
#data_summary(data)
#col=get_metadata("C:\\Data\\LogAnalysis\\nginx_json_logs.json")
#print("Type:{}".format(type(col)))
#print(col)
#samp_data=get_sample_data(in_file)
#print(samp_data.head())
