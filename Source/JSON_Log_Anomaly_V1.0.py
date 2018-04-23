
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os as os
from time import time
import math

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model,Sequential
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#Global Parameters
RANDOM_STATE=77
THRESHOLD=3 #2*STD away from mean will be considered as anomaly

MODEL_PATH="C:\\Data\\Log Ananlysis\\model_log.h5"
TB_LOG_DIR="C:\\Data\\Log Ananlysis\\logs"


# In[ ]:


#Use utility function and read the data
import sys
sys.path.append("C:\\git\\projects\\patternsearch\\POC2\\util")


# In[ ]:


#Custom Classes/Modules
import Read_Json_File as RJF
import Pre_Process_Data as PP
data=RJF.read_data("C:\\Data\\LogAnalysis\\nginx_json_logs.json")
RJF.data_summary(data)

data_process=PP.pre_process_data(data)
data_feat=PP.extract_features(data_process,fields=['bytes','remote_ip','request','response',
                                        'Date','Hour','Minute','Seconds'])
data_agg=PP.aggregate_data(data_feat,agg_on=['Date','Hour','Minute','remote_ip','response'],agg_fun=['size'])


# In[ ]:


#Encode Features 
data_agg,le_ip=PP.encode_feature(data_agg,'remote_ip')
data_agg.drop(['remote_ip'],axis=1,inplace=True)
print(data_agg.head())


# In[ ]:


#Convert hostname to features
def split_data(data,test_size):
    X_train, X_test = train_test_split(data, test_size=test_size, random_state=RANDOM_STATE)
    X_train = X_train.values
    X_test = X_test.values
    return X_train,X_test

#Build AutoEncoder
def build_model(encoding_dim,input_dim):
    model = Sequential()
    #input_layer = Input(shape=(input_dim, ))
    model.add(Dense(encoding_dim, input_shape=(input_dim,),activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5)))
    model.add(Dense(int(encoding_dim / 2), activation="relu"))
    
    model.add(Dense(int(encoding_dim / 2), activation='tanh'))
    model.add(Dense(input_dim, activation='relu'))
    #autoencoder = Model(inputs=input_layer, outputs=decoder)
    return model


# In[ ]:


#Split data into train/test
test_columns=list(data_agg.columns)
ix=test_columns.index("Count")
train,test=split_data(data_agg,test_size=0.1)
X_train=train[:,ix]
X_test=test[:,ix]
X_train=X_train.reshape(X_train.shape[0],1)
X_test=X_test.reshape(X_test.shape[0],1)


# In[ ]:


#Compile and train model
encoding_dim = int(math.log(X_train.shape[0]))
input_dim = X_train.shape[1]
model=build_model(encoding_dim,input_dim)

nb_epoch = int(math.pow(X_train.shape[0],3/5))
batch_size = int(math.sqrt(X_train.shape[0]))
model.compile(optimizer='adam', 
                    loss='mean_absolute_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=MODEL_PATH,
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir=TB_LOG_DIR,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = model.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


# In[ ]:


#Plot model training data
plt.subplot(2,1,1)
plt.title('Model Training Statistics')
plt.legend(['train', 'test'], loc='best')
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.ylabel("Accuracy")


plt.subplot(2,1,2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()


# In[ ]:


#Load the model and test
autoencoder = load_model(MODEL_PATH)
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 1), axis=1)
error_df = pd.DataFrame({'Actual':X_test[0:,0],'Pred':predictions[0:,0],'reconstruction_error': mse})


# In[ ]:


#plot 2 std away
err_mean=error_df['reconstruction_error'].mean()
err_std= error_df['reconstruction_error'].std()
lower_bnd=err_mean - (THRESHOLD*err_std)
uper_bnd=err_mean+(THRESHOLD*err_std)
#normal=error_df.loc[(error_df['reconstruction_error'] >= lower_bnd) &
#                   (error_df['reconstruction_error'] <= uper_bnd)]
#abnormal=error_df.loc[(error_df['reconstruction_error'] < lower_bnd )|
#                   (error_df['reconstruction_error'] > uper_bnd)]
normal=((error_df['reconstruction_error'] >= lower_bnd) &
                   (error_df['reconstruction_error'] <= uper_bnd))
abnormal=((error_df['reconstruction_error'] < lower_bnd )|
                   (error_df['reconstruction_error'] > uper_bnd))


# In[ ]:


#print(normal.shape)
#print(abnormal.shape)
error_df_normal = error_df.copy()
error_df_abnormal = error_df.copy()
error_df_normal[(normal==False)]=np.nan
error_df_abnormal[(normal==True)]=np.nan


# In[ ]:


plt.style.use('fivethirtyeight')
#plt.plot(normal.iloc[:,1], color='b')
#plt.plot(abnormal.iloc[:,1], color='r')
plt.plot(error_df_normal.iloc[:,2],marker='o',label='normal')
plt.plot(error_df_abnormal.iloc[:,2],marker='x',label='anomaly')
plt.legend()
plt.axhline(y=lower_bnd, color='green', linestyle='-')
plt.axhline(y=uper_bnd, color='green', linestyle='-')


# In[ ]:


res_df=pd.DataFrame(test[normal==False],columns=test_columns)
res_df.plot(kind='bar',
           x='remote_ip_Encoded',
           y='Count',style='line')
res_df=PP.decode_feature(res_df,'remote_ip_Encoded',le_ip)
#res_df.drop(['ip'],axis=1,inplace=True)
#print(res_df.head())


# In[ ]:


#Plot the anomaly points along with decoded host values, stacked bar chart on response types and counts on y axis

(res_df.pivot_table(index='remote_ip_Encoded_Decoded', columns='response', values='Count',
                 aggfunc='sum',fill_value=0)
   .plot.bar(stacked=True,title="Anomaly Data Points")
    )

