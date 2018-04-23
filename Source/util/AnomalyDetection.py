# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:43:40 2017

@author: vprayagala2

Class for Anomaly
"""
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import json
from sklearn import svm
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras import regularizers
import pickle

class LoggingCallback(Callback):
    """
    Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)


class AnomalyDetection:
    def __init__(self,logger):
        self.logger=logger
        
    #Build AutoEncoder
    def build_nn_model(self,encoding_dim,input_dim):
        model = Sequential()

        model.add(Dense(encoding_dim, input_shape=(input_dim,),activation="tanh", 
                        activity_regularizer=regularizers.l1(10e-5)))
        model.add(Dense(int(encoding_dim / 2), activation="relu"))
        
        model.add(Dense(int(encoding_dim / 2), activation='tanh'))
        model.add(Dense(input_dim, activation='relu'))
        self.logger.info("Neural Net Model has been built")
        return model
    
    def compile_nn_model(self,model,
                         loss='mean_absolute_error',
                         metrics=['mae'],
                         optimizer="adam"):
        #Compile and train model
        #model=build_model(encoding_dim,input_dim)
       
        model.compile(optimizer=optimizer, 
                            loss=loss, 
                            metrics=metrics)
        self.logger.info("Neural Net Model has been compiled")
        return model
        
    def run_nn_model(self,model,train_data,validation_data,
                     nb_epoch,batch_size,
                     checkpoint_path,
                     log_dir):
        logger=LoggingCallback(logging.info)
        checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                       verbose=0,
                                       save_best_only=True)
        tensorboard = TensorBoard(log_dir=log_dir,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)
        
        md_train = model.fit(train_data, train_data,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(validation_data, validation_data),
                            verbose=0,
                            callbacks=[logger,checkpointer, tensorboard])
        
        self.logger.info("\nNeural Net Model has been trained successfully\n")                    
        return model,md_train.history
    
    def predict_nn_model(self,model,test_data):
        pred = model.predict(test_data)
        mae = np.mean(np.power(test_data - pred, 1), axis=1)
        result = pd.DataFrame({'Actual':test_data[0:,0],'Pred':pred[0:,0],'reconstruction_error': mae}
                                )
        return result

    def build_svm_model(self,train_data,nu=0.05,kernal='rbf',gamma=0.001):
        model = svm.OneClassSVM(nu=nu, kernel=kernal, gamma=gamma)
        model.fit(train_data)
        self.logger.info("OneClassSVM Model has been built")
        return model

    def save_svm_model(self,model,path):
        with open(path,'wb') as f:
            pickle.dump(model,f)

    def predict_svm_model(self,model,test_data):
        pred = model.predict(test_data)
        pred_val=model.decision_function(test_data)
        actual=test_data.values
        mae = np.mean(np.power(actual - pred_val, 1), axis=1).reshape(test_data.shape[0],1)
        result = pd.DataFrame(np.hstack((actual,pred_val,mae)),
                                         columns=['Actual','Pred','reconstruction_error'] )

        return pred,result
    
    def plot_nn_training_hist(self,history):
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
        
    def export_nn_training_result(self,history,file_name):
        with open(file_name,'w') as out_file:
            json.dump(history,out_file)
            
    def export_nn_result(self,threshold,error_df,test_data,out_path):
        
        err_mean=error_df['reconstruction_error'].mean()
        err_std= error_df['reconstruction_error'].std()
        lower_bnd=err_mean - (threshold*err_std)
        uper_bnd=err_mean+(threshold*err_std)
    
        normal=((error_df['reconstruction_error'] >= lower_bnd) &
                       (error_df['reconstruction_error'] <= uper_bnd))
   
        x=error_df.copy()
        x.set_index(test_data.index,inplace=True)
        file_name=out_path + "nn_pred.json"
        with open(file_name,'w') as out_file:
            out_file.write(x.to_json(orient='records', lines=True))
            
                
        error_df_normal = error_df.copy()
        error_df_abnormal = error_df.copy()
        error_df_normal[(normal==False)]=np.nan
        error_df_abnormal[(normal==True)]=np.nan
        
        file_name=out_path + "nn_normal.json"
        with open(file_name,'w') as out_file:
            out_file.write(error_df_normal.to_json(orient='records', lines=True))
        
        file_name=out_path + "nn_abnormal.json"
        with open(file_name,'w') as out_file:
            out_file.write(error_df_abnormal.to_json(orient='records', lines=True))

        return error_df_normal,error_df_abnormal
    
    def plot_nn_result(self,threshold,error_df,test_data):
        err_mean=error_df['reconstruction_error'].mean()
        err_std= error_df['reconstruction_error'].std()
        lower_bnd=err_mean - (threshold*err_std)
        uper_bnd=err_mean+(threshold*err_std)
        
        normal=((error_df['reconstruction_error'] >= lower_bnd) &
                           (error_df['reconstruction_error'] <= uper_bnd))
        #abnormal=((error_df['reconstruction_error'] < lower_bnd )|
        #                   (error_df['reconstruction_error'] > uper_bnd))
        
        x=error_df.copy()
        x.set_index(test_data.index,inplace=True)
        #Plot the Test vs Predictions
        fig,ax=plt.subplots()
        
        x['Actual'].plot()
        x['Pred'].plot()
        ax.set(title="True VS Predicted",xlabel="Time",ylabel="Aggregation Values")
        fig.autofmt_xdate()
        
        error_df_normal = error_df.copy()
        error_df_abnormal = error_df.copy()
        error_df_normal[(normal==False)]=np.nan
        error_df_abnormal[(normal==True)]=np.nan
        
        fig,ax=plt.subplots()
        plt.style.use('fivethirtyeight')
        ax.plot(error_df_normal.iloc[:,2],marker='o',label='normal')
        ax.plot(error_df_abnormal.iloc[:,2],marker='x',label='anomaly')
        ax.legend()
        ax.axhline(y=lower_bnd, color='green', linestyle='-')
        ax.axhline(y=uper_bnd, color='green', linestyle='-')
        ax.set(title="Normal VS Abnromal Based on Threshold",
                  ylabel="Aggregation Values")
        plt.grid(True)
        plt.show()
        
    def plot_svm_result(self,test,pred):
        # inliers are labeled 1, outliers are labeled -1
        normal = test[pred == 1]
        abnormal = test[pred == -1]
    
        fig,ax=plt.subplots()
        normal['response'].plot()
        abnormal['response'].plot()
        #ax.set_xticks(range(len(Y_)))
        #ax.set_xticklabels(Y_, rotation='vertical')
        plt.ylabel('Aggregated Counts')
        plt.grid(True)
        #for xy in zip(normal[['HOSTNAME_', 'Count']]):                                    
        #    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
        plt.show()
    
    def export_svm_result(self,test,pred,out_path):
        normal = test[pred == 1]
        abnormal = test[pred == -1]
        
        file_name=out_path+"svm_normal.json"
        with open(file_name,'w') as out_file:
            out_file.write(normal.to_json(orient='records', lines=True))
        
        file_name=out_path+"svm_abnormal.json"
        with open(file_name,'w') as out_file:
            out_file.write(abnormal.to_json(orient='records', lines=True))

        return normal,abnormal

    def load_nn_model(self, path):
        # Load Model and Predict Classes/Probability
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(path)
        self.logger.info("Loaded model from disk")

        loaded_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    def load_svm_model(self,path):
        with open(path,'rb') as f:
            model=pickle.load(f)
            return model