"""
Author : 
    Ahmed Telili
"""
from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os 
from tensorflow_addons.metrics import RSquare
from keras import backend as K
from tqdm.keras import TqdmCallback
from scipy.stats import spearmanr
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from statistics import mean
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import pandas as pd
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau ,Callback,TensorBoard
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications 
import PIL
from keras.activations import softmax,sigmoid
import h5py
from PIL import Image
from keras.layers import Layer
from scipy.stats import spearmanr,pearsonr
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D ,Dense,Concatenate ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D,AveragePooling2D,Lambda,MaxPooling2D,TimeDistributed, Bidirectional, LSTM
import argparse
import random
from tqdm import tqdm
from sklearn.metrics import r2_score
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras import initializers

import time

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


tf.keras.backend.clear_session()
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ['CUDA_VISIBLE_DEVICES']=""



def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

'''
def data_generator_1(data,batch_size=4):              

    num_samples = len(data)

    while True:   
        for offset in range(0, num_samples, batch_size):
          
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            X_train = np.zeros((batch_size, 30,25,2560))
            y_train = np.zeros((batch_size,1))
            for i in range(batch_size):
              X_train[i,:,:,:] = np.load(batch_samples[i][0])
              y_train[i,:] = np.load(batch_samples[i][1])
            yield X_train


def data_generator_2(data,batch_size=1):              

    num_samples = len(data)

    while True:   
        for offset in range(0, num_samples, batch_size):
          
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            X_train = np.zeros((batch_size, 30,25,2560))
            y_train = np.zeros((batch_size,1))
            for i in range(batch_size):
              X_train[i,:,:,:] = np.load(batch_samples[i][0])
              y_train[i,:] = np.load(batch_samples[i][1])
            yield y_train
'''



def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
                    
def build_model(batch_shape):



  model = models.Sequential()
  model.add(TimeDistributed(LSTM(32,return_sequences=True, 
                               kernel_initializer='random_normal', 
                               recurrent_initializer='random_normal', dropout=0.4,recurrent_dropout=0), input_shape = batch_shape))
 
  #model.add(TimeDistributed(Bidirectional(LSTM(156,return_sequences=True,kernel_initializer='random_normal',recurrent_initializer='random_normal', dropout=0.4,recurrent_dropout=0))))

  model.add(TimeDistributed(Flatten()))

  #model.add(Bidirectional(LSTM(16,return_sequences=True,  
                               #kernel_initializer='random_normal', 
                               #recurrent_initializer='random_normal', dropout=0.4,recurrent_dropout=0)))
  model.add(LSTM(16,return_sequences=True, 
                               kernel_initializer='random_normal', 
                               recurrent_initializer='random_normal', dropout=0.4,recurrent_dropout=0))
  model.add(Flatten())                        
  #model.add(Dense(512,activation='relu',kernel_initializer='random_normal'))
  #model.add(layers.Dropout(rate=0.5))
  model.add(Dense(32,activation='relu',kernel_initializer='random_normal'))
  model.add(layers.Dropout(rate=0.5))
  model.add(layers.Dense(1, activation = 'relu', kernel_initializer='random_normal'))

  model.compile(optimizer=optimizers.Adam(), loss=root_mean_squared_error ,metrics='mae')
  model.summary()
  return model







if __name__ == '__main__':
    parser = argparse.ArgumentParser("End2End_train")


    parser.add_argument('-nf',
        '--num_frames',
        default=30,
        type=int,
        help='Number of cropped frames per video.'
    )

    parser.add_argument('-nb',
        '--num_frames',
        default=16,
        type=int,
        help='Number of cropped frames per video.'
    )
    parser.add_argument('-np',
        '--num_frames',
        default=156,
        type=int,
        help='Number of cropped patches per frame.'
    )


    parser.add_argument('-b',
        '--batch_size',
        default=1,
        type=int,
        help='batch_size.'
    )


    if not os.path.exists('./models'):
    	os.makedirs('./models')

    args = parser.parse_args()

    idx = [7]

    #prepare data
    l = os.listdir('./label')
    l.sort()
    lab_y = np.zeros((196,3))
    for i in range(len(l)):
        feat = np.load('./label/' + l[i])
        lab_y[i,:] = feat
    np.save('label.npy',dens_X)
    l = os.listdir('./features_X')
    l.sort()
    dens_X = np.zeros((196,2048)) # 2048 dim resnet50 output, change it with your own backbone model output shape
    for i in range(len(l)):
        feat = np.load('./features_X/' + l[i])
        dens_X[i,:] = feat
    np.save('features_X.npy',dens_X)




    X = np.load('./features_X.npy')
    y = np.load('./label.npy')
    label_P3 = y[:,2]
    label_P2 = y[:,1]
    label_P1 = y[:,0]

    score_r2_P3 = []
    score_srocc_P3 = []
    score_plcc_P3 = []
    score_r2_P2 = []
    score_srocc_P2 = []
    score_plcc_P2 = []
    score_r2_P1 = []
    score_srocc_P1 = []
    score_plcc_P1 = []

    start_time = time.time()
    for i in idx:
    	md = ModelCheckpoint(filepath='./models/vmaf/model.h5',monitor='val_loss', mode='auto',save_weights_only=False,save_best_only=True,verbose=1)
    	rd = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15,min_lr=1e-11, verbose=1, mode='min')
    	ear = EarlyStopping(monitor='val_loss',mode ='min', patience=50, verbose=1,restore_best_weights=False)
    	callbacks_k = [md,rd,ear] 	
    	tf.keras.backend.clear_session()

    	X_train, X_test, y_train, y_test = train_test_split(X, label_P1, test_size=0.2, random_state=i)
    	num_patch = 156
    	nb = 16
    	batch_size = args.batch_size

    	model = build_model((nb,num_patch,2048)) # 2048 dim resnet50 output, change it with your own backbone model output shape

    	history = model.fit(X_train,y_train,
    		epochs=200,validation_data=(X_test,y_test),verbose=2, shuffle= True, callbacks=[md,ear,rd], batch_size=4)

    	del model 

    	model = load_model('./models/vmaf/model.h5',compile=False)

    	y_p = model.predict(X_test)

    	del model

    	y_test = y_test.reshape(39)
    	y_p = y_p.reshape(39)



    	score_r2_P1.append(r2_score(y_test,y_p))
    	score_srocc_P1.append(spearmanr(y_p, y_test)[0])
    	score_plcc_P1.append(pearsonr(y_p, y_test)[0])




    end_time = time.time()
    print('time is ', end_time-start_time)
    print('r2 = ',score_r2_P1)
    print('srocc = ', score_srocc_P1)
    print('plcc = ' ,score_plcc_P1)


    	


