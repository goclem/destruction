#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Computes predictions
@author:
@contact:
@version:
'''

#%% HEADER

# Modules
from destruction_utilities import *
import numpy as np
import zarr
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, AUC
import shutil
from tensorflow.keras.utils import Sequence

#%% 
# DECLARATION
CITY = 'test'
TILE_SIZE = (128,128)
THRESHOLD = 0.5


#%% 
# FUNCTIONS
def tiled_profile(source:str, tile_size:tuple=(*TILE_SIZE, 1)) -> dict:
    '''Computes raster profile for tiles'''
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size[0] == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size[1] == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size[0], affine[1], affine[2], affine[3], affine[4] * tile_size[1], affine[5])
    profile.update(width=profile['width'] // tile_size[0], height=profile['height'] // tile_size[0], count=tile_size[2], transform=affine)
    return profile

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

auc = AUC(
    num_thresholds=200,
    curve='ROC',
    name = 'auc'
)

class CNNTestGenerator(Sequence):
    def __init__(self, images, batch_size=32):
        self.images = images
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.images)//self.batch_size
    
    def __getitem__(self, index):
        if(index == self.__len__()-1):
            X = self.images[index*self.batch_size:len(self.images)]
        else:
            X = self.images[index*self.batch_size:(index+1)*self.batch_size]            
        return X
    
    def augment(self, X):        
        # Brightness
        alpha = random.choice(np.linspace(0.85, 1.4))
        X = X * alpha
        
        return X

#%% 
# LOAD MODEL
model_path = f'../models/{CITY}/cnn'
best_model = load_model(model_path, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'auc': auc})

#%% 
# PREDICT
images = search_data(pattern(city=CITY, type='image'))

for image in images:
    profile = tiled_profile(image, tile_size=(*TILE_SIZE, 1))
    name = image.split("image_")[1]
    image = np.array(read_raster(images[3]))
    h,w,b = image.shape
    image = image.reshape(1,h,w,b)
    image = tile_sequences(image, tile_size=TILE_SIZE)   
    image = image.reshape(image.shape[0], image.shape[2], image.shape[3], image.shape[4])
    shutil.rmtree("temp.zarr")
    zarr.save("temp.zarr", np.empty((0,image.shape[1], image.shape[2], image.shape[3])))
    temp = zarr.open("temp.zarr", mode = 'a')
    temp.append(image)
    
    test_images = CNNTestGenerator(temp)
    labels = best_model.predict_generator(test_images)
    labels = np.greater(labels, THRESHOLD).astype('int')
    
    
    write_raster(labels.reshape((profile['height'], profile['width'])), profile, f"../data/predictions/prediction_{name}")    
    print(name.split(".tif")[0])
    
    shutil.rmtree("temp.zarr")