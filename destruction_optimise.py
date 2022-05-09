#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Optimises models
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.09
'''

#%% HEADER

# Modules
from timeit import repeat
import numpy as np
import pandas as pd
import itertools

from destruction_models import *
from destruction_utilities import *
from tensorflow.keras import callbacks, preprocessing
from os import path

# Paths
paths = dict(aleppo='../data/aleppo')

#%% FUNCTIONS

#%% READS DATA

# Reads images (subset)
images = search_files(paths['aleppo'], 'images.*tif$')[-1:]
images = np.array(list(map(read_raster, images)))
images = images_to_tiles(images, tile_size=(128, 128))

# Reads labels (subset)
labels = search_files(paths['aleppo'], 'label.*tif$')[-1:]
labels = np.array(list(map(read_raster, labels)))
labels = images_to_tiles(labels, tile_size=(1, 1))
labels = np.squeeze(labels, axis=(2, 3)).astype(float)

# Samples tiles (temporary)
random.seed(1)
index = np.concatenate([
    random.choice(np.where(labels == 0)[0], 2500),
    random.choice(np.where(labels == 3)[0], 2500)
])
images = images[index]
labels = labels[index]
labels = np.where(labels == 3, 1.0, 0.0) # Converts to float
del index

# Checks data (not enough context!)
for i in random.choice(range(len(images)), 5):
    display(images[i])
del i

# Splits samples (separate test sample)
samples_size = dict(train=0.8, valid=0.1, test=0.1)
images_train, images_valid, images_test = sample_split(images, sizes=samples_size, seed=1)
labels_train, labels_valid, labels_test = sample_split(labels, sizes=samples_size, seed=1)
samples_size = dict(train=len(images_train), valid=len(images_valid), test=len(images_test))

#%% AUGMENTATION

# Generator parameters
augmentation = dict(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True,
    rotation_range=15, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9,1.1],
    zoom_range=[0.9, 1.1],
    fill_mode='nearest'
)

# Initialises data generator
train_generator = preprocessing.image.ImageDataGenerator(**augmentation)
train_generator = train_generator.flow(images_train, labels_train, batch_size=32, shuffle=True, seed=1)
# del images_train, labels_train

#%% MODELS

# Initialises model
args  = dict(shape=(64, 64, 3), filters=16, units=32, dropout=0.1)
model = convolutional_network(**args)
model.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics='accuracy')
# display_structure(model, '../models/convolutional_network')

#%% OPTIMISATION

# Callbacks
train_callbacks = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    callbacks.BackupAndRestore(backup_dir='../models')
]

# Estimates parameters
training = model.fit(
    train_generator,
    steps_per_epoch=samples_size['train'] // 32,
    validation_data=(images_valid, labels_valid),
    epochs=100,
    verbose=1,
    callbacks=train_callbacks
)

# Saves estimated model
# models.save_model(model, 'siamese_convolutional_network.h5')
# np.save('siamese_convolutional_network.npy', training.history)

#%% CHECKS

# Check generator
# images_batch, labels_batch = data_generator.next()
# display(images_batch[1])
# del images_batch, labels_batch