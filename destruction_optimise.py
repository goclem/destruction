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
import numpy as np
import pandas as pd
import itertools
import destruction_models as models

from destruction_utilities import *
from numpy import random
from matplotlib import pyplot
from keras import callbacks, preprocessing
from os import path

#%% FUNCTIONS

def display_history(history:dict, stats:list=['accuracy', 'loss']) -> None:
    '''Displays model training history'''
    fig, axs = pyplot.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for ax, stat in zip(axs.ravel(), stats):
        ax.plot(history[stat])
        ax.plot(history[f'val_{stat}'])
        ax.set_title(f'Training {stat}', fontsize=15)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Training sample', 'Validation sample'], frameon=False)
    pyplot.tight_layout(pad=2.0)
    pyplot.show()

#%% READS DATA

# Reads images (subset)
images = search_data(pattern(city='aleppo', type='image'))[-1:]
images = np.array(list(map(read_raster, images)))
images = tile_sequences(images, tile_size=(128, 128))

# Reads labels (subset)
labels = search_data(pattern(city='aleppo', type='label'))[-1:]
labels = np.array(list(map(read_raster, labels)))
labels = tile_sequences(labels, tile_size=(1, 1))
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
    rotation_range=10, 
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
args  = dict(shape=(128, 128, 3), filters=16, units=32, dropout=0.1) # ! Check parameters before run
model = models.convolutional_network(**args)
model.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics='accuracy')
# display_structure(model, path.join(paths['models'], 'convolutional_network.html'))


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
# models.save_model(model, path.join(paths['models'], 'convolutional_network.h5'))
# np.save(path.join(paths['models'], 'convolutional_history.h5'), training.history)

#%% CHECKS

# Check generator
# images_batch, labels_batch = data_generator.next()
# display(images_batch[1])
# del images_batch, labels_batch