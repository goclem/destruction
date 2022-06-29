#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Experimental functions
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.11
'''

# Modules
import numpy as np
import destruction_models as models

from destruction_utilities import *
from keras import callbacks, metrics

#%% LOADS DATA

# Reads images as sequences (20x20 tile subset)
tile_size = (128, 128)
images  = search_data(pattern('aleppo', 'image'))
window  = center_window(images[0], (20*tile_size[0], 20*tile_size[1]))
images  = np.array([read_raster(image, window=window, dtype='uint8') for image in images])
images  = tile_sequences(images, tile_size)
del window

# Reads labels as sequences (20x20 tile subset)
tile_size = (1, 1)
labels = search_data(pattern('aleppo', 'label'))
window = center_window(labels[0], (20*tile_size[0], 20*tile_size[1]))
labels = np.array([read_raster(label, window=window, dtype='uint8') for label in labels])
labels = np.equal(labels, 3)
labels = tile_sequences(labels, tile_size)
del window

# Reads samples (20x20 tile subset)
samples = search_data('aleppo_samples.tif$')
window  = center_window(samples, (20, 20))
samples = read_raster(samples, window=window, dtype='int8')
samples = samples.flatten()
del window

#%% RESHAPES DATA FOR SIAMESE MODEL

def reshape_siamese(images):
    n, t, h, w, d = images.shape
    images_t0 = np.tile(np.take(images, [0], 1), (1, t-1, 1, 1, 1)).reshape(n * (t-1), h, w, d)
    images_tt = np.delete(images, 0, 1).reshape(n * (t-1), h, w, d)
    return images_t0, images_tt

# Split samples
_, images_train, images_test, images_valid = sample_split(images, samples)
_, labels_train, labels_test, labels_valid = sample_split(labels, samples)
del _

images_train_t0, images_train_tt = reshape_siamese(images_train)
images_valid_t0, images_valid_tt = reshape_siamese(images_valid)
labels_train = np.delete(labels_train, 0, 1).flatten()
labels_valid = np.delete(labels_valid, 0, 1).flatten()

#%% ESTIMATES SIAMESE MODEL

# Model structure
model = models.siamese_convolutional_network(shape=(128, 128, 3), args_encode=dict(filters=8, dropout=0), args_dense=dict(units=16, dropout=0))
model.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=metrics.AUC(num_thresholds=200, curve='ROC'))
model.summary()

# Estimates parameters
training = model.fit(    
    {'images_t0':images_train_t0, 'images_tt':images_train_tt}, 
    y=labels_train,
    epochs=100,
    verbose=1,
    callbacks=callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)
)

#%% SEQUENCE GENERATOR

# Modules
import numpy as np
import itertools

# Generates sequences of an arbitatry length
def generate_sequences(sequence:bool, size:int=2) -> list:
    index0 = np.where(sequence==0)[0]
    index1 = np.where(sequence==1)[0]
    assert len(index0) >= size, 'Sequence smaller than size'
    assert len(index1) >= size, 'Sequence smaller than size'
    sequence0 = itertools.combinations(index0, size)
    sequence1 = itertools.combinations(index1, size)
    sequences = list(itertools.product(sequence0, sequence1))
    return sequences

# Simulated data
sequence = np.array((False, False, False, True, True, True))
sequence = np.tile(sequence, 9).reshape(3, 3, len(sequence))

# Sequences
sequence = sequence.reshape(3 * 3, 6)
sequence = np.apply_along_axis(generate_sequences, 1, sequence)

# Checks sequence

def check_sequence(i:int, span:int=2):
    change = int(np.where(np.diff(labels_simaese[i]))[0]) + 1
    index  = (change - span, change + span)
    label  = labels[i][index[0]:index[1]]
    image  = images[i][index[0]:index[1]]
    date   = dates[index[0]:index[1]]
    caption = [f'{d}: {l}' for d, l in zip(date, label)]
    compare(image, caption)
