#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Experimental functions
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.11
'''

#%% SEQUENCE GENERATOR

# Modules
import numpy as np
import itertools
from destruction_preprocess import *

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

#%% FORMATS DATA AS SEQUENCES

import numpy as np
from destruction_utilities import *

# Splits the data multiple samples
def sample_split(images:np.ndarray, samples:dict) -> list:
    samples = [images[samples == value, ...] for value in np.unique(samples)]
    return samples

def shift_sequences(images:np.ndarray):
    n, t, h, w, d = images.shape
    sequence1 = images[:, :-1,...].reshape(n*(t-1), h, w, d)
    sequence2 = images[:,  1:,...].reshape(n*(t-1), h, w, d)
    return sequence1, sequence2

# Reads images as sequences
tile_size = (128, 128)
images  = search_data(pattern('aleppo', 'image'))
window  = center_window(images[0], (20*tile_size[0], 20*tile_size[1]))
images  = np.array([read_raster(image, window=window, dtype='uint8') for image in images])
images  = tile_sequences(images, tile_size)
del window

# Reads labels as sequences
tile_size = (1, 1)
labels = search_data(pattern('aleppo', 'label'))
window = center_window(labels[0], (20*tile_size[0], 20*tile_size[1]))
labels = np.array([read_raster(label, window=window, dtype='uint8') for label in labels])
labels = np.equal(labels, 3)
labels = tile_sequences(labels, tile_size)
del window

# Reads samples
samples = search_data('aleppo_samples.tif$')
window  = center_window(samples, (20, 20))
samples = read_raster(samples, window=window, dtype='int8')
samples = samples.flatten()
del window

# Split itno samples
_, images_train, images_test, images_valid = sample_split(images, samples)
sequence1, sequence2 = shift_sequences(images_train)

_, labels_train, labels_test, labels_valid = sample_split(labels, samples)
labels1, labels2 = shift_sequences(labels_train)

# Collapse labels as difference
labels_train = np.squeeze(np.invert(np.equal(labels1, labels2)))
del _

#%% Checks sequences
def check_sequence(i:int, span:int=2):
    change = int(np.where(np.diff(labels[i]))[0]) + 1
    index  = (change - span, change + span)
    label  = labels[i][index[0]:index[1]]
    image  = images[i][index[0]:index[1]]
    date   = dates[index[0]:index[1]]
    caption = [f'{d}: {l}' for d, l in zip(date, label)]
    compare(image, caption)

for i in range(5):
    check_sequence(i, 2)
len(generate_sequences(labels[1], 2))

#%% DEPRECIATED
images = np.concatenate((
    np.expand_dims(tile_sequences(images[:-1,...], tile_size), 1), 
    np.expand_dims(tile_sequences(images[1:,...],  tile_size), 1)), 
    1)