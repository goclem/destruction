#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Experimental functions
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.10
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

# Reads images as sequences
images = search_data(pattern('aleppo', 'image'))
dates  = get_dates(images)
window = center_window(images[0], (20*128, 20*128))
images = np.array([read_raster(image, window=window) for image in images])
images = images_to_sequences(images, tile_size=(128, 128))
del window

# Reads labels as sequences
labels = search_data(pattern('aleppo', 'label'))
window = center_window(labels[0], (20, 20))
labels = np.array([read_raster(label, window=window, dtype='int8') for label in labels])
labels = images_to_sequences(labels, tile_size=(1, 1))
labels = np.squeeze(labels)
del window

# Keeps sequences with destruction
analysis  = np.all(labels !=-1, axis=1)
destroyed = np.any(labels == 3, axis=1)
images    = images[analysis & destroyed]
labels    = labels[analysis & destroyed]
labels    = np.equal(labels, 3)
del analysis, destroyed

# Checks sequence
def check_destruction(i):
    change  = int(np.where(np.diff(labels[i]))[0]) + 1
    index   = (change - 2, change + 2)
    label   = labels[i][index[0]:index[1]]
    image   = images[i][index[0]:index[1]]
    caption = dates[index[0]:index[1]]
    caption = [f'{d}: {l}' for d, l in zip(date, label)]
    compare(image, caption)

check_destruction(2)
len(generate_sequences(labels[1], 2))