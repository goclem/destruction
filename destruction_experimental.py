#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Experimental functions
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.09
'''

#%% SEQUENCE GENERATOR

# Modules
import numpy as np
import itertools
from destruction_preprocess import get_dates

# Generates sequences of an arbitatry length
def generate_sequences(sequence:bool, size:int=2) -> list:
    index0  = np.where(sequence==0)[0]
    index1  = np.where(sequence==1)[0]
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

# %% READS SLICES OF RASTER

from destruction_utilities import *

images = search_data(pattern('aleppo', 'image'))
dates  = get_dates(images)
window = center_window(images[0], (20*128, 20*128))
images = np.array([read_raster(image, window=window) for image in images])
images = images_to_sequences(images, tile_size=(128, 128))

labels = search_data(pattern('aleppo', 'label'))
window = center_window(labels[0], (20, 20))
labels = np.array([read_raster(label, window=window, dtype='int8') for label in labels])
labels = images_to_sequences(labels, tile_size=(1, 1))
labels = np.squeeze(labels)

analysis  = np.all(labels !=-1, axis=1)
destroyed = np.any(labels == 3, axis=1)
images    = images[analysis & destroyed]
labels    = labels[analysis & destroyed]
labels    = np.equal(labels, 3)

def check_destruction(i):
    change = int(np.where(np.diff(labels[i]))[0])
    index  = (change - 2, change + 4)
    label  = labels[i][index[0]:index[1]]
    image  = images[i][index[0]:index[1]]
    date   = dates[index[0]:index[1]]
    compare(image, date)

check_destruction(16)


index = np.where(labels == 1)[0]
for i in np.random.choice(index, 5): 
    display(images[i], f'Label: {labels[i]}')



date_image = get_dates(images)
date_label = get_dates(labels)
list(set(date_label) - set(date_image))


#%% DEPRECIATED

source  = '../data/aleppo/images/image_2017-08-14.tif'damages = search_data('damage.gpkg$')
profile = search_data(pattern('aleppo', 'image'))[0]
damages = rasterise(damages, profile, '2016-09-18')[11264:13824, 17408:19968]
damages = np.expand_dims(np.equal(damages, 3), (0, 3))
damages = np.squeeze(images_to_tiles(damages, tile_size=(128, 128)))
damages = np.array([measure.block_reduce(damage, (4,4), np.max) for damage in damages])
