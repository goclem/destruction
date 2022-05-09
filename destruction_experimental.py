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

from rasterio import windows
from destruction_utilities import *
from numpy import random
from skimage import measure

windows = dict(
    aleppo_label = windows.Window.from_slices((88, 108), (136, 156)),
    aleppo_image = windows.Window.from_slices((11264, 13824), (17408, 19968))
)

damages = search_data('damage.gpkg$')
profile = search_data(pattern('aleppo', 'image'))[0]
damages = rasterise(damages, profile, '2016-09-18')[11264:13824, 17408:19968]
damages = np.expand_dims(np.equal(damages, 3), (0, 3))
damages = np.squeeze(images_to_tiles(damages, tile_size=(128, 128)))
damages = np.array([measure.block_reduce(damage, (4,4), np.max) for damage in damages])

images = search_data(pattern('aleppo', 'image'))
images = np.array([read_raster(image, window=windows['aleppo_image']) for image in [images[5], images[25]]])
images = images_to_tiles(images, tile_size=(128, 128))
image0 = images[:400]
image1 = images[400:]

labels = search_data(pattern('aleppo', 'label'))
labels = np.array([read_raster(label, window=windows['aleppo_label']) for label in [labels[5], labels[25]]])
labels = np.equal(labels, 3)
labels = images_to_tiles(labels, tile_size=(1, 1))
labels = np.squeeze(labels)
label0 = labels[:400]
label1 = labels[400:]

index = np.where(label0 != label1)[0]
for i in np.random.choice(index, 5): 
    compare([image0[i], image1[i], damages[i]], [label0[i], label1[i], 'damage'])