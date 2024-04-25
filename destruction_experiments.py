#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Experiments for the destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import numpy as np
import zarr
from destruction_utilities import *

#%% SHUFFLES ZARR FILES

images_zarr = f'{paths.data}/aleppo/zarr/images_train.zarr'
labels_zarr = f'{paths.data}/aleppo/zarr/labels_train.zarr'

# Reads datasets
images = zarr.open(images_zarr, mode='r')[:]
labels = zarr.open(labels_zarr, mode='r')[:]

# Subsets datasets
subset = (np.sum(np.isin(labels, [1,2,3]), axis=1) > 0).flatten()
images = images[subset]
labels = labels[subset]

# Shfulle indices
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

dataset = zarr.open(images_zarr, shape=images.shape, dtype=images.dtype, mode='w')
dataset[:] = images
dataset = zarr.open(labels_zarr, shape=labels.shape, dtype=labels.dtype, mode='w')
dataset[:] = labels

shuffle_zarr(images_zarr, labels_zarr)





dst_zarr = f'{paths.data}/aleppo/zarr/images_train2.zarr'
shuffle_zarr(src_zarr, dst_zarr)

def shuffle_zarr(src_zarr:str, dst_zarr:str):
    '''Shuffles a Zarr array along the first axis'''
    src_zarr = zarr.open(src_zarr, mode='r')
    shape    = src_zarr.shape
    dst_zarr = zarr.open(dst_zarr, shape=shape, dtype=src_zarr.dtype, mode='w')
    # Create a list of indices to shuffle
    indices = list(range(shape[0]))
    random.shuffle(indices)
    # Shuffle the input Zarr array along the first axis
    for i, idx in enumerate(indices):
        print(f'{i}/{shape[0]}')
        dst_zarr[i] = src_zarr[idx]