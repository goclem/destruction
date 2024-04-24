#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Preprocessing for the destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import zarr
from destruction_utilities import *

# Utilities
params = argparse.Namespace(city='aleppo', tile_size=128)

#%% DATA PREPROCESSING

#! Removes existing zarr
reset_folder(f'{paths.data}/{params.city}/zarr', remove=True)

# Files
images  = search_data(pattern(city=params.city, type='image'))
labels  = search_data(pattern(city=params.city, type='label'))

# Samples
samples = search_data(f'{params.city}_samples.tif$')
samples = load_sequences(samples, tile_size=1).squeeze()

# Writes zarr arrays
for i, (image, label) in enumerate(zip(images, labels)):
    print(f'Processing period {i+1:02d}/{len(images)}')
    # Loads images and labels
    arrays = dict(
        image=load_sequences([image], tile_size=params.tile_size).squeeze(1).numpy(),
        label=load_sequences([label], tile_size=1).squeeze(2, 3, 4).numpy())
    # Writes data for each sample
    for subsample, value in dict(train=1, valid=2, test=3).items():
        for label, array in arrays.items():
            array   = zarr.array(array[samples == value], dtype='u1')
            shape   = (array.shape[0], len(images), *array.shape[1:])
            dataset = f'{paths.data}/{params.city}/zarr/{label}_{subsample}.zarr'
            dataset = zarr.open(dataset, mode='a', shape=shape, dtype='u1')
            dataset[:,i] = array
            del array, shape, dataset
    del arrays
