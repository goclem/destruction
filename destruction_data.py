#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Data for the destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import zarr

from numpy import random
from destruction_utilities import *

# Utilities
params = argparse.Namespace(city='aleppo', tile_size=128, train_size=0.50, valid_size=0.25, test_size=0.25)

#%% COMPUTES SAMPLES

# Loads data
image      = search_data(pattern=pattern(city=params.city, type='image'))[0]
settlement = search_data(pattern=f'{params.city}_settlement.*gpkg$')[0]
noanalysis = search_data(pattern=f'{params.city}_noanalysis.*gpkg$')[0]

# Computes analysis zone
profile    = tiled_profile(image, tile_size=params.tile_size)
settlement = rasterise(source=settlement, profile=profile, update=dict(dtype='uint8')).astype(bool)
noanalysis = rasterise(source=noanalysis, profile=profile, update=dict(dtype='uint8')).astype(bool)
analysis   = np.logical_and(settlement, np.invert(noanalysis))
del image, settlement, noanalysis

# Splits samples
random.seed(0)
index = dict(train=params.train_size, valid=params.valid_size, test=params.test_size)
index = np.random.choice(np.arange(len(index)) + 1, np.sum(analysis), p=list(index.values()))
samples = analysis.astype(int)
np.place(samples, analysis, index)
write_raster(samples, profile, f'{paths.data}/{params.city}/others/{params.city}_samples.tif')
del index, samples, analysis

#%% COMPUTES LABELS

# Reads damage reports
damage = search_data(pattern=f'{params.city}_damage.*gpkg$')[0]
damage = gpd.read_file(damage)

# Extract images dates
dates = search_data(pattern=pattern(city=params.city, type='image'))
dates = extract(dates, r'\d{4}-\d{2}-\d{2}')

# Adds images dates to damage dataset
damage[list(set(dates) - set(damage.columns))] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)
damage, geoms = damage.drop(columns='geometry'), damage.geometry

# Fills missing dates
damage.insert(0,  'pre',  0) #? Insert pre-destruction label
damage.insert(len(damage.columns), 'post', damage[damage.T.last_valid_index()]) #? Insert post-destruction label (i.e. no reconstruction)
filling = damage.T.ffill().bfill().T
defined = damage.T.bfill().ffill().T
filling[filling != defined] = 255

'''Checks filling
for i in random.choice(np.arange(damage.shape[0]), 1):
    print(pd.concat((damage.iloc[i], filling.iloc[i]), axis=1))
'''

# Writes damage labels
damage = gpd.GeoDataFrame(data=filling[dates].astype(int), geometry=geoms)
del geoms, filling, defined

for date in dates:
    print(f'Processing {date}')
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting retains the maximum recorded destruction per pixel
    subset = rasterise(source=subset, profile=profile, varname=date)
    write_raster(array=subset, profile=profile, destination=f'{paths.data}/{params.city}/labels/label_{date}.tif')
del dates, date, subset

#%% WRITES ZARR DATASETS

''' #! Removes existing zarr
reset_folder(f'{paths.data}/{params.city}/zarr', remove=True)
'''

# Files and samples
images  = search_data(pattern(city=params.city, type='image'))
labels  = search_data(pattern(city=params.city, type='label'))
samples = search_data(f'{params.city}_samples.tif$')
samples = load_sequences(samples, tile_size=1).squeeze()

# Writes zarr arrays
for i, (image, label) in enumerate(zip(images, labels)):
    print(f'Processing period {i+1:02d}/{len(images)}')
    # Loads images and labels
    arrays = dict(
        images=load_sequences([image], tile_size=params.tile_size).squeeze(1).numpy(),
        labels=load_sequences([label], tile_size=1).squeeze(2, 3, 4).numpy())
    # Writes data for each sample
    for subsample, value in dict(train=1, valid=2, test=3).items():
        for label, array in arrays.items():
            array   = zarr.array(array[samples == value], dtype='u1')
            shape   = (array.shape[0], len(images), *array.shape[1:])
            dataset = f'{paths.data}/{params.city}/zarr/{label}_{subsample}.zarr'
            dataset = zarr.open(dataset, mode='a', shape=shape, dtype='u1')
            dataset[:,i] = array
            del label, array, shape, dataset
    del image, label, arrays
