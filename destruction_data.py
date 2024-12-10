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
params = argparse.Namespace(
    city='aleppo', 
    tile_size=128, 
    train_size=0.50, valid_size=0.25, test_size=0.25,
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})

#%% COMPUTES SAMPLES

# Computes analysis zone
profile    = search_data(pattern=pattern(city=params.city, type='image'))[0]
profile    = tiled_profile(profile, tile_size=params.tile_size)
settlement = search_data(pattern=f'{params.city}_settlement.*gpkg$')[0]
settlement = rasterise(source=settlement, profile=profile, update=dict(dtype='uint8')).astype(bool)
noanalysis = search_data(pattern=f'{params.city}_noanalysis.*gpkg$')[0]
noanalysis = rasterise(source=noanalysis, profile=profile, update=dict(dtype='uint8')).astype(bool)
analysis   = np.logical_and(settlement, np.invert(noanalysis))
del settlement, noanalysis

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
dates = extract(dates, r'\d{4}_\d{2}_\d{2}')

# Adds images dates to damage dataset
damage[list(set(dates) - set(damage.columns))] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)
damage, geoms = damage.drop(columns='geometry'), damage.geometry

# Fills missing dates
damage.insert(0,  'pre',  0)
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

#%% CREATES DATASETS FOR DESTRUCTION MODEL

#! Removes existing zarr
reset_folder(f'{paths.data}/{params.city}/zarr', remove=True)

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
    for sample, value in dict(train=1, valid=2, test=3).items():
        for label, array in arrays.items():
            array   = zarr.array(array[samples == value], dtype='u1')
            shape   = (array.shape[0], len(images), *array.shape[1:])
            dataset = f'{paths.data}/{params.city}/zarr/{label}_{sample}.zarr'
            dataset = zarr.open(dataset, mode='a', shape=shape, dtype='u1')
            dataset[:,i] = array

del images, labels, samples, i, image, label, arrays, sample, value, array, shape, dataset

#%% DOWNSAMPLES NO-DESTRUCTION SEQUENCES

for sample in ['train', 'valid', 'test']:
    print(f'Processing sample {sample}')
    # Define paths
    images_zarr = f'{paths.data}/{params.city}/zarr/images_{sample}.zarr'
    labels_zarr = f'{paths.data}/{params.city}/zarr/labels_{sample}.zarr'
    # Loads data
    images = zarr.open(images_zarr, mode='r')[:]
    labels = zarr.open(labels_zarr, mode='r')[:]
    # Subsets datasets
    destroy = [k for k, v in params.label_map.items() if v == 1]
    destroy = (np.sum(np.isin(labels, destroy), axis=1) > 0).flatten()
    indices = np.concatenate((
        np.where(destroy)[0],
        np.random.choice(np.where(~destroy)[0], np.sum(destroy), replace=False)))
    images = images[indices]
    labels = labels[indices]
    # Writes data
    dataset = zarr.open(images_zarr, shape=images.shape, dtype=images.dtype, mode='w')
    dataset[:] = images
    dataset = zarr.open(labels_zarr, shape=labels.shape, dtype=labels.dtype, mode='w')
    dataset[:] = labels
    # Shuffles datasets
    shuffle_zarr(images_zarr, labels_zarr)

del sample, images_zarr, labels_zarr, images, labels, destroy, indices, dataset

#%% RESHAPES ZARR DATASETS FOR THE VITMAE MODEL

for sample in ['train', 'valid', 'test']:
    print(f'Processing sample {sample}')
    src_dataset   = f'{paths.data}/{params.city}/zarr/images_{sample}.zarr'
    src_dataset   = zarr.open(src_dataset, mode='r')
    n, T, c, h, w = src_dataset.shape
    dst_dataset   = f'{paths.data}/{params.city}/zarr/images_{sample}_vitmae.zarr'
    dst_dataset   = zarr.open(dst_dataset, mode='w', shape=(n * T, c, h, w), dtype=src_dataset.dtype)
    for t in range(T):
        dst_dataset[t*n:(t+1)*n,:] = src_dataset[:,t,:]

del sample, src_dataset, dst_dataset, n, T, c, h, w, t

#%% DOWNLOADS FEATURE EXTRACTOR (DEPRECIATED)

'''
import satlaspretrain_models

feature_extractor = satlaspretrain_models.Weights()
feature_extractor = feature_extractor.get_pretrained_model(model_identifier='Aerial_SwinB_SI', fpn=True, device='cpu')
torch.save(feature_extractor, f'{paths.models}/Aerial_SwinB_SI.pth')
del feature_extractor
'''
