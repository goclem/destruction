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

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='aleppo', help='City name')

# Parse command-line arguments
args = parser.parse_args()

# Utilities
params = argparse.Namespace(
    city=args.city, # aleppo
    tile_size=128, 
    train_size=0.50, valid_size=0.25, test_size=0.25,
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))},
    sequence_ratio=1,
    tile_ratio=1)

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

# Writes damage labels
damage = gpd.GeoDataFrame(data=filling[dates].astype(int), geometry=geoms)

print('Writing damage labels')
for date in dates:
    print(f' - Processing period {date}')
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting retains the maximum recorded destruction per pixel
    subset = rasterise(source=subset, profile=profile, varname=date)
    write_raster(array=subset, profile=profile, destination=f'{paths.data}/{params.city}/labels/label_{date}.tif')

del damage, geoms, filling, defined, dates, date, subset

'''Checks filling
for i in random.choice(np.arange(damage.shape[0]), 1):
    print(pd.concat((damage.iloc[i], filling.iloc[i]), axis=1))
'''

#%% CREATES THE SEQUENCES DATASETS

#! Removes existing zarr
reset_folder(f'{paths.data}/{params.city}/zarr', remove=True)

# Files and samples
images  = search_data(pattern(city=params.city, type='image'))
labels  = search_data(pattern(city=params.city, type='label'))
samples = search_data(f'{params.city}_samples.tif$')
samples = load_sequences(samples, tile_size=1).squeeze()

# Writes zarr arrays
print('Creating the sequences datasets')
for i, (image, label) in enumerate(zip(images, labels)):
    print(f' - Processing period {i+1: 2d}/{len(images)}')
    # Loads images and labels
    arrays = dict(
        images=load_sequences([image], tile_size=params.tile_size).squeeze(1).numpy(),
        labels=load_sequences([label], tile_size=1).squeeze(2, 3, 4).numpy())
    # Writes data for each sample
    for sample, value in dict(train=1, valid=2, test=3).items():
        for label, array in arrays.items():
            array   = zarr.array(array[samples == value], dtype='u1')
            shape   = (array.shape[0], len(images), *array.shape[1:])
            dataset = f'{paths.data}/{params.city}/zarr/{label}_sequence_{sample}.zarr'
            dataset = zarr.open(dataset, mode='a', shape=shape, dtype='u1')
            dataset[:,i] = array

del images, labels, samples, i, image, label, arrays, sample, value, array, shape, dataset

#%% BALANCES THE SEQUENCE DATASET BY DOWNSAMPLING NO-DESTRUCTION SEQUENCES

"""print('Downsamples no-destruction sequences')
for sample in ['train', 'valid', 'test']:
    print(f' - Processing sample {sample}')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_sequence_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_sequence_{sample}_balanced.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_{sample}_balanced.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')[:]
    src_labels = zarr.open(src_labels, mode='r')[:]
    # Subsets source datasets
    destroy = [k for k, v in params.label_map.items() if v == 1]
    destroy = (np.sum(np.isin(src_labels, destroy), axis=1) > 0).flatten()
    untouch = np.where(~destroy)[0]
    indices = np.concatenate((
        np.where(destroy)[0], # Includes all destroyed samples
        np.random.choice(untouch, params.sequence_ratio * np.sum(destroy), replace=True)))
    np.random.shuffle(indices)
    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(len(indices), *src_images.shape[1:]), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(indices), *src_labels.shape[1:]), dtype=src_labels.dtype)
    dst_images[:] = src_images[indices]
    dst_labels[:] = src_labels[indices]

del sample, src_images, src_labels, dst_images, dst_labels, destroy, untouch, indices"""

#%% RESHAPES THE SEQUENCES DATASET INTO THE TILES DATASET

print('Reshaping the sequences dataset into the tiles dataset')
for sample in ['train', 'valid', 'test']:
    print(f' - Processing {sample} sample')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_sequence_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_tile_{sample}.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_tile_{sample}.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')
    src_labels = zarr.open(src_labels, mode='r')
    n, T, c, h, w = src_images.shape
    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(n * T, c, h, w), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(n * T, 1),       dtype=src_labels.dtype)
    for t in range(T):
        dst_images[t*n:(t+1)*n,:] = src_images[:,t,:]
        dst_labels[t*n:(t+1)*n,:] = src_labels[:,t,:]

del sample, src_images, src_labels, dst_images, dst_labels, n, T, c, h, w, t

#%% BALANCES THE TILE DATASET BY DOWNSAMPLING NO-DESTRUCTION TILES

balancing_method = 2
balancing_method_dict = {0: "Keep original composition",
                         1: "Downsampling no-destruction tiles",
                         2: "Upsampling destruction tiles"}

print(f'Balancing the dataset: {balancing_method_dict[balancing_method]}')

for sample in ['train', 'valid', 'test']:
    print(f' - Processing {sample} sample')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_tile_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_tile_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_tile_{sample}_balanced.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_tile_{sample}_balanced.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')[:]
    src_labels = zarr.open(src_labels, mode='r')[:]
    
    # Implement different sampling strategies
    destroy = [k for k, v in params.label_map.items() if v == 1]                        # list of categories indicating destruction (see label_map)
    destroy = np.isin(src_labels, destroy).flatten()                                    # list of bools, indicating whether label is destroyed or not-destroyed 
    untouch = np.where(np.logical_and(~destroy, ~np.isnan(src_labels.flatten())))[0]    # list of undestroyed tiles

    print(f"\t - destruction: {len(np.where(destroy)[0])}, no-destruction: {len(untouch)}, total: {len(src_labels)}")
    
    if balancing_method == 0:
    # Keep original composition
        indices = range(len(src_labels.flatten()))
        
    elif balancing_method == 1:
    # Downsample undestroyed tiles
        indices = np.concatenate((
            np.where(destroy)[0],                                                           # list of destroyed tiles
            np.random.choice(untouch, params.tile_ratio * np.sum(destroy), replace=False))) # sample of undestroyed of same size
        np.random.shuffle(indices)
    
    elif balancing_method == 2:
    # Upsampling destroyed tiles
        indices = np.concatenate((
            untouch,                                                                        # list of undestroyed tiles
            np.random.choice(np.where(destroy)[0], len(untouch), replace=True)))           # sample of destroyed of same size
        np.random.shuffle(indices)
        
    print(f"\t - total after rebalancing final: {len(indices)}")
        
    # Writes data
    dst_images = zarr.open(dst_images, mode='w', shape=(len(indices), *src_images.shape[1:]), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(indices), *src_labels.shape[1:]), dtype=src_labels.dtype)
    dst_images[:] = src_images[indices]
    dst_labels[:] = src_labels[indices]

del sample, src_images, src_labels, dst_images, dst_labels, destroy, untouch, indices

#%%
