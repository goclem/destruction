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

# Parameters
params = argparse.Namespace(
    city='aleppo',
    tile_size=128, 
    train_size=0.50, valid_size=0.25, test_size=0.25,
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))},
    sequence_ratio=1,
    prepost_npre=1, #! Number of pre-images
    prepost_ratio=1,
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

'''Checks filling
for i in random.choice(np.arange(damage.shape[0]), 1):
    print(pd.concat((damage.iloc[i], filling.iloc[i]), axis=1, keys=['damage', 'filling']))
'''

# Writes damage labels
damage = gpd.GeoDataFrame(data=filling[dates].astype(int), geometry=geoms)

print('Writing damage labels')
for date in dates:
    print(f' - Processing period {date}')
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting retains the maximum recorded destruction per pixel
    subset = rasterise(source=subset, profile=profile, varname=date)
    write_raster(array=subset, profile=profile, destination=f'{paths.data}/{params.city}/labels/label_{date}.tif')

del damage, geoms, filling, defined, date, subset

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
for t, (image, label) in enumerate(zip(images, labels)):
    print(f' - Processing period {t+1:02d}/{len(images):02d}')
    # Loads images and labels
    src_images = read_raster(image, dtype='uint8')
    src_images = torch.tensor(src_images).permute(2, 0, 1)
    src_images = image_to_tiles(src_images, tile_size=params.tile_size, stride=params.tile_size).numpy()
    src_labels = read_raster(label, dtype='uint8')
    src_labels = torch.tensor(src_labels).permute(2, 0, 1)
    src_labels = image_to_tiles(src_labels, tile_size=1, stride=1).squeeze(2, 3).numpy()
    # Writes data for each sample
    for sample, value in dict(train=1, valid=2, test=3).items():
        dst_images = f'{paths.data}/{params.city}/zarr/images_sequence_{sample}.zarr'
        dst_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_{sample}.zarr'
        subset     = samples == value
        dst_images = zarr.open(dst_images, mode='a', shape=(subset.sum(), len(images), *src_images.shape[1:]), dtype='u1')
        dst_labels = zarr.open(dst_labels, mode='a', shape=(subset.sum(), len(images), *src_labels.shape[1:]), dtype='u1')
        dst_images[:,t] = zarr.array(src_images[subset], dtype='u1')
        dst_labels[:,t] = zarr.array(src_labels[subset], dtype='u1')
            
del images, image, labels, label, samples, sample, src_images, src_labels, dst_images, dst_labels, value, subset, t

#%% RESHAPES THE SEQUENCES DATASET INTO THE PRE-POST DATASET

print('Reshaping the sequences dataset into the pre-post dataset')
for sample in ['train', 'valid', 'test']:
    print(f' - Processing {sample} sample')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_sequence_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_prepost_{sample}.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_prepost_{sample}.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')
    src_labels = zarr.open(src_labels, mode='r')
    n, T, c, h, w = src_images.shape
    T_pre  = np.arange(0, params.prepost_npre)
    T_post = np.arange(params.prepost_npre, T)
    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(len(T_pre) * n * len(T_post), 2, c, h, w), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(T_pre) * n * len(T_post), 1), dtype=src_labels.dtype)
    idx = 0
    for t_pre in T_pre:
        for t_post in T_post:
            print(f'   - Writing ({idx+1}/{len(T_pre)*len(T_post)}) pre {dates[t_pre]} & post {dates[t_post]}')
            dst_images[idx*n:(idx+1)*n,0,:] = src_images[:,t_pre,:]
            dst_images[idx*n:(idx+1)*n,1,:] = src_images[:,t_post,:]
            dst_labels[idx*n:(idx+1)*n,:]   = src_labels[:,t_post,:]
            idx += 1

del sample, src_images, src_labels, dst_images, dst_labels, n, T, T_pre, T_post, t_pre, t_post, c, h, w, idx

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
    dst_labels = zarr.open(dst_labels, mode='w', shape=(n * T, 1), dtype=src_labels.dtype)
    for t in range(T):
        dst_images[n*t:(t+1)*n,:] = src_images[:,t,:]
        dst_labels[n*t:(t+1)*n,:] = src_labels[:,t,:]

del sample, src_images, src_labels, dst_images, dst_labels, n, T, c, h, w, t

#%% BALANCES THE SEQUENCE DATASET BY DOWNSAMPLING NO-DESTRUCTION SEQUENCES

print('Downsampling no-destruction sequences')
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
        np.random.choice(untouch, params.sequence_ratio * np.sum(destroy), replace=False)))
    np.random.shuffle(indices)
    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(len(indices), *src_images.shape[1:]), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(indices), *src_labels.shape[1:]), dtype=src_labels.dtype)
    dst_images[:] = src_images[indices]
    dst_labels[:] = src_labels[indices]

del sample, src_images, src_labels, dst_images, dst_labels, destroy, untouch, indices

#%% BALANCES THE PRE-POST DATASET BY DOWNSAMPLING NO-DESTRUCTION PRE-POST PAIRS

print('Downsampling no-destruction pre-post pairs')
for sample in ['train', 'valid', 'test']:
    print(f' - Processing {sample} sample')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_prepost_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_prepost_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_prepost_{sample}_balanced.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_prepost_{sample}_balanced.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')[:]
    src_labels = zarr.open(src_labels, mode='r')[:]
    # Subsets datasets
    destroy = [k for k, v in params.label_map.items() if v == 1]
    destroy = np.isin(src_labels, destroy).flatten()
    untouch = np.where(np.logical_and(~destroy, src_labels.flatten() != 255))[0]
    indices = np.concatenate((
        np.where(destroy)[0],
        np.random.choice(untouch, params.prepost_ratio * np.sum(destroy), replace=False)))
    np.random.shuffle(indices)
    # Writes data
    dst_images = zarr.open(dst_images, mode='w', shape=(len(indices), *src_images.shape[1:]), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(indices), *src_labels.shape[1:]), dtype=src_labels.dtype)
    dst_images[:] = src_images[indices]
    dst_labels[:] = src_labels[indices]

del sample, src_images, src_labels, dst_images, dst_labels, destroy, untouch, indices

#%% BALANCES THE TILE DATASET BY DOWNSAMPLING NO-DESTRUCTION TILES

print('Downsampling no-destruction tiles')
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
    # Subsets datasets
    destroy = [k for k, v in params.label_map.items() if v == 1]
    destroy = np.isin(src_labels, destroy).flatten()
    untouch = np.where(np.logical_and(~destroy, src_labels.flatten() != 255))[0]
    indices = np.concatenate((
        np.where(destroy)[0],
        np.random.choice(untouch, params.tile_ratio * np.sum(destroy), replace=False)))
    np.random.shuffle(indices)
    # Writes data
    dst_images = zarr.open(dst_images, mode='w', shape=(len(indices), *src_images.shape[1:]), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(indices), *src_labels.shape[1:]), dtype=src_labels.dtype)
    dst_images[:] = src_images[indices]
    dst_labels[:] = src_labels[indices]

del sample, src_images, src_labels, dst_images, dst_labels, destroy, untouch, indices

#%%