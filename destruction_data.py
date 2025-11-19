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
args,unknown = parser.parse_known_args()

# Utilities
params = argparse.Namespace(
    city=args.city, # aleppo
    buffer_around_destruction=False,
    buffer_around_destruction_DISTANCE=1, # in patches
    reset_zarr=False,
    image_size=224,
    patch_size=32, 
    sample_sizes={'train':0.50, 'val':0.25, 'test':0.25},
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))},
    sequence_ratio=1,
    prepost_npre=1, #! Number of pre-images
    prepost_ratio=1,
    tile_ratio=1,
    chunk_size=500,
    )     # Adjust chunk size according to your memory constraints.

if params.image_size % params.patch_size != 0:
    raise ValueError('Image size must be divisible by patch size')

#%% BUILD ANALYSIS MASK, SPLIT IN TRAIN/TEST/VALIDATION AND SAVE RASTER "_samples.tif"

# Assemble the analysis mask (settlement minus noanalysis), on a tiled profile
profile    = search_data(pattern=pattern(city=params.city, type='image'))[0]
profile    = tiled_profile(profile, tile_size=params.image_size, crop_size=params.image_size, return_window=False)
settlement = search_data(pattern=f'{params.city}_settlement.*gpkg$')[0]
settlement = rasterise(source=settlement, profile=profile, update=dict(dtype='uint8')).astype(bool)
noanalysis = search_data(pattern=f'{params.city}_noanalysis.*gpkg$')[0]
noanalysis = rasterise(source=noanalysis, profile=profile, update=dict(dtype='uint8')).astype(bool)
analysis   = np.logical_and(settlement, np.invert(noanalysis))
del settlement, noanalysis

# Assign each True pixel to a split using target proportions (multinomial randomness)
# Note: proportions are *probabilities*, not exact quotas.
random.seed(0)
index = np.random.choice(np.arange(len(params.sample_sizes)) + 1, np.sum(analysis), p=list(params.sample_sizes.values()))
samples = analysis.astype(int)
np.place(samples, analysis, index)
write_raster(samples, profile, f'{paths.data}/{params.city}/others/{params.city}_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_samples.tif')
del profile, index, samples, analysis

#%% COMPUTES LABELS
   
# Reads damage reports
damage = search_data(pattern=f'{params.city}_damage.*gpkg$')[0]
damage = gpd.read_file(damage)

# Extract images dates (pre and post combined)
#dates = search_data(pattern=pattern(city=params.city, type='image'))
dates_pre = search_data(pattern(city=params.city, type='pre/image'))
dates_pre = extract(dates_pre, r'\d{4}_\d{2}_\d{2}')
dates_post = search_data(pattern(city=params.city, type='post/image'))
dates_post = extract(dates_post, r'\d{4}_\d{2}_\d{2}')

# Adds images dates to damage dataset
damage[list(set(dates_post) - set(damage.columns))] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)
damage, geoms = damage.drop(columns='geometry'), damage.geometry

# Fills missing dates
damage.insert(0,  'pre',  0)
damage.insert(len(damage.columns), 'post', damage[damage.T.last_valid_index()]) #? Insert post-destruction label (i.e. no reconstruction)
filling = damage.T.ffill().bfill().T
defined = damage.T.bfill().ffill().T
filling[filling != defined] = 255
filling[dates_pre] = 0
'''Checks filling
for i in random.choice(np.arange(damage.shape[0]), 1):
    print(pd.concat((damage.iloc[i], filling.iloc[i]), axis=1, keys=['damage', 'filling']))
'''

dates = np.concatenate([dates_pre, dates_post])
# Writes damage labels
damage = gpd.GeoDataFrame(data=filling[dates].astype(int), geometry=geoms)
profile = search_data(pattern=pattern(city=params.city, type='image'))[0]
profile = tiled_profile(source=profile, tile_size=params.patch_size, crop_size=params.image_size)

keys_equal_one = [k for k, v in params.label_map.items() if v == 1]

print('Writing damage labels')
for date in dates:
    print(f' - Processing period {date}')
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting retains the maximum recorded destruction per pixel
    subset = rasterise(source=subset, profile=profile, varname=date)
    
    if params.buffer_around_destruction:
        # Experiment of setting places adjacent to destruction as 255
        arr = subset[..., 0]
        pos = np.isin(arr, keys_equal_one)

        H, W = pos.shape
        neighbors8 = [(-1,0),(1,0),(0,-1),(0,1),
                      (-1,-1),(-1,1),(1,-1),(1,1)]
        # --- radius 1 dilation ---
        nbr1 = np.zeros_like(pos, dtype=bool)
        for dy, dx in neighbors8:
            shift_or(nbr1, pos, dy, dx, H, W)

        nbr = nbr1
        if params.buffer_around_destruction_DISTANCE == 2:
            # --- radius 2 dilation (dilate nbr1 further) ---
            nbr2 = np.zeros_like(pos, dtype=bool)
            for dy, dx in neighbors8:
                shift_or(nbr2, nbr1 | pos, dy, dx, H, W)  # include original pos so growth is from all radius-1 pixels

            nbr = nbr1 | nbr2
            
        out = arr.copy()
        out[nbr & ~pos] = 255
        subset = out[..., None]

    write_raster(array=subset, profile=profile, destination=f'{paths.data}/{params.city}/labels/label_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{date}.tif')

#del damage, geoms, filling, defined, date, subset




#%% CREATES THE SEQUENCES DATASETS

#! Removes existing zarr
if params.reset_zarr:
    reset_folder(f'{paths.data}/{params.city}/zarr', remove=True)

# Files and samples
images  = search_data(pattern(city=params.city, type='image'))
labels  = search_data(pattern(city=params.city, type=f'label_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}'))
samples = search_data(f'{params.city}_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_samples.tif$')
samples = load_sequences(samples, tile_size=1).squeeze()
_, window = tiled_profile(source=images[0], tile_size=params.patch_size, crop_size=params.image_size, return_window=True)


# Writes zarr arrays
print('Creating the sequences datasets')
for t, (image, label) in enumerate(zip(images, labels)):
    print(f' - Processing period {t+1:02d}/{len(images):02d}')
    # Loads images and labels
    src_images = read_raster(image, dtype='uint8', window=window)
    src_images = torch.tensor(src_images).permute(2, 0, 1)
    src_images = image_to_tiles(src_images, tile_size=params.image_size).numpy()
    src_labels = read_raster(label, dtype='uint8', window=window)
    src_labels = torch.tensor(src_labels).permute(2, 0, 1)
    src_labels = image_to_tiles(src_labels, tile_size=params.image_size//params.patch_size).numpy()
    # Writes data for each sample
    for sample, value in dict(train=1, valid=2, test=3).items():
        dst_images = f'{paths.data}/{params.city}/zarr/images_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
        dst_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
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
    src_images = f'{paths.data}/{params.city}/zarr/images_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_prepost_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_prepost_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')
    src_labels = zarr.open(src_labels, mode='r')
    n, T, cx, hx, wx = src_images.shape
    n, T, cy, hy, wy = src_labels.shape
    T_pre  = np.arange(0, params.prepost_npre)
    T_post = np.arange(params.prepost_npre, T)
    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(len(T_pre) * n * len(T_post), 2, cx, hx, wx), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(T_pre) * n * len(T_post), cy, hy, wy), dtype=src_labels.dtype)
    idx = 0
    for t_pre in T_pre:
        for t_post in T_post:
            print(f'   - Writing ({idx+1}/{len(T_pre)*len(T_post)}) pre {dates[t_pre]} & post {dates[t_post]}')
            dst_images[idx*n:(idx+1)*n,0,:] = src_images[:,t_pre,:]
            dst_images[idx*n:(idx+1)*n,1,:] = src_images[:,t_post,:]
            dst_labels[idx*n:(idx+1)*n,:]   = src_labels[:,t_post,:]
            idx += 1

del sample, src_images, src_labels, dst_images, dst_labels, n, T, T_pre, T_post, t_pre, t_post, cx, cy, hx, hy, wx, wy, idx

#%% RESHAPES THE SEQUENCES DATASET INTO THE TILES DATASET

print('Reshaping the sequences dataset into the tiles dataset')
for sample in ['train', 'valid', 'test']:
    print(f' - Processing {sample} sample')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_tile_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_tile_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')
    src_labels = zarr.open(src_labels, mode='r')
    n, T, cx, hx, wx = src_images.shape
    n, T, cy, hy, wy = src_labels.shape
    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(n * T, cx, hx, wx), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(n * T, cy, hy, wy), dtype=src_labels.dtype)
    for t in range(T):
        dst_images[n*t:(t+1)*n,:] = src_images[:,t,:]
        dst_labels[n*t:(t+1)*n,:] = src_labels[:,t,:]

del sample, src_images, src_labels, dst_images, dst_labels, n, T, cx, cy, hx, hy, wx, wy, t

#%% BALANCES THE SEQUENCE DATASET BY DOWNSAMPLING NO-DESTRUCTION SEQUENCES

print('Downsampling no-destruction sequences')
for sample in ['train', 'valid', 'test']:
    print(f' - Processing sample {sample}')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}_balanced.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_sequence_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}_balanced.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')
    total_samples = src_images.shape[0]
    src_labels = zarr.open(src_labels, mode='r')[:]
    # Subsets source datasets
    destroy = [k for k, v in params.label_map.items() if v != 0 and k != 255]
    destroy = np.any(np.isin(src_labels, destroy), axis=(1, 2, 3, 4))
    #destroy = (np.sum(np.isin(src_labels, destroy), axis=1) > 0).flatten()
    untouch = np.where(~destroy)[0]
    
    # Downsampling
    requested = int(params.prepost_ratio * sum(destroy))  # target #negatives
    k = min(len(untouch), requested)                      # cap to available
    indices = np.concatenate((
        np.where(destroy)[0], # Includes all destroyed samples
        np.random.choice(untouch, k, replace=False))) 
    
    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(len(indices), *src_images.shape[1:]), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(indices), *src_labels.shape[1:]), dtype=src_labels.dtype)
    
    # Check if the data is too large to be processed at once
    if src_images.shape[0] > params.chunk_size:
        print(f"\t - Number of sequences {src_images.shape[0]} will be loaded in chunks of {params.chunk_size}")
        # Sort the desired indices so that we can iterate over them in order.
        indices_sorted = np.sort(indices)
        pos_dst = 0  # Keeps track of where to write in the destination array

        # Iterate over the source array in chunks
        for start in range(0, total_samples, params.chunk_size):
            stop = min(total_samples, start + params.chunk_size)
            
            # Find all indices from indices_sorted that fall in the current chunk [start, stop)
            mask = (indices_sorted >= start) & (indices_sorted < stop)
            if not np.any(mask):
                continue  # no desired indices in this chunk
            
            # Get the indices (local to the chunk) that we want to extract
            local_indices = indices_sorted[mask] - start
            
            # Load the current chunk from the source (only this chunk is read into memory)
            chunk_data = src_images[start:stop]
            
            # Extract the required images from the chunk using local indices
            extracted = chunk_data[local_indices]
            
            # Write them into the destination array at the correct position
            dst_images[pos_dst:pos_dst + len(extracted)] = extracted
            pos_dst += len(extracted)
        
        # Set the destination labels to the same indices as the destination images
        dst_labels[:] = src_labels[indices_sorted]

        # Now, shuffle the destination arrays so that the order is randomized.
        perm = np.random.permutation(dst_images.shape[0])
        # Shuffle images.
        shuffled_images = dst_images[:]  # load entire destination images into memory
        dst_images[:] = shuffled_images[perm]
        
        # Shuffle labels in the same order.
        shuffled_labels = dst_labels[:]  # load entire destination labels into memory
        dst_labels[:] = shuffled_labels[perm]
    
    # If the data can be processed at once    
    else: 
        print(f"\t - Number of sequences {src_images.shape[0]} will be loaded at once")
        np.random.shuffle(indices)
        src_images = src_images[:]
        dst_images[:] = src_images[indices]
        dst_labels[:] = src_labels[indices]
    
del sample, src_images, src_labels, dst_images, dst_labels, destroy, untouch, indices

#%% BALANCES THE PRE-POST DATASET BY DOWNSAMPLING NO-DESTRUCTION PRE-POST PAIRS

print('Downsampling no-destruction pre-post pairs')
for sample in ['train', 'valid', 'test']:
    print(f' - Processing {sample} sample')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_prepost_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_prepost_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_prepost_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}_balanced.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_prepost_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}_balanced.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')
    total_samples = src_images.shape[0]
    src_labels = zarr.open(src_labels, mode='r')[:]
    # Subsets datasets
    destroy = [k for k, v in params.label_map.items() if v != 0 and k != 255] # v == 1
    destroy = np.any(np.isin(src_labels, destroy), axis=(1, 2, 3))
    #destroy = np.isin(src_labels, destroy).flatten()
    #untouch = np.where(np.logical_and(~destroy, src_labels.flatten() != 255))[0]
    untouch = np.where(~destroy)[0]
    
    # Downsampling
    requested = int(params.prepost_ratio * sum(destroy))  # target #negatives
    k = min(len(untouch), requested)                      # cap to available
    indices = np.concatenate((
        np.where(destroy)[0],
        np.random.choice(untouch, k, replace=False))) 

    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(len(indices), *src_images.shape[1:]), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(indices), *src_labels.shape[1:]), dtype=src_labels.dtype)
    
    # Check if the data is too large to be processed at once
    if src_images.shape[0] > params.chunk_size:
        print(f"\t - Number of sequences {src_images.shape[0]} will be loaded in chunks of {params.chunk_size}")
        # Sort the desired indices so that we can iterate over them in order.
        indices_sorted = np.sort(indices)
        pos_dst = 0  # Keeps track of where to write in the destination array

        # Iterate over the source array in chunks
        for start in range(0, total_samples, params.chunk_size):
            stop = min(total_samples, start + params.chunk_size)
            
            # Find all indices from indices_sorted that fall in the current chunk [start, stop)
            mask = (indices_sorted >= start) & (indices_sorted < stop)
            if not np.any(mask):
                continue  # no desired indices in this chunk
            
            # Get the indices (local to the chunk) that we want to extract
            local_indices = indices_sorted[mask] - start
            
            # Load the current chunk from the source (only this chunk is read into memory)
            chunk_data = src_images[start:stop]
            
            # Extract the required images from the chunk using local indices
            extracted = chunk_data[local_indices]
            
            # Write them into the destination array at the correct position
            dst_images[pos_dst:pos_dst + len(extracted)] = extracted
            pos_dst += len(extracted)
        
        # Set the destination labels to the same indices as the destination images
        dst_labels[:] = src_labels[indices_sorted]

        # Now, shuffle the destination arrays so that the order is randomized.
        perm = np.random.permutation(dst_images.shape[0])
        # Shuffle images.
        shuffled_images = dst_images[:]  # load entire destination images into memory
        dst_images[:] = shuffled_images[perm]
        
        # Shuffle labels in the same order.
        shuffled_labels = dst_labels[:]  # load entire destination labels into memory
        dst_labels[:] = shuffled_labels[perm]
    
    # If the data can be processed at once    
    else: 
        print(f"\t - Number of sequences {src_images.shape[0]} will be loaded at once")
        np.random.shuffle(indices)
        src_images = src_images[:]
        dst_images[:] = src_images[indices]
        dst_labels[:] = src_labels[indices]
del sample, src_images, src_labels, dst_images, dst_labels, destroy, untouch, indices

#%% BALANCES THE TILE DATASET BY DOWNSAMPLING NO-DESTRUCTION TILES

print('Downsampling no-destruction tiles')
for sample in ['train', 'valid', 'test']:
    print(f' - Processing {sample} sample')
    # Defines datasets paths
    src_images = f'{paths.data}/{params.city}/zarr/images_tile_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    src_labels = f'{paths.data}/{params.city}/zarr/labels_tile_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}.zarr'
    dst_images = f'{paths.data}/{params.city}/zarr/images_tile_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}_balanced.zarr'
    dst_labels = f'{paths.data}/{params.city}/zarr/labels_tile_img{params.image_size}_pat{params.patch_size}_buf{params.buffer_around_destruction}_{sample}_balanced.zarr'
    # Reads source datasets
    src_images = zarr.open(src_images, mode='r')
    total_samples = src_images.shape[0]
    src_labels = zarr.open(src_labels, mode='r')[:]
    # Subsets datasets
    destroy = [k for k, v in params.label_map.items() if v != 0 and k != 255] # v==1
    #destroy = np.isin(src_labels, destroy).flatten()
    destroy = np.any(np.isin(src_labels, destroy), axis=(1, 2, 3))
    #untouch = np.where(np.logical_and(~destroy, src_labels.flatten() != 255))[0]
    untouch = np.where(~destroy)[0]
    indices = np.concatenate((
        np.where(destroy)[0],
        np.random.choice(untouch, params.tile_ratio * np.sum(destroy), replace=False)))
    
    # Writes destination datasets
    dst_images = zarr.open(dst_images, mode='w', shape=(len(indices), *src_images.shape[1:]), dtype=src_images.dtype)
    dst_labels = zarr.open(dst_labels, mode='w', shape=(len(indices), *src_labels.shape[1:]), dtype=src_labels.dtype)
    
    # Check if the data is too large to be processed at once
    if src_images.shape[0] > params.chunk_size:
        print(f"\t - Number of sequences {src_images.shape[0]} will be loaded in chunks of {params.chunk_size}")
        # Sort the desired indices so that we can iterate over them in order.
        indices_sorted = np.sort(indices)
        pos_dst = 0  # Keeps track of where to write in the destination array

        # Iterate over the source array in chunks
        for start in range(0, total_samples, params.chunk_size):
            stop = min(total_samples, start + params.chunk_size)
            
            # Find all indices from indices_sorted that fall in the current chunk [start, stop)
            mask = (indices_sorted >= start) & (indices_sorted < stop)
            if not np.any(mask):
                continue  # no desired indices in this chunk
            
            # Get the indices (local to the chunk) that we want to extract
            local_indices = indices_sorted[mask] - start
            
            # Load the current chunk from the source (only this chunk is read into memory)
            chunk_data = src_images[start:stop]
            
            # Extract the required images from the chunk using local indices
            extracted = chunk_data[local_indices]
            
            # Write them into the destination array at the correct position
            dst_images[pos_dst:pos_dst + len(extracted)] = extracted
            pos_dst += len(extracted)
        
        # Set the destination labels to the same indices as the destination images
        dst_labels[:] = src_labels[indices_sorted]

        # Now, shuffle the destination arrays so that the order is randomized.
        perm = np.random.permutation(dst_images.shape[0])
        # Shuffle images.
        shuffled_images = dst_images[:]  # load entire destination images into memory
        dst_images[:] = shuffled_images[perm]
        
        # Shuffle labels in the same order.
        shuffled_labels = dst_labels[:]  # load entire destination labels into memory
        dst_labels[:] = shuffled_labels[perm]
    
    # If the data can be processed at once    
    else: 
        print(f"\t - Number of sequences {src_images.shape[0]} will be loaded at once")
        np.random.shuffle(indices)
        src_images = src_images[:]
        dst_images[:] = src_images[indices]
        dst_labels[:] = src_labels[indices]

del sample, src_images, src_labels, dst_images, dst_labels, destroy, untouch, indices

#%%
