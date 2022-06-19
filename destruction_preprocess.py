#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Preprocessing script
@author: Clement Gorin, Arogya K
@contact: gorinclem@gmail.com; arogya@berkeley.edu
@version: 2022.06.02
'''

#%% HEADER

# Modules
import numpy as np
import geopandas
import rasterio
import re
import zarr
import gc
import os
import shutil
from rasterio import features
from destruction_utilities import *
import random


#%% 
## FUNCTIONS    
def tiled_profile(source:str, tile_size:tuple=(128, 128, 1)) -> dict:
    '''Computes raster profile for tiles'''
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size[0] == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size[1] == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size[0], affine[1], affine[2], affine[3], affine[4] * tile_size[1], affine[5])
    profile.update(width=profile['width'] // tile_size[0], height=profile['height'] // tile_size[0], count=tile_size[2], transform=affine)
    return profile

# Helper to save zarr
def save_zarr(data, type, suffix):
    p, i, w, h, b = data.shape
    print("Save ZARR:", type, suffix, data.shape)
    data = data.reshape(p*i, h, w, b)
    path = f'../data/{CITY}/others/{CITY}_{type}s_{suffix}.zarr'
    if not os.path.exists(path):
        zarr.save(path, data)        
    else:
        za = zarr.open(path, mode='a')
        za.append(data)

#%% 
# DECLARATION
CITY = 'aleppo'

#%% 
# COMPUTES TILE SAMPLES

# Files
image      = search_data(pattern(city=CITY, type='image'))[0]
settlement = search_data(f'{CITY}_settlement.*gpkg$')
noanalysis = search_data(f'{CITY}_noanalysis.*gpkg$')


# Computes analysis zone
profile    = tiled_profile(image, tile_size=(128, 128, 1))
settlement = rasterise(settlement, profile, dtype='bool')
noanalysis = rasterise(noanalysis, profile, dtype='bool')
analysis   = np.logical_and(settlement, np.invert(noanalysis))
del image, settlement, noanalysis

# Splits samples
np.random.seed(1)
index   = dict(training=0.70, validation=0.15, test=0.15)
index   = np.random.choice(np.arange(len(index)) + 1, np.sum(analysis), p=list(index.values()))
samples = analysis.astype(int)
np.place(samples, analysis, index)
write_raster(samples, profile, f'../data/{CITY}/others/{CITY}_samples.tif', nodata=-1, dtype='int8')
del index, samples, analysis


#%% 
# COMPUTES LABELS

# Reads damage reports
damage = search_data(f'{CITY}_damage.*gpkg$')
damage = geopandas.read_file(damage)

# Extract report dates
dates = search_data(pattern(city=CITY, type='image'))
dates = extract(dates, '\d{4}_\d{2}_\d{2}')
dates= list(map(lambda x: x.replace("_", "-"), dates))

# # Fills missing dates (!) Discuss (!)

damage[list(set(dates) - set(damage.columns))] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)
damage_geom = damage.geometry
damage = damage.drop(columns='geometry')
damage.insert(0,-1,0)
damage['9999'] = damage[damage.T.last_valid_index()]
values = damage.T.fillna(method='ffill').fillna(method='bfill')
temp = damage.T.fillna(method='bfill').fillna(method='ffill')
values[values != temp] = -1
damage = geopandas.GeoDataFrame(values.T.drop([-1, '9999'], axis=1), geometry=damage_geom)
damage = damage[dates + ['geometry']] # Drops dates not in images

# Writes damage labels
for date in dates:
    print(date)
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting takes the max per pixel
    subset = rasterise(subset, profile, date)
    write_raster(subset, profile, f'../data/{CITY}/labels/label_{date}.tif', nodata=-1, dtype='int8')
del date, subset

#%% 
# SAVE ND-ARRAYS (images)
samples = read_raster(f'../data/{CITY}/others/{CITY}_samples.tif')
images  = search_data(pattern(city=CITY, type='image'))
labels  = search_data(pattern(city=CITY, type='label'))

for i in range(len(images)):
    label = np.array(read_raster(labels[i]))
    w, h, b = label.shape
    label = label.reshape(1, w, h, b)
    label = tile_sequences(label, tile_size=(1,1))
    exclude = np.where(label.flatten() == -1)
    label = np.delete(label, exclude, 0)
    label[label!=3.0] = 0.0
    label[label==3.0] = 1.0
    noanalysis, train, validate, test = sample_split(label, np.delete(samples.flatten(), exclude))
    train_shuffle = np.arange(len(train))
    validate_shuffle = np.arange(len(validate))
    test_shuffle = np.arange(len(test))
    np.random.shuffle(train_shuffle)
    np.random.shuffle(validate_shuffle)
    np.random.shuffle(test_shuffle)
    save_zarr(train[train_shuffle], 'label', 'train')
    save_zarr(validate[validate_shuffle], 'label','validate')
    save_zarr(test[test_shuffle], 'label','test')
    del noanalysis, train, validate, test, label

    image = np.array(read_raster(images[i]))
    w,h,b = image.shape
    image = image.reshape(1,w,h,b)
    image = tile_sequences(image,  tile_size=(128, 128))
    image = np.delete(image, exclude, 0)
    noanalysis, train, validate, test = sample_split(image, np.delete(samples.flatten(), exclude))
    save_zarr(train[train_shuffle], 'image', 'train')
    save_zarr(validate[validate_shuffle], 'image','validate')
    save_zarr(test[test_shuffle], 'image','test') 
    del noanalysis, train, validate, test, image, exclude
    print(f'{dates[i]}')
    gc.collect(generation=2)

# %%
# GENERATE BALANCED DATA
def make_balanced(city, set):
    z_l = get_zarr(city, 'label', set)
    z_i = get_zarr(city, 'image', set)
    print(city, set)

    path_l = f'../data/{city}/others/{city}_labels_{set}_balanced.zarr'
    path_i = f'../data/{city}/others/{city}_images_{set}_balanced.zarr'

    zarr.save(path_l, z_l)
    zarr.save(path_i, z_i)

    z_l_positives = np.where(np.squeeze(z_l) == 1)[0]
    z_l_negatives = np.where(np.squeeze(z_l) == 0)[0]
    sample_length = len(z_l_negatives) - len(z_l_positives)
    indices = random.choices(z_l_positives, k=sample_length)

    # del z_i, z_l

    z_i_a = zarr.open(path_i, mode = 'a')
    z_l_a = zarr.open(path_l, mode = 'a')
    
    step_size = 5000
    for i, t in enumerate(make_tuple_pair(z_i.shape[0], step_size)):
        sub_indices = [num for num in indices if num >= t[0] and num < t[1]]
        sub_indices = list(map(lambda x: x-(i*step_size), sub_indices))
        to_add_i = z_i[t[0]:t[1]][sub_indices]
        to_add_l = z_l[t[0]:t[1]][sub_indices]
        z_i_a.append(to_add_i)
        z_l_a.append(to_add_l)
        print(t, to_add_i.shape)

    gc.collect()

def shuffle_balanced(city, set):
    path_l = f'../data/{city}/others/{city}_labels_{set}_balanced.zarr'
    path_i = f'../data/{city}/others/{city}_images_{set}_balanced.zarr'
    path_l_s = f'../data/{city}/others/{city}_labels_{set}_balanced_shuffled.zarr'
    path_i_s = f'../data/{city}/others/{city}_images_{set}_balanced_shuffled.zarr'
    
    z_l = zarr.open(path_l)
    z_i = zarr.open(path_i)
    n = z_l.shape[0]
    tuple_pair = make_tuple_pair(n, 250)
    np.random.shuffle(tuple_pair)
    # print(tuple_pair)

    zarr.save(path_i_s, np.empty((0,128,128,3)))
    zarr.save(path_l_s, np.empty((0,1,1,1)))

    z_i_s = zarr.open(path_i_s, mode='a')
    z_l_s = zarr.open(path_l_s, mode='a')
    print(f"Reordering array in batches of 250. Total {len(tuple_pair)} sets..")
    for i, t in enumerate(tuple_pair):
        if i % 25 == 0:
            print(f"Finished {i} sets")
        images = z_l[t[0]:t[1]]
        labels = z_i[t[0]:t[1]]
        z_l_s.append(images)
        z_i_s.append(labels)
    shutil.rmtree(path_i)
    shutil.rmtree(path_l)

    del z_i_s, z_l_s, tuple_pair

    zarr.save(path_i, np.empty((0,128,128,3)))
    zarr.save(path_l, np.empty((0,1,1,1)))

    z_i = zarr.open(path_i, mode='a')
    z_l = zarr.open(path_l, mode='a')
    z_i_s = zarr.open(path_i_s)
    z_l_s = zarr.open(path_l_s)
    tuple_pair = make_tuple_pair(n, 10000)
    print(f"Shuffling array in batches of 10000. Total {len(tuple_pair)} sets..")
    for i, t in enumerate(tuple_pair):
        if i % 5 == 0:
            print(f"Finished {i} sets")
        shuffled = np.arange(0, 10000)
        np.random.shuffle(shuffled)
        images = z_i_s[t[0]:t[1]][shuffled]
        labels = z_l_s[t[0]:t[1]][shuffled]
        z_i.append(images)
        z_l.append(labels)
    shutil.rmtree(path_i_s)
    shutil.rmtree(path_l_s)




       
make_balanced(CITY, 'train')
shuffle_balanced(CITY, 'train')
 
# %%
