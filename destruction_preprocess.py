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
import os
from rasterio import features
from destruction_utilities import *
import time

#%% 
# DECLARATION
CITY = 'aleppo'
TILE_SIZE = (128,128)
ZERO_DAMAGE_BEFORE_YEAR = 2012

#%% 
## FUNCTIONS    
def tiled_profile(source:str, tile_size:tuple=(*TILE_SIZE, 1)) -> dict:
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
# COMPUTES TILE SAMPLES

# Files
image      = search_data(pattern(city=CITY, type='image'))[0]
settlement = search_data(f'{CITY}_settlement.*gpkg$')
noanalysis = search_data(f'{CITY}_noanalysis.*gpkg$')


# Computes analysis zone
profile    = tiled_profile(image, tile_size=(*TILE_SIZE, 1))
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
last_annotation_date = sorted(damage.columns)[-2]

# Extract report dates
dates = search_data(pattern(city=CITY, type='image'))
dates = extract(dates, '\d{4}_\d{2}_\d{2}')
dates= list(map(lambda x: x.replace("_", "-"), dates))


damage[list(set(dates) - set(damage.columns))] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)
pre_cols = [col for col in damage.drop('geometry', axis=1).columns if int(col.split("-")[0]) < ZERO_DAMAGE_BEFORE_YEAR]
damage[pre_cols] = 0.0
post_cols = [col for col in damage.drop('geometry', axis=1).columns if time.strptime(col, "%Y-%m-%d") > time.strptime(last_annotation_date, "%Y-%m-%d")]
post_uncertain = list(*np.where(damage[last_annotation_date] == 0.0))
damage.loc[post_uncertain,post_cols] = -1.0
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
    image = tile_sequences(image,  tile_size=TILE_SIZE)
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
make_balanced(CITY, 'train')
shuffle_balanced(CITY, 'train', TILE_SIZE, 1000, 5000)

# %%
