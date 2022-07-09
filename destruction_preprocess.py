#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Preprocessing script
@author: Clement Gorin, Arogya K
@contact: gorinclem@gmail.com; arogya@berkeley.edu
@version: 2022.06.02
'''

#%% HEADER

# We load required packages..
print('\n--- We load required packages..')
import numpy as np
import geopandas
import rasterio
import re
import zarr
import os
from rasterio import features
from destruction_utilities import *
import time
import shutil

#%% 
# Declare variables..
print('--- Declare variables..')

CITY = 'aleppo'
TILE_SIZE = (128,128)

# Convolutional Network Settings
ZERO_DAMAGE_BEFORE_YEAR = 2012

# Siamese Network Settings
PRE_IMG_INDEX = 0

#%% 
## Declare functions..  
print('--- Declare functions..')
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

#%% 
# Split tiles into train, test, and validate..
print('--- Split tiles into train, test, and validate..')

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
# Calculate labels for each tile..
print('--- Calculate labels for each tile..')
# Reads damage reports
damage = search_data(f'{CITY}_damage.*gpkg$')
damage = geopandas.read_file(damage)
last_annotation_date = sorted(damage.columns)[-2]

# Extract report dates
dates = search_data(pattern(city=CITY, type='image'))
dates = extract(dates, '\d{4}_\d{2}_\d{2}')
dates= list(map(lambda x: x.replace("_", "-"), dates))

# Forward and back-filling based on annotation, mapping and dropping uncertain (-1) class tiles
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
    print(f'------ {date}')
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting takes the max per pixel
    subset = rasterise(subset, profile, date)
    write_raster(subset, profile, f'../data/{CITY}/labels/label_{date}.tif', nodata=-1, dtype='int8')
del date, subset

#%% 
# Save images to disk as zarr (for CNN)..
print('--- Save images to disk as zarr (for CNN)..')

# SAVE ND-ARRAYS (images)
samples = read_raster(f'../data/{CITY}/others/{CITY}_samples.tif')
images  = search_data(pattern(city=CITY, type='image'))
labels  = search_data(pattern(city=CITY, type='label'))

delete_zarr_if_exists(CITY, 'labels_conv_train')
delete_zarr_if_exists(CITY, 'labels_conv_valid')
delete_zarr_if_exists(CITY, 'labels_conv_test')
delete_zarr_if_exists(CITY, 'images_conv_train')
delete_zarr_if_exists(CITY, 'images_conv_valid')
delete_zarr_if_exists(CITY, 'images_conv_test')

for i in range(len(images)):
    label = np.array(read_raster(labels[i]))
    w, h, b = label.shape
    label = label.reshape(1, w, h, b)
    label = tile_sequences(label, tile_size=(1,1))
    exclude = np.where(label.flatten() == -1)
    label = np.delete(label, exclude, 0)
    label[label!=3.0] = 0.0
    label[label==3.0] = 1.0
    _, train, validate, test = sample_split(label, np.delete(samples.flatten(), exclude))
    train_shuffle = np.arange(len(train))
    validate_shuffle = np.arange(len(validate))
    test_shuffle = np.arange(len(test))
    np.random.shuffle(train_shuffle)
    np.random.shuffle(validate_shuffle)
    np.random.shuffle(test_shuffle)
    save_zarr(train[train_shuffle].reshape(np.take(train.shape, [0,2,3,4])), CITY, 'labels_conv_train')
    save_zarr(validate[validate_shuffle].reshape(np.take(validate.shape, [0,2,3,4])), CITY, 'labels_conv_valid')
    save_zarr(test[test_shuffle].reshape(np.take(test.shape, [0,2,3,4])), CITY, 'labels_conv_test')
    del _, train, validate, test, label

    image = np.array(read_raster(images[i]))
    w,h,b = image.shape
    image = image.reshape(1,w,h,b)
    image = tile_sequences(image,  tile_size=TILE_SIZE)
    image = np.delete(image, exclude, 0)
    _, train, validate, test = sample_split(image, np.delete(samples.flatten(), exclude))
    save_zarr(train[train_shuffle].reshape(np.take(train.shape, [0,2,3,4])), CITY, 'images_conv_train')
    save_zarr(validate[validate_shuffle].reshape(np.take(validate.shape, [0,2,3,4])), CITY,'images_conv_valid')
    save_zarr(test[test_shuffle].reshape(np.take(test.shape, [0,2,3,4])), CITY,'images_conv_test') 
    del _, train, validate, test, image, exclude
    print(f'------ {dates[i]}')
    gc.collect(generation=2)
del samples, images, labels

#%% 
# Generate a balanced (upsampled) dataset and shuffle it..
print('\n--- Generate a balanced (upsampled) dataset and shuffle it..')
delete_zarr_if_exists(CITY, 'labels_conv_train_balanced')
delete_zarr_if_exists(CITY, 'images_conv_train_balanced')
delete_zarr_if_exists(CITY, 'labels_conv_train_balanced_shuffled')
delete_zarr_if_exists(CITY, 'images_conv_train_balanced_shuffled')
balance(CITY)
shuffle(CITY, TILE_SIZE, (1000,7500))

#%% 
# Save images to disk as zarr (for SNN)..
print('\n--- Save images to disk as zarr (for SNN)..')
samples = read_raster(f'../data/{CITY}/others/{CITY}_samples.tif')
samples = samples.flatten()

images  = search_data(pattern('aleppo', 'image'))

delete_zarr_if_exists(CITY, 'images_siamese_train_tt')
delete_zarr_if_exists(CITY, 'images_siamese_test_tt')
delete_zarr_if_exists(CITY, 'images_siamese_valid_tt')
delete_zarr_if_exists(CITY, 'images_siamese_train_t0')
delete_zarr_if_exists(CITY, 'images_siamese_test_t0')
delete_zarr_if_exists(CITY, 'images_siamese_valid_t0')

for i, image in enumerate(images):
    t = len(images)
    if i != PRE_IMG_INDEX:
        img = read_raster(image, dtype='uint8')
        img = tile_sequences(np.array([img]), TILE_SIZE)
        _, img_train, img_test, img_valid = sample_split(img, samples)
        save_zarr(flatten_image(img_train), CITY, 'images_siamese_train_tt')
        save_zarr(flatten_image(img_test), CITY, 'images_siamese_test_tt')
        save_zarr(flatten_image(img_valid), CITY, 'images_siamese_valid_tt')
        del _, img_train, img_test, img_valid
    else:
        img = read_raster(image, dtype='uint8')
        img = tile_sequences(np.array([img]), TILE_SIZE)
        for j in range(0,t-1):            
            _, img_train, img_test, img_valid = sample_split(img, samples)
            save_zarr(flatten_image(img_train), CITY, 'images_siamese_train_t0')
            save_zarr(flatten_image(img_test), CITY, 'images_siamese_test_t0')
            save_zarr(flatten_image(img_valid), CITY, 'images_siamese_valid_t0')
            del _, img_train, img_test, img_valid
        
delete_zarr_if_exists(CITY, 'labels_siamese_train')
delete_zarr_if_exists(CITY, 'labels_siamese_test')
delete_zarr_if_exists(CITY, 'labels_siamese_valid')

labels  = search_data(pattern('aleppo', 'label'))
for i, label in enumerate(labels):
    t = len(labels)
    if i != PRE_IMG_INDEX:
        lbl = read_raster(label, dtype='uint8')
        lbl = tile_sequences(np.array([lbl]), (1,1))
        _, lbl_train, lbl_test, lbl_valid = sample_split(lbl, samples)
        save_zarr(lbl_train.flatten(), CITY, 'labels_siamese_train')
        save_zarr(lbl_test.flatten(), CITY, 'labels_siamese_test')
        save_zarr(lbl_valid.flatten(), CITY, 'labels_siamese_valid')