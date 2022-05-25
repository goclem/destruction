#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Optimises models
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.25
'''

#%% HEADER

# Modules
import numpy as np
import geopandas
import rasterio
import re

from numpy import random
from os import path
from rasterio import features
from destruction_utilities import *

#%% FUNCTIONS
    
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

#%% COMPUTES TILE SAMPLES

# Files
city       = 'aleppo'
image      = search_data(pattern(city=city, type='image'))[0]
settlement = search_data(f'{city}_settlement\\.gpkg$')
noanalysis = search_data(f'{city}_noanalysis\\.gpkg$')

# Computes analysis zone
profile    = tiled_profile(image, tile_size=(128, 128, 1))
settlement = rasterise(settlement, profile, dtype='bool')
noanalysis = rasterise(noanalysis, profile, dtype='bool')
analysis   = np.logical_and(settlement, np.invert(noanalysis))
del image, settlement, noanalysis

# Splits samples
random.seed(1)
index   = dict(training=0.70, validation=0.15, test=0.15)
index   = np.random.choice(np.arange(len(index)) + 1, np.sum(analysis), p=list(index.values()))
samples = analysis.astype(int)
np.place(samples, analysis, index)
write_raster(samples, profile, f'../data/{city}/others/{city}_samples.tif', nodata=-1, dtype='int8')
del index, samples, analysis

#%% COMPUTES LABELS

# Reads damage reports
damage = search_data(f'{city}_damage.*gpkg$')
damage = geopandas.read_file(damage)

# Extract report dates
dates = search_data(pattern(city=city, type='image'))
dates = extract(dates, '\d{4}-\d{2}-\d{2}')

# Fills missing dates (!) Discuss (!)
damage[list(set(dates) - set(damage.columns))] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)
values = damage.drop(columns='geometry').T
values.fillna(method='ffill', inplace=True)
values.fillna(method='bfill', inplace=True)
damage = geopandas.GeoDataFrame(values.T.astype(int), geometry=damage.geometry)
damage = damage[dates + ['geometry']] # Drops dates not in images
del values

# Writes damage labels
for date in dates:
    print(date)
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting takes the max per pixel
    subset = rasterise(subset, profile, date)
    write_raster(subset, profile, f'../data/{city}/labels/label_{date}.tif', nodata=-1, dtype='int8')
del dates, date, subset
# %%
