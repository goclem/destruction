#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Optimises models
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.09
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
    
def tiled_profile(source:str, tile_size:tuple=(128, 128, 1)):
    '''Computes raster profile for tiles'''
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size[0] == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size[1] == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size[0], affine[1], affine[2], affine[3], affine[4] * tile_size[1], affine[5])
    profile.update(width=profile['width'] // tile_size[0], height=profile['height'] // tile_size[0], count=tile_size[2], transform=affine)
    return profile

def rasterise(source:str, profile:tuple, attribute:str=None, dtype:str='uint8'):
    '''Tranforms vector data into raster'''
    vector     = geopandas.read_file(source)
    geometries = vector['geometry']
    if attribute is not None:
        geometries = zip(geometries, vector[attribute])
    image  = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    image  = image.astype(dtype)
    return image

#%% COMPUTES TILES

# Files
image      = search_data(pattern(city='aleppo', type='image'))[0]
settlement = search_data('aleppo_settlement.*gpkg$')
noanalysis = search_data('aleppo_noanalysis.*gpkg$')

# Computes analysis zone
profile    = tiled_profile(source=image, tile_size=(128, 128, 1))
settlement = rasterise(source=settlement, profile=profile, dtype='bool')
noanalysis = rasterise(source=noanalysis, profile=profile, dtype='bool')
analysis   = np.logical_and(settlement, np.invert(noanalysis))
del image, settlement, noanalysis

# Splits samples
random.seed(1)
index   = dict(training=0.70, validation=0.15, test=0.15)
index   = np.random.choice(np.arange(len(index)) + 1, np.sum(analysis), p=list(index.values()))
samples = analysis.astype(int)
np.place(samples, analysis, index)
del index

# Saves sample raster
write_raster(samples, profile, '../data/aleppo/vectors/aleppo_samples.tif', nodata=0)
del samples

#%% COMPUTES LABELS

# Reads damage reports
damage = search_data('aleppo_damage.*gpkg$')
damage = geopandas.read_file(damage)

# Extract report dates
regex = re.compile('\d{4}-\d{2}-\d{2}')
dates = search_data(pattern(city='aleppo', type='image'))
dates = [regex.search(date).group() for date in dates]
dates = list(set(dates) - set(damage.columns))
del regex

# Fills missing dates
damage[dates] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)
values = damage.drop(columns='geometry').T
values.fillna(method='ffill', inplace=True)
values.fillna(method='bfill', inplace=True)
damage = geopandas.GeoDataFrame(values.T.astype(int), geometry=damage.geometry)
del values, dates

# Writes damage labels
dates = damage.columns[:-1]
for date in dates:
    print(date)
    subset = damage[[date, 'geometry']].sort_values(by=date)
    subset = zip(subset['geometry'], subset[date])
    subset = features.rasterize(subset, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    subset = np.where(analysis, subset, -1)
    write_raster(subset, profile, f'../data/aleppo/labels/label_{date}.tif', nodata=-1, dtype='int8')
del dates, date, subset