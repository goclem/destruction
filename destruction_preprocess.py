#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Optimises models
@author: Clement Gorin, Arogya Koirala
@contact: gorinclem@gmail.com
@version: 2022.05.24
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
image      = search_data(pattern(city='aleppo', type='image'))[0]
settlement = search_data('aleppo_settlement.*gpkg$')
noanalysis = search_data('aleppo_noanalysis.*gpkg$')

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
write_raster(samples, profile, '../data/aleppo/others/aleppo_samples.tif', nodata=-1, dtype='int8')
del index, samples, analysis

#%% COMPUTES LABELS

# Reads damage reports
damage = search_data('aleppo_damage.*gpkg$')
damage = geopandas.read_file(damage)

# Extract report dates
dates = search_data(pattern(city='aleppo', type='image'))
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
    write_raster(subset, profile, f'../data/aleppo/labels/label_{date}.tif', nodata=-1, dtype='int8')
del dates, date, subset
# %%

#% ONE TIME PREP SCRIPTS

# Prepare damage data in the format needed by ML architecture
def prep_damage(city:str, suffix:str) -> None:
    '''One time run to prep damage data for a given city'''
    path = search_data(city+'_damage.*gpkg')
    damage = gpd.read_file(path)
    
    sensor_date_columns = [col for col in damage.columns if 'SensDt' in col]
    damage_class_columns = [col for col in damage.columns if 'DmgCls' in col and 'GrpDmgCls' not in col ]
    
    
    for i, sensor_date_col in enumerate(sensor_date_columns):
        if i==0:
            allDates = damage[sensor_date_col]
        else:
            allDates = allDates.append(damage[sensor_date_col])

    sensor_date_values = allDates.unique()
    sensor_date_values = sensor_date_values[sensor_date_values != np.array(None)]
    
    new_damage = []
    for i, row in damage.iterrows():

        row_entry = {}
        row_entry['geometry'] = row['geometry']

        for j, sensor_date_col in enumerate(sensor_date_columns):
            if(row[sensor_date_col] != None):
                row_entry[row[sensor_date_col]] = row[damage_class_columns[j]]

        new_damage.append(row_entry)


    df = gpd.GeoDataFrame(new_damage)
    df = df.replace({np.nan: 0, "Destroyed": 3, "Severe Damage": 2, "Moderate Damage": 1, "No Visible Damage": 0})
    df.to_file(path.split(".gpkg")[0] + suffix + ".gpkg", driver="GPKG")

# Prepare settlement data in the format needed by ML architecture
def prep_settlement(city: str, suffix: str) -> None:
    '''One time manual run to prep settlement data for a given city'''
    path = search_data(city+'_settlement.*gpkg')
    settlement = gpd.read_file(path)
    settlement = settlement.filter(['geometry'])
    settlement.to_file(path.split(".gpkg")[0] + suffix + ".gpkg", driver="GPKG")
    
# Prepare noanalysis data in the format needed by ML architecture
def prep_noanalysis(city: str, suffix: str) -> None:
    '''One time manual run to prep noanalysis data for a given city'''
    path = search_data(city+'_noanalysis.*gpkg')
    noanalysis = gpd.read_file(path)
    noanalysis.columns = ['reason', 'geometry']
    noanalysis.to_file(path.split(".gpkg")[0] + suffix + ".gpkg", driver="GPKG")
    
# Helper to prepare all data for a given city at once
def prep_all(city:str, suffix: str) -> None:
    '''One time script to prep damage, noanalysis, and settlement data'''
    prep_damage(city, suffix)
    prep_settlement(city, suffix)
    prep_noanalysis(city, suffix)