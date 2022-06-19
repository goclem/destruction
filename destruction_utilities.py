#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.11
'''

#%% HEADER

# Modules
import geopandas
import numpy as np
import pandas as pd
import os
import rasterio
import re

from matplotlib import pyplot
from rasterio import features, windows
from tensorflow.keras import utils
import zarr

#%% FILE UTILITIES

def search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    if len(files) == 1: files = files[0]
    return files

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    return f'^.*{city}/.*/{type}_{date}\.{ext}$'

def extract(files:list, pattern:str='\d{4}-\d{2}-\d{2}') -> list:
    pattern = re.compile(pattern)
    match   = [pattern.search(file).group() for file in files]
    return match

#%% RASTER UTILITIES

def read_raster(source:str, band:int=None, window=None, dtype:str=None) -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    return image

def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='uint8') -> None:
    '''Writes a numpy array as a raster'''
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    array = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = array.shape
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as raster:
        raster.write(array)
        raster.close()

def rasterise(source, profile, attribute:str=None, dtype:str='int8') -> np.ndarray:
    '''Tranforms vector data into raster'''
    if isinstance(source, str): 
        source = geopandas.read_file(source)
    if isinstance(profile, str): 
        profile = rasterio.open(profile).profile
    geometries = source['geometry']
    if attribute is not None:
        geometries = zip(geometries, source[attribute])
    image = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    image = image.astype(dtype)
    return image

def center_window(source:str, size:dict):
    '''Computes the windows for the centre of a raster'''
    profile = rasterio.open(source).profile
    centre  = (profile['width'] // 2, profile['height'] // 2)
    window  = windows.Window.from_slices(
        (centre[0] - size[0] // 2, centre[0] + size[0] // 2),
        (centre[1] - size[1] // 2, centre[1] + size[1] // 2)
    )
    return window

#%% ARRAY UTILITIES

def tile_sequences(images:np.ndarray, tile_size:tuple=(128, 128)) -> np.ndarray:
    '''Converts images to sequences of tiles'''
    n_images, image_width, image_height, n_bands = images.shape
    tile_width, tile_height = tile_size
    assert image_width  % tile_width  == 0
    assert image_height % tile_height == 0
    n_tiles_width  = (image_width  // tile_width)
    n_tiles_height = (image_height // tile_height)
    sequence = images.reshape(n_images, n_tiles_width, tile_width, n_tiles_height, tile_height, n_bands)
    sequence = np.moveaxis(sequence.swapaxes(2, 3), 0, 2)
    sequence = sequence.reshape(-1, n_images, tile_width, tile_height, n_bands)
    return sequence

def sample_split(images:np.ndarray, samples:dict) -> list:
    '''Splits the data structure into multiple samples'''
    samples = [images[samples == value, ...] for value in np.unique(samples)]
    return samples

#%% DISPLAY UTILITIES
    
def display(image:np.ndarray, title:str='', cmap:str='gray') -> None:
    '''Displays an image'''
    fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

def compare(images:list, titles:list=['Image'], cmaps:list=['gray']) -> None:
    '''Displays multiple images'''
    nimage = len(images)
    if len(titles) == 1:
        titles = titles * nimage
    if len(cmaps) == 1:
        cmaps = cmaps * nimage
    fig, axs = pyplot.subplots(nrows=1, ncols=nimage, figsize=(10, 10 * nimage))
    for ax, image, title, cmap in zip(axs.ravel(), images, titles, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

def display_structure(model, path:str) -> None:
    '''Displays keras model structure'''
    summary = pd.DataFrame([dict(Name=layer.name, Type=layer.__class__.__name__, Shape=layer.output_shape, Params=layer.count_params()) for layer in model.layers])
    summary.style.to_html(f'{path}.html', index=False) 
    utils.plot_model(model, to_file=f'{path}.pdf', show_shapes=True)
    
    
#% ONE TIME DATA PREP UTILS
# Prepare damage data in the format needed by ML architecture
def prep_damage(city:str, suffix:str, datadir:str) -> None:
    '''One time run to prep damage data for a given city'''
    path = search_data(city+'_damage.*gpkg', datadir)
    print(path)
    damage = geopandas.read_file(path)
    
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


    df = geopandas.GeoDataFrame(new_damage)
    df = df.replace({np.nan: 0, "Destroyed": 3, "Severe Damage": 2, "Moderate Damage": 1, "No Visible Damage": 0})
    df.to_file(path.split(".gpkg")[0] + suffix + ".gpkg", driver="GPKG")

# Prepare settlement data in the format needed by ML architecture
def prep_settlement(city: str, suffix: str, datadir: str) -> None:
    '''One time manual run to prep settlement data for a given city'''
    path = search_data(city+'_settlement.*gpkg', datadir)
    print(path)
    settlement = geopandas.read_file(path)
    settlement = settlement.filter(['geometry'])
    settlement.to_file(path.split(".gpkg")[0] + suffix + ".gpkg", driver="GPKG")
    
# Prepare noanalysis data in the format needed by ML architecture
def prep_noanalysis(city: str, suffix: str, datadir:str) -> None:
    '''One time manual run to prep noanalysis data for a given city'''
    path = search_data(city+'_noanalysis.*gpkg', datadir)
    print(path)
    noanalysis = geopandas.read_file(path)
    noanalysis.columns = ['reason', 'geometry']
    noanalysis.to_file(path.split(".gpkg")[0] + suffix + ".gpkg", driver="GPKG")
    
# Helper to prepare all data for a given city at once
def prep_all(city:str, suffix: str, datadir: str) -> None:
    '''One time script to prep damage, noanalysis, and settlement data'''
    prep_damage(city, suffix, datadir)
    prep_settlement(city, suffix, datadir)
    prep_noanalysis(city, suffix, datadir)



# Helper to get zarr file for a specified city, type (label or image), and set (test, train, validate)
def get_zarr(city, type, set):
    path = f'../data/{city}/others/{city}_{type}s_{set}.zarr'
    return zarr.open(path)

# Create tuple pairs for a given step size. used in balanced data generation
# Input = make_tuple_pair(16433, 5000)
# Output = [(0,5000), (5000, 10000), (10000,16433)] 
def make_tuple_pair(n, step_size):
    iters = n//step_size
    l = []
    for i in range(0, iters):
        if i == iters - 1:
            t = (i*step_size, n)
            l.append(t)
        else:
            t = (i*step_size, (i+1)*step_size)
            l.append(t)
    return l