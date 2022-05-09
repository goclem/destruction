#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.09
'''

#%% HEADER

# Modules
import geopandas
import numpy as np
import pandas as pd
import os
import rasterio
import re

from typing import Union
from matplotlib import pyplot
from rasterio import features
from tensorflow.keras import utils

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

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif'):
    '''Regular expressions for search_data'''
    return f'^.*{city}/.*/{type}_{date}\.{ext}$'

#%% RASTER UTILITIES

def read_raster(source:str, band:int=None, window=None, dtype:str='uint8') -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    return image

def write_raster(array:np.ndarray, profile:Union[str, dict], destination:str, nodata:int=None, dtype:str='uint8') -> None:
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

def rasterise(source, profile:tuple, attribute:str=None, dtype:str='uint8') -> np.ndarray:
    '''Tranforms vector data into raster'''
    if isinstance(source, str): 
        source = geopandas.read_file(source)
    if isinstance(profile, str): 
        profile = rasterio.open(profile).profile
    geometries = source['geometry']
    if attribute is not None:
        geometries = zip(geometries, source[attribute])
    image  = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    image  = image.astype(dtype)
    return image

#%% ARRAY UTILITIES

def images_to_tiles(images:np.ndarray, tile_size:tuple=(128, 128)) -> np.ndarray:
    '''Converts images to tiles of a given size'''
    n_images, image_width, image_height, n_bands = images.shape
    tile_width, tile_height = tile_size
    n_tiles_width  = (image_width  // tile_width)
    n_tiles_height = (image_height // tile_height)
    tiles = images.reshape(n_images, n_tiles_width, tile_width, n_tiles_height, tile_height, n_bands).swapaxes(2, 3)
    tiles = tiles.reshape(-1, tile_width, tile_height, n_bands)
    return tiles

def index_empty(tiles:np.ndarray, value:int=255) -> bool:
    '''Indexes empty blocks'''
    empty = np.full(tiles.shape[1:], value)
    index = [np.equals(tiles, empty).all() for tile in tiles]
    return index

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
