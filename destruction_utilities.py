#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.00
'''

#%% HEADER

# Modules
import numpy as np
import pandas as pd
import os
import rasterio
import re

from matplotlib import pyplot
from tensorflow.keras import utils

#%% FILE UTILITIES

# Builds regular expression for search_data
def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif'):
    return f'^.*{city}/.*/{type}_{date}\.{ext}$'

# Sorted list of files in a directory that match a regular expression
def search_data(pattern:str='.*', directory:str='../data') -> list:
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files

#%% RASTER UTILITIES

# Reads a raster as an array
def read_raster(source:str, band:int=None, dtype:type=np.uint8) -> np.ndarray:
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read()
    image  = image.transpose([1, 2, 0]).astype(dtype)
    return image

# Writes an array as a raster
def write_raster(array:np.ndarray, source:str, destination:str, nodata:int=0, dtype:str='uint8') -> None:
    raster = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = raster.shape
    profile = rasterio.open(source).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as dest:
        dest.write(raster)
        dest.close()

#%% ARRAY UTILITIES

# Converts images to tiles of a given size
def images_to_tiles(images:np.ndarray, tile_size:tuple=(64, 64)) -> np.ndarray:
    # Defines quantities
    n_images, image_width, image_height, n_bands = images.shape
    tile_width, tile_height = tile_size
    n_tiles_width  = (image_width  // tile_width)
    n_tiles_height = (image_height // tile_height)
    # Maps images to tiles
    tiles = images.reshape(n_images, n_tiles_width, tile_width, n_tiles_height, tile_height, n_bands).swapaxes(2, 3)
    tiles = tiles.reshape(-1, tile_width, tile_height, n_bands)
    return tiles

# Converts tiles to images of a given size
def tiles_to_images(tiles:np.ndarray, image_size:tuple, tile_size:tuple=(64, 64), shift:bool=False) ->  np.ndarray:
    # Defines quantities
    n_images, image_width, image_height, n_bands = image_size
    tile_width, tile_height = tile_size
    n_tiles_width  = (image_width  // tile_width  + 1 + shift)
    n_tiles_height = (image_height // tile_height + 1 + shift)
    # Defines padding
    pad_width  = int(((n_tiles_width)  * tile_width  - image_width)  / 2)
    pad_height = int(((n_tiles_height) * tile_height - image_height) / 2)
    # Maps tiles to images
    images = tiles.reshape(-1, n_tiles_width, n_tiles_height, tile_width, tile_height, n_bands).swapaxes(2, 3)
    images = images.reshape(-1, (image_width + (2 * pad_width)), (image_height + (2 * pad_height)), n_bands)
    images = images[:, pad_width:image_width + pad_width, pad_height:image_height + pad_height, :]
    return images

# Filters empty blocks
def index_empty(tiles:np.ndarray, value:int=255) -> bool:
    empty = np.full(tiles.shape[1:], value)
    index = [np.equals(tiles, empty).all() for tile in tiles]
    return index

#%% DISPLAY UTILITIES
    
# Displays an image
def display(image:np.ndarray, title:str='', cmap:str='gray') -> None:
    fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

# Displays multiple images
def compare(images:list, titles:list=['Image'], cmaps:list=['gray']) -> None:
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

# Displays model structure
def display_structure(model, path:str) -> None:
    summary = pd.DataFrame([dict(Name=layer.name, Type=layer.__class__.__name__, Shape=layer.output_shape, Params=layer.count_params()) for layer in model.layers])
    summary.style.to_html(f'{path}.html', index=False) 
    utils.plot_model(model, to_file=f'{path}.pdf', show_shapes=True)
