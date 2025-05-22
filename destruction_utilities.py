#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import argparse
import geopandas as gpd
import math
import numpy as np
import os
import rasterio
import re
import shutil
import time
import torch
import zarr

from matplotlib import pyplot as plt
from rasterio import enums, features, windows
from torch import nn, utils
from torch.nn import functional as F
from torcheval import metrics

#%% PATH UTILITIES

home  = os.path.expanduser('~')
paths = argparse.Namespace(
    data='../data',
    models='../models',
    figures='../figures',
    desktop=os.path.join(home, 'Desktop'),
    temporary=os.path.join(home, 'Desktop', 'temporary')
)
del home

#%% FILE AND FOLDER UTILITIES

def search_data(pattern:str='.*', directory:str=paths.data) -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files

def reset_folder(path:str, remove:bool=False) -> None:
    '''Resets a folder'''
    if remove:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(path)

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    regex = fr'^.*{city}/.*/{type}_{date}\.{ext}$'
    return regex

def extract(files:list, pattern:str=r'\d{4}-\d{2}-\d{2}') -> list:
    regex = re.compile(pattern)
    match = np.array([regex.search(file).group() for file in files])
    return match

#%% RASTER UTILITIES

def read_raster(source:str, band:int=None, window=None, dtype:str='float') -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is None:
        image = raster.read(window=window)
    else: 
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    image = image.transpose([1, 2, 0]).astype(dtype)
    return image

def write_raster(array:np.ndarray, profile, destination:str, update:dict=None) -> None:
    '''Writes a numpy array as a raster'''
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    array = array.transpose([2, 0, 1])
    count, height, width = array.shape
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    if update is not None:
        profile.update(**update)
    profile.update(count=count)
    with rasterio.open(fp=destination, mode='w', **profile) as raster:
        raster.write(array)
        raster.close()

def rasterise(source, profile, layer:str=None, varname:str=None, all_touched:bool=False, merge_alg:str='replace', update:dict=None) -> np.ndarray:
    '''Rasterises a vector file'''
    if isinstance(source, str):
        source = gpd.read_file(source, layer=layer)
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    if update is not None: 
        profile.update(count=1, **update)
    geometries = source['geometry']
    if varname is not None: 
        geometries = zip(geometries, source[varname])
    image = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'], all_touched=all_touched, merge_alg=enums.MergeAlg(merge_alg.upper()), dtype=profile['dtype'])
    image = np.expand_dims(image, 2)
    return image

def center_window(source:str, size:dict) -> windows.Window:
    '''Computes the windows for the centre of a raster'''
    profile = rasterio.open(source).profile
    centre  = (profile['width'] // 2, profile['height'] // 2)
    window  = windows.Window.from_slices(
        (centre[0] - size[0] // 2, centre[0] + size[0] // 2),
        (centre[1] - size[1] // 2, centre[1] + size[1] // 2)
    )
    return window

def tiled_profile(source:str, tile_size:int=224, crop_size:int=224, return_window:bool=False) -> dict:
    with rasterio.open(source) as raster:
        profile = raster.profile.copy()
        width   = profile['width']  - (profile['width']  % crop_size)
        height  = profile['height'] - (profile['height'] % crop_size)
        window  = rasterio.windows.Window(0, 0, width, height)
        transform = raster.window_transform(window)
        transform = rasterio.Affine(transform.a * tile_size, transform.b, transform.c, transform.d, transform.e * tile_size, transform.f)
        profile.update(width=width // tile_size, height=height // tile_size, transform=transform)
        if return_window:
            return profile, window
        else:
            return profile

#%% ARRAY UTILITIES

def image_to_tiles(image:torch.Tensor, tile_size:int, stride:int=None):
    '''Converts an image tensor to a tensor of tiles'''
    if stride is None:
        stride = tile_size
    depth, height, width = image.size()
    tiles = image.unfold(1, tile_size, stride).unfold(2, tile_size, stride)
    ntiles_h, ntiles_w = tiles.size(1), tiles.size(2)
    tiles = tiles.moveaxis(0, 2).contiguous()
    tiles = tiles.view(-1, depth, tile_size, tile_size)
    return tiles

def load_sequences(files:list, tile_size:int, window:int=None, stride:int=None) -> torch.Tensor:
    '''Loads a sequence of rasters as a tensor of tiles'''
    if window is not None:
        window = center_window(source=files[0], size=(window*tile_size, window*tile_size))
    sequences = [read_raster(file, window=window, dtype='uint8') for file in files]
    sequences = [torch.tensor(image).permute(2, 0, 1) for image in sequences]
    sequences = [image_to_tiles(image, tile_size=tile_size, stride=stride) for image in sequences]
    sequences = torch.stack(sequences).swapaxes(1, 0)
    return sequences

#%% DISPLAY UTILITIES
    
def display_image(image:torch.Tensor, title:str='', figsize=(10, 10), fontsize=15, path:str=None, dpi:int=300) -> None:
    image   = torch.einsum('chw -> hwc', image)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    ax.set_title(title, fontsize=fontsize)
    ax.set_axis_off()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=dpi)
    else:
        plt.show()
    plt.close()

def display_sequence(images:torch.Tensor, titles:list=[''], figsize=(10, 10), fontsize=20, path:str=None, dpi:int=300) -> None:
    images = torch.einsum('nchw -> nhwc', images)
    nimage = len(images)
    if len(titles) == 1:
        titles = titles * nimage
    fig, axs = plt.subplots(nrows=1, ncols=nimage, figsize=(figsize[1] * nimage, figsize[0]))
    for ax, image, title in zip(axs.ravel(), images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=fontsize)
        ax.set_axis_off()
    plt.tight_layout(pad=2.0)
    if path is not None:
        plt.savefig(path, dpi=dpi)
    else:
        plt.show()
    plt.close()

def display_grid(images:torch.Tensor, titles:list=[''], gridsize:tuple=(3, 3), figsize:tuple=(15, 15), fontsize=15, suptitle:str=None, path:str=None, dpi:int=300) -> None:
    images = torch.einsum('nchw -> nhwc', images)
    if len(titles) == 1: 
        titles = titles * np.prod(gridsize)
    fig, axs = plt.subplots(nrows=gridsize[0], ncols=gridsize[1], figsize=figsize)
    for ax, image, title in zip(axs.ravel(), images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=fontsize)
        ax.set_axis_off()
        plt.tight_layout(pad=2)
    if suptitle is not None:
        fig.suptitle(suptitle, y=1.05, fontsize=fontsize*2)
    if path is not None:
        plt.savefig(path, dpi=dpi)
    else:
        plt.show()
    plt.close()

#%% DATASET UTILITIES

def shuffle_zarr(images_zarr:str, labels_zarr:str=None) -> None:
    '''Shuffles a Zarr array along the first axis'''
    # Images
    images  = zarr.open(images_zarr, mode='r')[:]
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images  = images[indices]
    dataset = zarr.open(images_zarr, shape=images.shape, dtype=images.dtype, mode='w')
    dataset[:] = images
    # Labels
    if labels_zarr is not None:
        labels  = zarr.open(labels_zarr, mode='r')[:]
        labels  = labels[indices]
        dataset = zarr.open(labels_zarr, shape=labels.shape, dtype=labels.dtype, mode='w')
        dataset[:] = labels

#%% MODEL UTILITIES

def empty_cache(device:torch.device) -> None:
    '''Empties the cache of a device'''
    if device == 'cuda':
        torch.cuda.empty_cache()
    if device == 'mps':
        torch.mps.empty_cache()

#%%
