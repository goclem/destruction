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

from matplotlib import pyplot
from rasterio import enums, features, windows
from torch import nn, utils
from torch.nn import functional as F
from torcheval import metrics

#%% PATH UTILITIES

home  = os.path.expanduser('~')
paths = argparse.Namespace(
    #data='/lustre/ific.uv.es/ml/iae091/data',
    #models='/lustre/ific.uv.es/ml/iae091/models',
    data= "../data",
    model= "../model",
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

def tiled_profile(source:str, tile_size:int) -> dict:
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size, affine[1], affine[2], affine[3], affine[4] * tile_size, affine[5])
    profile.update(width=profile['width'] // tile_size, height=profile['height'] // tile_size, count=tile_size, transform=affine)
    return profile

#%% ARRAY UTILITIES

def image_to_tiles(image:torch.Tensor, tile_size:int, stride:int=None) -> torch.Tensor:
    '''Converts an image tensor to a tensor of tiles'''
    depth, height, width = image.size()
    if stride is None: 
        stride = tile_size
    pad_h, pad_w = [math.ceil((dim - tile_size) / stride) * stride + tile_size - dim for dim in (height, width)]
    image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=255)
    tiles = image.unfold(1, tile_size, stride).unfold(2, tile_size, stride)
    tiles = tiles.moveaxis(0, 2).contiguous()
    tiles = tiles.view(-1, depth, tile_size, tile_size)
    return tiles

def tiles_to_image(tiles:torch.Tensor, image_size:int, stride:int=None) -> torch.Tensor:
    '''Converts a tensor of tiles to an image tensor'''
    depth, height, width = image_size
    tile_size = tiles.size(-1)
    if stride is None: 
        stride = tile_size
    pad_h, pad_w  = [math.ceil((dim - tile_size) / stride) * stride + tile_size - dim for dim in (height, width)]
    height, width = height + pad_h, width + pad_w
    # Tiles to image
    tiles = tiles.view(-1, depth * tile_size * tile_size)
    tiles = tiles.t().unsqueeze(0)
    image = F.fold(tiles, output_size=(height, width), kernel_size=tile_size, stride=stride)
    # Normalisation
    norm  = torch.ones_like(image)
    norm  = norm.unfold(2, tile_size, stride).unfold(3, tile_size, stride)
    norm  = norm.permute(0, 1, 4, 5, 2, 3).contiguous()
    norm  = norm.view(1, depth * tile_size * tile_size, -1)
    norm  = F.fold(norm, output_size=(height, width), kernel_size=tile_size, stride=stride)
    image = (image / norm).squeeze(0)
    # Removes padding
    image = image[:, :-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]
    return image

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
    
def display(image:torch.Tensor, title:str='', cmap:str='gray', channel_first:bool=True) -> None:
    '''Displays an image'''
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if channel_first:
        image = image.permute(1, 2, 0)
    fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

def display_sequence(images:torch.Tensor, titles:list=None, grid_size:tuple=None, channel_first:bool=True) -> None:
    '''Displays a grid of images'''
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    if channel_first:
        images = images.permute(0, 2, 3, 1)
    if grid_size is None: grid_size = (1, images.size(0))
    if titles is None: titles = [None] * images.size(0)
    if isinstance(titles, torch.Tensor): titles = titles.tolist()
    fig, axs = pyplot.subplots(nrows=grid_size[0], ncols=grid_size[1], figsize=(3*grid_size[1], 3*grid_size[0]))
    for ax, tile, title in zip(axs.ravel(), images, titles):
        ax.imshow(tile)
        ax.set_title(title)
    for ax in axs.ravel():
        ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

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

#%% MODEL TRAINING UTILITIES

def count_parameters(model:nn.Module) -> None:
    '''Counts the number of parameters in a model'''
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrain  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Trainable parameters: {trainable:,} | Non-trainable parameters: {nontrain:,}')

def set_trainable(module:nn.Module, trainable:bool|list[bool]) -> None:
    '''Sets the trainable status of a model'''
    if isinstance(trainable, bool):
        trainable = [trainable] * len(list(module.parameters()))
    for param, status in zip(module.parameters(), trainable):
        param.requires_grad = status

def empty_cache(device:torch.device) -> None:
    '''Empties the cache of a device'''
    if device == 'cuda':
        torch.cuda.empty_cache()
    if device == 'mps':
        torch.mps.empty_cache()

def optimise(model:nn.Module, train_loader, device:torch.device, criterion, optimiser, accumulate:int=1) -> torch.Tensor:
    '''Optimises a model using a training sample for one epoch'''
    model.train()
    accuracy = metrics.BinaryAccuracy(device='cpu')
    auroc    = metrics.BinaryAUROC(device='cpu')
    run_loss, n_obs = 0.0, 0
    for i, (X, Y) in enumerate(train_loader):
        # Optimisation
        optimiser.zero_grad()
        X, Y = X.to(device), Y.to(device)
        Yh   = model(X)
        loss = criterion(Yh, Y)
        loss.backward()
        # Gradient accumulation
        if ((i + 1) % accumulate == 0) or (i + 1 == len(train_loader)):
            optimiser.step()
        else:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.add_(param.grad.data)
        # Statistics
        subset = ~torch.isnan(Y)
        n_obs += subset.sum()
        run_loss += (loss * subset.sum()).item()
        accuracy.update(Y[subset], Yh[subset] > .5)
        auroc.update(Y[subset], Yh[subset])
        # Print statistics
        print(f"{'Training': <10} | Batch {i+1:03d}/{len(train_loader):03d} | Loss {(run_loss / n_obs):.4f} | Accuracy {accuracy.compute().item():.4f} | Auroc {auroc.compute().item():.4f}", end='\r' if i+1 < len(train_loader) else '\n')
    return run_loss / n_obs

def validate(model:nn.Module, loader, device:torch.device, criterion, threshold:float=0.5) -> torch.Tensor:
    '''Validates a model using a validation sample for one epoch'''
    model.eval()
    accuracy = metrics.BinaryAccuracy(device='cpu')
    auroc    = metrics.BinaryAUROC(device='cpu')
    run_loss, n_obs = 0.0, 0
    with torch.no_grad():                       
        for i, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            Yh   = model(X)
            loss = criterion(Yh, Y)
            # Statistics
            subset = ~torch.isnan(Y)
            n_obs += subset.sum()
            run_loss += (loss * subset.sum()).item()
            accuracy.update(Y[subset], Yh[subset] > .5)
            auroc.update(Y[subset], Yh[subset])
            # Print statistics
            print(f"{'Validation': <10} | Batch {i+1:03d}/{len(loader):03d} | Loss {(run_loss / n_obs):.4f} | Accuracy {accuracy.compute().item():.4f} | Auroc {auroc.compute().item():.4f}", end='\r' if i+1 < len(loader) else '\n')
        return run_loss / n_obs

def predict(model:nn.Module, loader, device:torch.device, n_batches:int=None) -> tuple:
    '''Predicts the labels of a sample'''
    model.eval()
    if n_batches is None:
        n_batches = len(loader)
    Ys, Yhs = [None]*n_batches, [None]*n_batches
    with torch.no_grad():
        for i in range(n_batches):
            X, Y   = next(iter(loader))
            Ys[i]  = Y
            Yhs[i] = model(X.to(device))
            print(f'Batch {i+1:03d}/{n_batches:03d}', end='\r')
    Ys  = torch.cat(Ys)
    Yhs = torch.cat(Yhs)
    return Ys, Yhs

#%%
