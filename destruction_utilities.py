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
import torch

from matplotlib import pyplot
from rasterio import enums, features, windows
from torch import nn

#%% PATHS UTILITIES

home  = os.path.expanduser('~')
paths = argparse.Namespace(
    data='../data',
    models='../models',
    figures='../figures',
    desktop=os.path.join(home, 'Desktop'),
    temporary=os.path.join(home, 'Desktop', 'temporary')
)
del home

#%% FILE UTILITIES

def search_data(pattern:str='.*', directory:str=paths.data) -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    regex = fr'^.*{city}/.*/{type}_{date}\.{ext}$'
    return regex

def extract(files:list, pattern:str=r'\d{4}-\d{2}-\d{2}') -> list:
    regex = re.compile(pattern)
    match = np.array([regex.search(file).group() for file in files])
    return match

def reset_folder(path:str, remove:bool=False) -> None:
    '''Resets a folder'''
    if remove:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(path)

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

def center_window(source:str, size:dict):
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
    depth, height, width = image.shape
    if stride is None: 
        stride = tile_size
    pad_h = math.ceil((height - tile_size) / stride) * stride + tile_size - height
    pad_w = math.ceil((width  - tile_size) / stride) * stride + tile_size - width
    image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=255)
    tiles = image.unfold(1, tile_size, stride).unfold(2, tile_size, stride)
    tiles = tiles.permute(1, 2, 0, 3, 4)
    tiles = tiles.contiguous().view(-1, depth, tile_size, tile_size)
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

def sample_split(images:np.ndarray, samples:np.ndarray) -> list:
    '''Splits the data structure into multiple samples'''
    samples = [images[samples == value, ...] for value in np.unique(samples)]
    return samples

def to_numpy(tensor:torch.Tensor):
    '''Converts a tensor to a numpy array'''
    return tensor.detach().cpu().numpy()

#%% DISPLAY UTILITIES
    
def display(image:torch.Tensor, title:str='', cmap:str='gray') -> None:
    '''Displays an image'''
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0)
    fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

def display_sequence(images:torch.Tensor, titles:list=None, grid_size:tuple=None) -> None:
    '''Displays a grid of images'''
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

#%% TRAINING UTILITIES
    
def count_parameters(model:nn.Module):
    '''Counts the number of parameters in a model'''
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrain  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Trainable parameters: {trainable:,} | Non-trainable parameters: {nontrain:,}')

def set_trainable(model:nn.Module, trainable:bool):
    '''Sets the trainable status of a model'''
    for param in model.parameters():
        param.requires_grad = trainable
    if trainable:
        model.train()
    else:
        model.eval()
    return model

def empty_cache(device:torch.device):
    '''Empties the cache of a device'''
    if device == 'cuda':
        torch.cuda.empty_cache()
    if device == 'mps':
        torch.mps.empty_cache()

class BceLoss(nn.Module):
    def __init__(self, focal:bool=True, drop_nan:bool=True, alpha:float=0.25, gamma:int=2):
        super().__init__()
        self.focal = focal
        self.drop_nan = drop_nan
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        subset = torch.ones(targets.size(), dtype=torch.bool)
        if self.drop_nan:
            subset = ~torch.isnan(targets)
        loss = nn.functional.binary_cross_entropy(inputs[subset], targets[subset], reduction='none')
        if self.focal:
            loss = self.alpha * (1 - torch.exp(-loss))**self.gamma * loss
        loss = torch.mean(loss)
        return loss

def print_statistics(i_batch:int, n_batches:int, running_loss:torch.Tensor, n_obs:int, n_correct:int, label:str, print_format:str=f'02') -> None:
    '''Prints the current statistics of a batch'''
    end_print = '\r' if i_batch+1 < n_batches else '\n'
    print(f'{label: <10} | Batch {i_batch+1:{print_format}d}/{n_batches:{print_format}d} | Loss {running_loss / n_obs:.4f} | Accuracy {n_correct / n_obs:.4f}', end=end_print)

def print_epoch(i_epoch:int, n_epochs:int) -> None:
    '''Prints the current epoch'''
    print_format = f'0{len(str(n_epochs))}'
    print(f'Epoch {i_epoch+1:{print_format}d}/{n_epochs}')

def optimise(model:nn.Module, loader, device:torch.device, criterion, optimiser, accumulate:int=1):
    '''Optimises a model using a training sample for one epoch'''
    model.train()
    n_correct, n_obs, running_loss = 0, 0, 0.0
    for i, (X, Y) in enumerate(loader):
        # Optimisation
        optimiser.zero_grad()
        X, Y = X.to(device), Y.to(device)
        Yh = model(X)
        loss = criterion(Yh, Y)
        loss.backward()
        # Gradient accumulation
        if ((i + 1) % accumulate == 0) or (i + 1 == len(loader)):
            optimiser.step()
        else:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.add_(param.grad.data)
        # Statistics
        subset = ~torch.isnan(Y)
        n_obs += torch.sum(subset)
        n_correct += (Yh[subset].round() == Y[subset]).sum().item()
        running_loss += (loss * torch.sum(subset)).item()
        print_statistics(i_batch=i, n_batches=len(loader), running_loss=running_loss, n_obs=n_obs, n_correct=n_correct, label='Training')
    return running_loss / n_obs

def validate(model, loader, device:torch.device, criterion, threshold:float=0.5):
    '''Validates a model using a validation sample for one epoch'''
    model.eval()
    n_correct, n_obs, running_loss = 0, 0, 0.0
    with torch.no_grad():                       
        for i, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            Yh   = model(X)
            loss = criterion(Yh, Y)
            # Statistics
            subset = ~torch.isnan(Y)
            n_obs += torch.sum(subset)
            n_correct += ((Yh[subset] > threshold).float() == Y[subset]).sum().item()
            running_loss += (loss * torch.sum(subset)).item()
            print_statistics(i_batch=i, n_batches=len(loader), running_loss=running_loss, n_obs=n_obs, n_correct=n_correct, label='Validation')
        return running_loss / n_obs

def train(model, train_loader, valid_loader, device:torch.device, criterion, optimiser, n_epochs:int=1, patience:int=1, accumulate:int=1, path=paths.models):
    '''Trains a model using a training and validation sample'''
    best_loss, counter = torch.tensor(float('inf')), 0
    for epoch in range(n_epochs):
        print_epoch(i_epoch=epoch, n_epochs=n_epochs)
        train_loss = optimise(model=model, loader=train_loader, device=device, criterion=criterion, optimiser=optimiser, accumulate=accumulate)
        # Early stopping
        valid_loss = validate(model=model, loader=valid_loader, device=device, criterion=criterion)
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model, f'{path}/{model.__class__.__name__}_best.pth')
        else:
            counter += 1
            if counter >= patience:
                print('- Early stopping')
                return model
                break

def predict(model, loader, device:torch.device):
    '''Predicts the labels of a sample'''
    print_format = len(str(len(loader)))
    model.eval()
    Ys, Yhs = [None]*len(loader), [None]*len(loader)
    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            Ys[i]  = Y
            Yhs[i] = model(X.to(device))
            print(f'Batch {i+1:{print_format}d}/{len(loader):{print_format}d}', end='\r')
    Ys  = torch.cat(Ys)
    Yhs = torch.cat(Yhs)
    return Ys, Yhs
# %%
