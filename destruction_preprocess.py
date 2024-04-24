#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Preprocessing for the destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import zarr

from destruction_utilities import *

# Utilities
params = argparse.Namespace(city='aleppo', tile_size=128)

#%% DATA PREPROCESSING

def load_sequences(files:list, tile_size:int, window:int=None, stride:int=None) -> torch.Tensor:
    if window is not None:
        window = center_window(source=files[0], size=(window*tile_size, window*tile_size))
    sequences = [read_raster(file, window=window, dtype='uint8') for file in files]
    sequences = [torch.tensor(image).permute(2, 0, 1) for image in sequences]
    sequences = [image_to_tiles(image, tile_size=tile_size, stride=stride) for image in sequences]
    sequences = torch.stack(sequences).swapaxes(1, 0)
    return sequences

# Reads images
images = search_data(pattern(city=params.city, type='image'))[:5]
images = load_sequences(images, tile_size=params.tile_size)
images = images / 255

# Reads labels
labels = search_data(pattern(city=params.city, type='label'))
labels = load_sequences(labels, grid_size=params.grid_size, tile_size=1)
labels = labels.squeeze(2, 3) 

# Remaps label values
# labels = labels.apply_(lambda val: params.mapping.get(str(val))).type(torch.float)
# labels = torch.where(labels == 255, torch.tensor(float('nan')), labels)

# Reads samples
samples = search_data('aleppo_samples.tif$')
samples = load_sequences(samples, grid_size=params.grid_size, tile_size=1)
samples = samples.squeeze()