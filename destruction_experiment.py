#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Experiments for destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import itertools
import numpy as np
import zarr

from destruction_utilities import *

# Utilities
params = argparse.Namespace(batch_size=16)

def load_subset(dataset, steps, i):
    return zarr.open(dataset)[steps[i]]

cities = ['aleppo', 'rakka']

images_zarrs = [f'{paths.data}/{city}/zarr/images_train.zarr' for city in cities]
labels_zarrs = [f'{paths.data}/{city}/zarr/images_train.zarr' for city in cities]
sizes = np.array([len(zarr.open(images_zarr)) for images_zarr in images_zarrs])
steps = (sizes / np.sum(sizes) * params.batch_size).round().astype(int)
steps = [np.split(np.arange(size), np.arange(step, size, step=step)) for step, size in zip(steps, sizes)]
n_batches = np.min([len(step) for step in steps])

for i in range(n_batches):
    images_batch = np.concatenate(list(map(load_subset, images_zarrs, steps, itertools.repeat(0))))
    labels_batch = np.concatenate(list(map(load_subset, labels_zarrs, steps, itertools.repeat(0))))


slice(1, 10, 2)


