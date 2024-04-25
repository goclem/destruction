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
from torch import nn


# Utilities
params = argparse.Namespace(batch_size=16, label_map={0:0, 1:1, 2:1, 3:1, 255:torch.tensor(float('nan'))})



#%% MULTI CITY TRAINING


#%% TEST

datasets = [
    ZarrDataset(images_zarr=f'{paths.data}/aleppo/zarr/images_train.zarr', labels_zarr=f'{paths.data}/aleppo/zarr/labels_train.zarr'),
    ZarrDataset(images_zarr=f'{paths.data}/rakka/zarr/images_test.zarr',  labels_zarr=f'{paths.data}/rakka/zarr/labels_test.zarr'),
    ZarrDataset(images_zarr=f'{paths.data}/rakka/zarr/images_valid.zarr',  labels_zarr=f'{paths.data}/rakka/zarr/labels_valid.zarr')
    ]

train_loader = ZarrDataLoader(datasets, batch_size=32, label_map=params.label_map)
X, Y = next(iter(train_loader))
display_sequence(X[0], titles=Y[0], grid_size=(5,5))
#%% DEPRECIATED

#self.indices = [np.split(np.arange(size), np.arange(step, size, step)) for step, size in zip(steps, sizes)]