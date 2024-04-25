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
params = argparse.Namespace(batch_size=16)

#%% MULTI CITY TRAINING

class ZarrDataset(utils.data.Dataset):
    '''Zarr dataset for PyTorch'''
    def __init__(self, images_zarr:str, labels_zarr:str, mapping:dict=None):
        self.images  = zarr.open(images_zarr, mode='r')
        self.labels  = zarr.open(labels_zarr, mode='r')
        self.mapping = mapping
        self.length  = len(self.images)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Read data from the Zarr dataset at the specified index
        X = torch.from_numpy(self.images[idx]).float()
        Y = torch.from_numpy(self.labels[idx]).float()
        X = torch.div(X, 255)
        if self.mapping is not None:
            Y.apply_(lambda y: self.mapping.get(y, y))
        return X, Y

class ZarrDataLoader:
    def __init__(self, datasets, batch_size:int=16, pad_value:int=255):
        self.datasets    = datasets
        self.batch_size  = batch_size
        self.pad_value   = pad_value
        self.slice_sizes = None
        self.n_batches   = None
        self.compute_slice_sizes()

    def compute_slice_sizes(self):
        dataset_sizes    = np.array([len(dataset) for dataset in self.datasets])
        self.slice_sizes = (np.divide(dataset_sizes, np.sum(dataset_sizes)) * self.batch_size).round().astype(int)
        self.n_batches   = np.min(np.floor(np.divide(dataset_sizes, self.slice_sizes))).astype(int)

    def __iter__(self):
        data_loaders = [utils.data.DataLoader(dataset, batch_size=int(slice_size)) for dataset, slice_size in zip(self.datasets, self.slice_sizes)]
        for batch in zip(*data_loaders):
            n_times = max([data[0].size(1) for data in batch])
            images  = torch.cat([nn.functional.pad(data[0], pad=(0, 0, 0, 0, 0, 0, 0, n_times - data[0].size(1), 0, 0), value=self.pad_value) for data in batch])
            labels  = torch.cat([nn.functional.pad(data[1], pad=(0, 0, 0, n_times - data[1].size(1), 0, 0), value=self.pad_value) for data in batch])
            indices = torch.randperm(images.size(0))
            yield images[indices], labels[indices]

    def __len__(self):
        return self.n_batches

#%% TEST

datasets = [
    ZarrDataset(images_zarr=f'{paths.data}/aleppo/zarr/images_train.zarr', labels_zarr=f'{paths.data}/aleppo/zarr/labels_train.zarr', mapping=params.mapping),
    ZarrDataset(images_zarr=f'{paths.data}/rakka/zarr/images_test.zarr',  labels_zarr=f'{paths.data}/rakka/zarr/labels_test.zarr', mapping=params.mapping),
    ZarrDataset(images_zarr=f'{paths.data}/rakka/zarr/images_valid.zarr',  labels_zarr=f'{paths.data}/rakka/zarr/labels_valid.zarr', mapping=params.mapping)
    ]

train_loader = ZarrDataLoader(datasets, batch_size=32)
X, Y = next(iter(train_loader))

#%% DEPRECIATED

#self.indices = [np.split(np.arange(size), np.arange(step, size, step)) for step, size in zip(steps, sizes)]