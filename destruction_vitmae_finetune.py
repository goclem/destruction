#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Fine-tunes the vision transformer on destruction tiles
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import accelerate
import argparse
import matplotlib.pyplot as plt
import numpy as np
import transformers
import torch

from torch import optim
from destruction_models import *
from destruction_utilities import *

# Utilities
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
params = argparse.Namespace(
    cities=['aleppo', 'moschun'],
    batch_size=64,
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})

#%% TRAINING UTILITIES

class ZarrDataset(utils.data.Dataset):

    def __init__(self, images_zarr:str, labels_zarr:str):
        self.images = zarr.open(images_zarr, mode='r')
        self.labels = zarr.open(labels_zarr, mode='r')
        self.length = len(self.images)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.images[idx])
        Y = torch.from_numpy(self.labels[idx])
        return X, Y

class ZarrDataLoader:

    def __init__(self, datafiles:list, datasets:list, label_map:dict, batch_size:int, preprocessor=None):
        self.datafiles    = datafiles
        self.datasets     = datasets
        self.label_map    = label_map
        self.batch_size   = batch_size
        self.batch_index  = 0
        self.data_sizes   = np.array([len(dataset) for dataset in datasets])
        self.data_indices = self.compute_data_indices()
        self.preprocessor = preprocessor

    def compute_data_indices(self):
        slice_sizes  = np.cbrt(self.data_sizes) #! Large impact
        slice_sizes  = np.divide(slice_sizes, slice_sizes.sum())
        slice_sizes  = np.random.multinomial(self.batch_size, slice_sizes, size=int(np.max(self.data_sizes / self.batch_size)))
        data_indices = np.vstack((np.zeros(len(self.data_sizes), dtype=int), np.cumsum(slice_sizes, axis=0)))
        data_indices = data_indices[np.all(data_indices < self.data_sizes, axis=1)]
        return data_indices

    def __len__(self):
        return len(self.data_indices) - 1
    
    def __iter__(self):
        self.batch_index = 0
        for city in self.datafiles:
            print(f'Shuffling {city}', end='\r')
            shuffle_zarr(self.datafiles[city]['images_zarr'])
            shuffle_zarr(self.datafiles[city]['labels_zarr'])
        return self

    def __next__(self):
        if self.batch_index == len(self):
            raise StopIteration 
        X, Y = list(), list()
        for dataset, indices in zip(self.datasets, self.data_indices.T):
            start = indices[self.batch_index]
            end   = indices[self.batch_index + 1]
            if start != end: # Skips empty batches
                X_ds, Y_ds = dataset[start:end]
                X.append(X_ds), 
                Y.append(Y_ds)
        X, Y = torch.cat(X), torch.cat(Y)
        if self.preprocessor is not None:
            X = self.preprocessor(X.moveaxis(1, -1), return_tensors='pt', do_resize=True)
        for key, value in self.label_map.items():
            Y = torch.where(Y == key, value, Y)
        self.batch_index += 1
        return X, Y
    
class BceLoss(nn.Module):
    '''Binary cross-entropy loss with optional focal loss'''
    def __init__(self, focal:bool=True, drop_nan:bool=True, alpha:float=0.25, gamma:float=2.0):
        super().__init__()
        self.focal    = focal
        self.drop_nan = drop_nan
        self.alpha    = alpha
        self.gamma    = gamma

    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        subset = torch.ones(targets.size(), dtype=torch.bool)
        if self.drop_nan:
            subset = ~torch.isnan(targets)
        loss = nn.functional.binary_cross_entropy(inputs[subset], targets[subset], reduction='none')
        if self.focal:
            loss = self.alpha * (1 - torch.exp(-loss))**self.gamma * loss
        loss = torch.mean(loss)
        return loss

#%% INITIALISES DATA LOADERS

# Initialises datasets
train_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_train_vitmae.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_train_vitmae.zarr') for city in params.cities]))
valid_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_valid_vitmae.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_valid_vitmae.zarr') for city in params.cities]))
test_datafiles  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_test_vitmae.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_test_vitmae.zarr')  for city in params.cities]))
train_datasets  = [ZarrDataset(**train_datafiles[city]) for city in params.cities]
valid_datasets  = [ZarrDataset(**valid_datafiles[city]) for city in params.cities]
test_datasets   = [ZarrDataset(**test_datafiles[city])  for city in params.cities]

# Intialises data loaders
preprocessor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
train_loader = ZarrDataLoader(datafiles=train_datafiles, datasets=train_datasets, batch_size=params.batch_size, label_map=params.label_map, preprocessor=preprocessor)
valid_loader = ZarrDataLoader(datafiles=valid_datafiles, datasets=valid_datasets, batch_size=params.batch_size, label_map=params.label_map, preprocessor=preprocessor)
test_loader  = ZarrDataLoader(datafiles=test_datafiles,  datasets=test_datasets,  batch_size=params.batch_size, label_map=params.label_map, preprocessor=preprocessor)

del train_datafiles, valid_datafiles, test_datafiles, train_datasets, valid_datasets, test_datasets

''' Checks data loaders
X, Y = next(train_loader)
idx  = np.random.choice(range(len(X)), size=25, replace=False)
display_sequence(X[idx], Y[idx], grid_size=(5, 5))
del idx, X, Y
''' 

#%% INITIALISES MODEL

class ModelWrapper(nn.Module):
    def __init__(self, image_encoder:nn.Module, prediction_head:nn.Module):
        super().__init__()
        self.image_encoder   = image_encoder
        self.prediction_head = prediction_head

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        # Encodes images
        H = self.image_encoder(**X)
        Y = self.prediction_head(H.last_hidden_state[:, 0, :])
        return Y

# Initialises model components
image_config    = transformers.ViTMAEConfig.from_pretrained('../models/checkpoint-9920')
image_encoder   = transformers.ViTMAEModel.from_pretrained('../models/checkpoint-9920', config=image_config)
prediction_head = PredictionHead(input_dim=768, output_dim=1)

# Initialises model
model = ModelWrapper(image_encoder, prediction_head)
model = model.to(device)
count_parameters(model)

del preprocessor, image_encoder, prediction_head

#%% ALIGNS HEAD PARAMETERS

def train(model:nn.Module, train_loader, valid_loader, device:torch.device, criterion, optimiser, model_path:str, n_epochs:int=1, patience:int=1, accumulate:int=1):
    best_loss, counter = torch.tensor(float('inf')), 0
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1:03d}/{n_epochs:03d}')
        train_loss = optimise(model=model, train_loader=train_loader, device=device, criterion=criterion, optimiser=optimiser, accumulate=accumulate)
        # Early stopping
        valid_loss = validate(model=model, loader=valid_loader, device=device, criterion=criterion)
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model, model_path)
        else:
            counter += 1
            if counter >= patience:
                print('- Early stopping')
                break

# Loss and optimiser
criterion = BceLoss(focal=True, drop_nan=True, alpha=0.25, gamma=2.0)
set_trainable(model.image_encoder, False)
optimiser = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

# Training
train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device,
      criterion=criterion, 
      optimiser=optimiser, 
      model_path='../models/vitmae_finetune.pth',
      n_epochs=100, 
      patience=3,
      accumulate=4)

# Clears GPU memory
empty_cache(device)

#%% FINES TUNES MODEL

# Loss and optimiser
criterion = BceLoss(focal=True, drop_nan=True, alpha=0.25, gamma=2.0)
set_trainable(model.image_encoder, True)
optimiser = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

# Training
train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device,
      criterion=criterion, 
      optimiser=optimiser, 
      model_path='../models/vitmae_finetune.pth',
      n_epochs=100, 
      patience=3,
      accumulate=4)

# Clears GPU memory
empty_cache(device)
#%%
