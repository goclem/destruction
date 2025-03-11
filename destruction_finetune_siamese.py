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
import pytorch_lightning as pl
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

    def __init__(self, datafiles:list, datasets:list, label_map:dict, batch_size:int, shuffle:bool=True):
        self.datafiles    = datafiles
        self.datasets     = datasets
        self.label_map    = label_map
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.batch_index  = 0
        self.data_sizes   = np.array([len(dataset) for dataset in datasets])
        self.data_indices = self.compute_data_indices()

    def compute_data_indices(self):
        slice_sizes  = np.cbrt(self.data_sizes)
        slice_sizes  = np.divide(slice_sizes, slice_sizes.sum())
        slice_sizes  = np.random.multinomial(self.batch_size, slice_sizes, size=int(np.max(self.data_sizes / self.batch_size)))
        data_indices = np.vstack((np.zeros(len(self.data_sizes), dtype=int), np.cumsum(slice_sizes, axis=0)))
        data_indices = data_indices[np.all(data_indices < self.data_sizes, axis=1)]
        return data_indices

    def __len__(self):
        return len(self.data_indices) - 1
    
    def __iter__(self):
        self.batch_index = 0
        if self.shuffle:
            for city in self.datafiles:
                print(f'Shuffling {city}', end='\r')
                shuffle_zarr(
                    images_zarr=self.datafiles[city]['images_zarr'], 
                    labels_zarr=self.datafiles[city]['labels_zarr'])
        return self

    def __next__(self):
        if self.batch_index == len(self):
            raise StopIteration 
        # Loads tiles
        X, Y = list(), list()
        for dataset, indices in zip(self.datasets, self.data_indices.T):
            start = indices[self.batch_index]
            end   = indices[self.batch_index + 1]
            if start != end: # Skips empty batches
                X_ds, Y_ds = dataset[start:end]
                X.append(X_ds), 
                Y.append(Y_ds)
        X, Y = torch.cat(X), torch.cat(Y)
        # Remaps labels
        for key, value in self.label_map.items():
            Y = torch.where(Y == key, value, Y)
        # Updates batch index
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

def unprocess_images(images:torch.Tensor, preprocessor:nn.Module) -> torch.Tensor:
    means  = torch.Tensor(preprocessor.image_mean).view(3, 1, 1)
    stds   = torch.Tensor(preprocessor.image_std).view(3, 1, 1)
    images = (images * stds + means) * 255
    images = torch.clip(images, 0, 255).to(torch.uint8)
    return images

#%% INITIALISES DATA LOADERS

# Initialises datasets
train_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_train_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_train_balanced.zarr') for city in params.cities]))
valid_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_valid_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_valid_balanced.zarr') for city in params.cities]))
test_datafiles  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_test_balanced.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_test_balanced.zarr')  for city in params.cities]))
train_datasets  = [ZarrDataset(**train_datafiles[city]) for city in params.cities]
valid_datasets  = [ZarrDataset(**valid_datafiles[city]) for city in params.cities]
test_datasets   = [ZarrDataset(**test_datafiles[city])  for city in params.cities]

# Intialises data loaders
params = dict(
    batch_size=params.batch_size, 
    label_map=params.label_map,
    shuffle=True)

train_loader = ZarrDataLoader(datafiles=train_datafiles, datasets=train_datasets, **params)
valid_loader = ZarrDataLoader(datafiles=valid_datafiles, datasets=valid_datasets, **params)
test_loader  = ZarrDataLoader(datafiles=test_datafiles,  datasets=test_datasets,  **params)
del train_datafiles, valid_datafiles, test_datafiles, train_datasets, valid_datasets, test_datasets, params

''' Checks data loaders
X, Y = next(train_loader)
for idx in np.random.choice(range(len(X)), size=5, replace=False):
    display_sequence(X[idx], [0] + [int(Y[idx])])
    print(np.equal(X[idx][0], X[idx][1]).all())
del X, Y, idx
'''

#! Something odd is that images are duplicated... NEEDS CHECKING

#%% INITIALISES MODEL

class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin 

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

class SiameseModel(nn.Module):
    
    def __init__(self, preprocessor:nn.Module, image_encoder:nn.Module):
        super().__init__()
        self.preprocessor     = preprocessor
        self.image_encoder    = image_encoder
        self.projection_block = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 128))
        self.output_layer = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward_once(self, X:torch.Tensor) -> torch.Tensor:
        H = self.preprocessor(X, return_tensors='pt', do_resize=True).to(device)
        H = self.image_encoder(**H.to(device))
        H = self.prediction_head(H.last_hidden_state[:, 0, :])
        H = self.projection_block(H)
        return H

    def forward(self, X0:torch.Tensor, X1:torch.Tensor) -> torch.Tensor:
        H0 = self.forward_once(X0)
        H1 = self.forward_once(X1)
        distance = F.pairwise_distance(H0, H1, keepdim=True)
        return distance

# Initialises model components
preprocessor    = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base', device=device)
image_config    = transformers.ViTMAEConfig.from_pretrained('../models/checkpoint-9920')
image_encoder   = transformers.ViTMAEModel.from_pretrained('../models/checkpoint-9920', config=image_config)
status_list     = [param.requires_grad for param in image_encoder.parameters()] # Records the original trainable status of the image encoder's parameters
prediction_head = PredictionHead(input_dim=768, output_dim=1)
count_parameters(image_encoder)

# Initialises model
model = ModelWrapper(preprocessor, image_encoder, prediction_head)
model = model.to(device)
count_parameters(model=model)

del preprocessor, image_encoder, prediction_head

#%% OPTIMISATION 1: OPTIMISE REGRESSION HEAD

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

# Freezes image encoder's parameters
set_trainable(module=model.image_encoder, trainable=False)
set_trainable(module=model.prediction_head, trainable=True)
count_parameters(model=model)

# Initialises optimiser and criterion
criterion = BceLoss(focal=True, drop_nan=True, alpha=0.25, gamma=2.0)
optimiser = optim.AdamW(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999))

# Training
train(model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      device=device,
      criterion=criterion,
      optimiser=optimiser,
      model_path='../models/vitmae_prepost_aligned.pth',
      n_epochs=100,
      patience=3,
      accumulate=4)

# Clears GPU memory
empty_cache(device=device)

#%% OPTIMISATION 2: FINES TUNES THE ENTIRE MODEL

# Unfreezes image encoder's parameters
set_trainable(module=model.image_encoder, trainable=status_list)
set_trainable(module=model.prediction_head, trainable=True)
count_parameters(model=model)

# Initialises optimiser and criterion
criterion = BceLoss(focal=True, drop_nan=True, alpha=0.25, gamma=2.0)
optimiser = optim.AdamW(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999))

# Training
train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device,
      criterion=criterion, 
      optimiser=optimiser, 
      model_path='../models/vitmae_prepost_finetuned.pth',
      n_epochs=100, 
      patience=3,
      accumulate=4)

# Clears GPU memory
empty_cache(device=device)

#%%
