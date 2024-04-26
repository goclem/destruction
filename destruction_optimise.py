#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Model optimisation for the destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import numpy as np
import torch
import typing

from destruction_models import *
from destruction_utilities import *
from torch import optim, nn, utils
from os import path
from sklearn import metrics

# Utilities
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
params = argparse.Namespace(
    cities=['aleppo'],
    tile_size=128, 
    batch_size=32, 
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})

#%% INITIALISES DATA LOADERS

# Datasets
train_datasets = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_train.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_train.zarr') for city in params.cities]))
valid_datasets = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_valid.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_valid.zarr') for city in params.cities]))
test_datasets  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_test.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_test.zarr')  for city in params.cities]))

# Intialises data loaders
train_loader = [ZarrDataset(**train_datasets[city]) for city in params.cities]
valid_loader = [ZarrDataset(**valid_datasets[city]) for city in params.cities]
test_loader  = [ZarrDataset(**test_datasets[city])  for city in params.cities]
train_loader = ZarrDataLoader(train_loader, batch_size=params.batch_size, label_map=params.label_map)
valid_loader = ZarrDataLoader(valid_loader, batch_size=params.batch_size, label_map=params.label_map)
test_loader  = ZarrDataLoader(test_loader,  batch_size=params.batch_size, label_map=params.label_map)

''' Checks data loaders
X, Y = next(iter(train_loader))
display_sequence(X[0], Y[0], grid_size=(5,5))
del X, Y
''' 

#%% INTIALISES MODELS

# Initialises model components
image_encoder    = ImageEncoder(feature_extractor=torch.load(f'{paths.models}/Aerial_SwinB_SI.pth'))
sequence_encoder = dict(input_dim=512, max_length=25, n_heads=4, hidden_dim=512, n_layers=4, dropout=0.0)
sequence_encoder = SequenceEncoder(**sequence_encoder)
prediction_head  = PredictionHead(input_dim=512, output_dim=1)

# Initialises model wrapper
model = ModelWrapper(image_encoder, sequence_encoder, prediction_head)
model.to(device)

# Checks model parameters
count_parameters(model)
count_parameters(model.image_encoder)
count_parameters(model.sequence_encoder)

del image_encoder, sequence_encoder, prediction_head

#%% OPTIMISATION

''' #? Loads previous checkpoint
model = torch.load(f'{paths.models}/ModelWrapper_full123.pth')
'''

#? Parameter alignment i.e. freezes image encoder's parameters
set_trainable(model.image_encoder.feature_extractor, False)
count_parameters(model)
optimiser = optim.AdamW(model.parameters(), lr=1e-4)

''' #? Fine tuning i.e. unfreezes image encoder's parameters
set_trainable(model.image_encoder.feature_extractor, True)
optimiser = optim.AdamW(model.parameters(), lr=1e-5)
count_parameters(model)
'''

def train(model:nn.Module, train_loader, valid_loader, device:torch.device, criterion, optimiser, n_epochs:int=1, patience:int=1, accumulate:int=1, path:str=paths.models):
    '''Trains a model using a training and validation sample'''
    best_loss, counter = torch.tensor(float('inf')), 0
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1:03d}/{n_epochs:03d}')
        train_loss = optimise(model=model, loader=train_loader, device=device, criterion=criterion, optimiser=optimiser, accumulate=accumulate)
        # Early stopping
        valid_loss = validate(model=model, loader=valid_loader, device=device, criterion=criterion)
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model, f'{path}/{model.__class__.__name__}_best.pth')
            # Shuffles zarr datasets
            for city in params.cities:
                shuffle_zarr(**train_datasets[city])
                shuffle_zarr(**valid_datasets[city])
        else:
            counter += 1
            if counter >= patience:
                print('- Early stopping')
                return model
                break

# Loss function
criterion = BceLoss(focal=True, drop_nan=True, alpha=0.25, gamma=2.0)

# Training
train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device, 
      criterion=criterion, 
      optimiser=optimiser, 
      n_epochs=25, 
      patience=3,
      accumulate=1,
      path=paths.models)

empty_cache(device)

#%% ESTIMATES THRESHOLD

def compute_threshold(model:nn.Module, loader, device:torch.device, n_batches:int=None) -> float:
    '''Estimates threshold for binary classification'''
    Y, Yh  = predict(model, loader=train_loader, device=device, n_batches=n_batches)
    subset = ~Y.isnan()
    fpr, tpr, thresholds = metrics.roc_curve(Y[subset].cpu(), Yh[subset].cpu())
    threshold = thresholds[np.argmax(tpr - fpr)]
    return threshold

# Threshold estimation  
threshold = compute_threshold(model=model, loader=train_loader, device=device)
print(f'Threshold: {threshold:.2f}')

# Testing
validate(model=model, loader=test_loader, device=device, criterion=criterion, threshold=threshold)

#%% CHECKS PREDICTIONS

with torch.no_grad():
    X, Y = next(iter(test_loader))
    Yh = model(X.to(device)).cpu()

Y, Yh  = Y.squeeze().numpy(), Yh.squeeze_().numpy()
status = np.where(np.isnan(Y), 'Nan', np.equal(Y, Yh > threshold))

for i in np.random.choice(range(len(Y)), 2, replace=False):
    titles = [f'{s}\nY: {y:.0f} - Yh: {yh > threshold:.0f} ({yh:.2f})'for s, y, yh in zip(status[i], Y[i], Yh[i])]
    display_sequence(X[i], titles, grid_size=(5,5))
del titles

#%%
