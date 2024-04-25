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
    tile_size=128, 
    grid_size=20, 
    batch_size=16, 
    mapping={0:0, 1:1, 2:1, 3:1, 255:torch.tensor(float('nan'))}) #! Choose

#%% INITIALISES DATA LOADERS

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
    
def init_loader(images_zarr:str, labels_zarr:str, mapping:dict):
    train_loader = ZarrDataset(
    images=f'{paths.data}/aleppo/zarr/images_train.zarr',
    labels=f'{paths.data}/aleppo/zarr/labels_train.zarr',
    mapping=params.mapping)

train_loader = ZarrDataset(
    images=f'{paths.data}/aleppo/zarr/images_train.zarr',
    labels=f'{paths.data}/aleppo/zarr/labels_train.zarr',
    mapping=params.mapping)

valid_loader = ZarrDataset(
    images=f'{paths.data}/aleppo/zarr/images_valid.zarr',
    labels=f'{paths.data}/aleppo/zarr/labels_valid.zarr',
    mapping=params.mapping)

test_loader = ZarrDataset(
    images=f'{paths.data}/aleppo/zarr/images_test.zarr',
    labels=f'{paths.data}/aleppo/zarr/labels_test.zarr',
    mapping=params.mapping)

train_loader = utils.data.DataLoader(train_loader, batch_size=params.batch_size, shuffle=False)
valid_loader = utils.data.DataLoader(valid_loader, batch_size=params.batch_size, shuffle=False)
test_loader  = utils.data.DataLoader(test_loader,  batch_size=params.batch_size, shuffle=False)

''' Checks data loaders
X, Y = next(iter(train_loader))
display_sequence(X[0], Y[0], grid_size=(5,5))
del X, Y
''' 

#%% INTIALISES MODELS

''' #! Downloads pre-trained model
import satlaspretrain_models
image_encoder = satlaspretrain_models.Weights()
image_encoder = image_encoder.get_pretrained_model(model_identifier='Aerial_SwinB_SI', fpn=True, device='cpu')
torch.save(image_encoder, path.join(paths.models, 'Aerial_SwinB_SI.pth'))
del image_encoder
'''

# Initialises model components
image_encoder    = torch.load(path.join(paths.models, 'Aerial_SwinB_SI.pth'))
image_encoder    = ImageEncoder(image_encoder)
sequence_encoder = dict(input_dim=512, max_length=25, n_heads=4, hidden_dim=512, n_layers=4, dropout=0.0)
sequence_encoder = SequenceEncoder(**sequence_encoder)
prediction_head  = PredictionHead(input_dim=512, output_dim=1)

# Initialises model wrapper
model = ModelWrapper(image_encoder, sequence_encoder, prediction_head, n_features=512)
model.to(device)

# Checks model parameters
count_parameters(model)
count_parameters(model.image_encoder)
count_parameters(model.sequence_encoder)

del image_encoder, sequence_encoder, prediction_head

#%% OPTIMISATION

#? Loads previous checkpoint
model = torch.load(f'{paths.models}/ModelWrapper_best.pth')

#? Parameter alignment i.e. freezes image encoder's parameters
model.image_encoder = set_trainable(model.image_encoder, False)
optimiser = optim.AdamW(model.parameters(), lr=1e-4)
count_parameters(model)

''' #? Fine tuning i.e. unfreezes image encoder's parameters
model.image_encoder = set_trainable(model.image_encoder, True)
optimiser = optim.AdamW(model.parameters(), lr=1e-5)
count_parameters(model)
'''

def train(model:nn.module, train_loader, valid_loader, device:torch.device, criterion, optimiser, n_epochs:int=1, patience:int=1, accumulate:int=1, path:str=paths.models):
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
      n_epochs=10, 
      patience=3,
      accumulate=1,
      path=paths.models)

# Restores best model
model = torch.load(f'{paths.models}/ModelWrapper_best.pth')

# Testing
validate(model=model, loader=test_loader, device=device, criterion=criterion)
empty_cache(device)

#%% ESTIMATES THRESHOLD

# Threshold estimation
Y, Yh  = predict(model, loader=train_loader, device=device)
subset = ~Y.isnan()
precision, recall, threshold = metrics.precision_recall_curve(Y[subset].cpu(), Yh[subset].cpu())
fscore    = (2 * precision * recall) / (precision + recall)
threshold = threshold[np.argmax(fscore)]
del Y, Yh, subset, precision, recall, fscore

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
