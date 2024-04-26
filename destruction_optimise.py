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
    grid_size=20, 
    batch_size=32, 
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})

#%% INITIALISES DATA LOADERS

class ZarrDataset(utils.data.Dataset):
    '''Zarr dataset for PyTorch'''
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
    def __init__(self, datasets:list, label_map:dict, batch_size:int):
        self.datasets    = datasets
        self.batch_size  = batch_size
        self.label_map   = label_map
        self.slice_sizes = None
        self.n_batches   = None
        self.compute_slice_sizes()

    def compute_slice_sizes(self):
        dataset_sizes    = torch.tensor([len(dataset) for dataset in self.datasets])
        self.slice_sizes = (dataset_sizes / dataset_sizes.sum() * self.batch_size).round().int()
        self.n_batches   = (dataset_sizes // self.slice_sizes).min()
    
    def pad_sequence(self, sequence:torch.Tensor, value:int, seq_len:int=None, dim:int=1) -> torch.Tensor:
        pad = torch.zeros(2*len(sequence.size()), dtype=int)
        pad = pad.index_fill(0, torch.tensor(2*dim), seq_len-sequence.size(dim)).flip(0).tolist()
        pad = nn.functional.pad(sequence, pad=pad, value=value)
        return pad

    def __iter__(self):
        data_loaders = [utils.data.DataLoader(dataset, batch_size=int(slice_size)) for dataset, slice_size in zip(self.datasets, self.slice_sizes)]
        for batch in zip(*data_loaders):
            seq_len = max([data[0].size(1) for data in batch])
            X = torch.cat([self.pad_sequence(data[0], value=0,   seq_len=seq_len, dim=1) for data in batch])
            Y = torch.cat([self.pad_sequence(data[1], value=255, seq_len=seq_len, dim=1) for data in batch])
            X = torch.div(X.float(), 255)
            for key, value in self.label_map.items():
                Y = torch.where(Y == key, value, Y)
            idx = torch.randperm(X.size(0))
            yield X[idx], Y[idx]

    def __len__(self):
        return self.n_batches

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

''' #! Downloads pre-trained model
import satlaspretrain_models
feature_extractor = satlaspretrain_models.Weights()
feature_extractor = feature_extractor.get_pretrained_model(model_identifier='Aerial_SwinB_SI', fpn=True, device='cpu')
torch.save(feature_extractor, path.join(paths.models, 'Aerial_SwinB_SI.pth'))
del feature_extractor
'''

# Initialises model components
image_encoder    = ImageEncoder(feature_extractor=torch.load(f'{paths.models}/Aerial_SwinB_SI.pth'))
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
