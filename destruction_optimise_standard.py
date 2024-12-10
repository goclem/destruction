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
from torcheval import metrics

# Utilities
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
params = argparse.Namespace(
    cities=['aleppo', 'moschun'],
    tile_size=128,
    batch_size=8,
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

    def __init__(self, datasets:list, label_map:dict, batch_size:int):
        self.datasets    = datasets
        self.batch_size  = batch_size
        self.label_map   = label_map
        self.slice_sizes = None
        self.n_batches   = None
        self.compute_slice_sizes()

    def compute_slice_sizes(self):
        dataset_sizes    = torch.tensor([len(dataset) for dataset in self.datasets])
        self.slice_sizes = (torch.div(dataset_sizes, dataset_sizes.sum()) * self.batch_size).round().int()
        self.n_batches   = np.divide(dataset_sizes, self.slice_sizes, where=self.slice_sizes > 0).floor().int()
        self.n_batches   = self.n_batches[self.n_batches > 0].min()
    
    def pad_sequence(self, sequence:torch.Tensor, value:int, seq_len:int=None, dim:int=1) -> torch.Tensor:
        pad = torch.zeros(2*len(sequence.size()), dtype=int)
        pad = pad.index_fill(0, torch.tensor(2*dim), seq_len-sequence.size(dim)).flip(0).tolist()
        pad = nn.functional.pad(sequence, pad=pad, value=value)
        return pad

    def __iter__(self):
        data_loaders = [utils.data.DataLoader(dataset, batch_size=int(slice_size)) for dataset, slice_size in zip(self.datasets, self.slice_sizes) if slice_size > 0]
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
        loss = torch.mean(loss) #? Mean weighted by number of non-missing
        return loss

#%% INITIALISES DATA LOADERS

# Initialises datasets
train_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_train.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_train.zarr') for city in params.cities]))
valid_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_valid.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_valid.zarr') for city in params.cities]))
test_datafiles  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_test.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_test.zarr')  for city in params.cities]))
train_datasets  = [ZarrDataset(**train_datafiles[city]) for city in params.cities]
valid_datasets  = [ZarrDataset(**valid_datafiles[city]) for city in params.cities]
test_datasets   = [ZarrDataset(**test_datafiles[city])  for city in params.cities]

# Intialises data loaders
train_loader = ZarrDataLoader(train_datasets, batch_size=params.batch_size, label_map=params.label_map)
valid_loader = ZarrDataLoader(valid_datasets, batch_size=params.batch_size, label_map=params.label_map)
test_loader  = ZarrDataLoader(test_datasets,  batch_size=params.batch_size, label_map=params.label_map)

del train_datafiles, valid_datafiles, test_datafiles, train_datasets, valid_datasets, test_datasets

# Prints excluded cities
[print(f'Excluding: {city}') for city, size in zip(params.cities, train_loader.slice_sizes) if size == 0]

''' Checks data loaders
X, Y = next(train_loader)
for i in range(5):
    display_sequence(X[i], Y[i], grid_size=(5,5))
del X, Y
''' 

#%% INITIALISES MODEL

# Initialises model components
#? feature_extractor = torch.load(f'{paths.models}/Aerial_SwinB_SI.pth')
feature_extractor = ResNextExtractor(dropout=0.0)
image_encoder     = ImageEncoder(feature_extractor=feature_extractor)
sequence_encoder  = dict(input_dim=512, max_length=25, n_heads=4, hidden_dim=512, n_layers=2, dropout=0.0)
sequence_encoder  = SequenceEncoder(**sequence_encoder)
prediction_head   = PredictionHead(input_dim=512, output_dim=1)

# Initialises model wrapper
model = ModelWrapper(image_encoder, sequence_encoder, prediction_head)
model.to(device)

# Checks model parameters
count_parameters(model)
count_parameters(model.image_encoder)
count_parameters(model.sequence_encoder)

del image_encoder, sequence_encoder, prediction_head

#%% ALIGNING PARAMETERS

''' #? Loads previous checkpoint
if path.exists(f'{paths.models}/ModelWrapper_best.pth'):
    model = torch.load(f'{paths.models}/ModelWrapper_best.pth')
'''

#? Parameter alignment i.e. freezes image encoder's parameters
# set_trainable(model.image_encoder.feature_extractor, False)
# count_parameters(model)

optimiser = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
criterion = BceLoss(focal=False, drop_nan=True, alpha=0.25, gamma=2.0)

def train(model:nn.Module, train_loader, valid_loader, device:torch.device, criterion, optimiser, n_epochs:int=1, patience:int=1, accumulate:int=1, label:str=''):
    best_loss, counter = torch.tensor(float('inf')), 0
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1:03d}/{n_epochs:03d}')
        train_loss = optimise(model=model, train_loader=train_loader, device=device, criterion=criterion, optimiser=optimiser, accumulate=accumulate)
        # Early stopping
        valid_loss = validate(model=model, loader=valid_loader, device=device, criterion=criterion)
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model, f'{paths.models}/{model.__class__.__name__}_{label}_best.pth')
            # Shuffles zarr datasets
            for city in params.cities:
                shuffle_zarr(**train_datafiles[city])
                shuffle_zarr(**valid_datafiles[city])
        else:
            counter += 1
            if counter >= patience:
                print('- Early stopping')
                break

# Training
train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device, 
      criterion=criterion, 
      optimiser=optimiser, 
      n_epochs=25, 
      patience=3,
      accumulate=4,
      label='resnext_encoder')

# Clears GPU memory
empty_cache(device)

#%% FINE TUNES ENTIRE MODEL

'''
#? Fine tuning i.e. unfreezes image encoder's parameters
model = torch.load(f'{paths.models}/ModelWrapper_aligmnent_best.pth') # Loads best model
set_trainable(model.image_encoder.feature_extractor, True)

optimiser = optim.AdamW(model.parameters(), lr=1e-5)
count_parameters(model)

train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device, 
      criterion=criterion, 
      optimiser=optimiser, 
      n_epochs=25, 
      patience=3,
      accumulate=4,
      label='finetuning')

# Clears GPU memory
empty_cache(device)
'''

#%% ESTIMATES THRESHOLD

model = torch.load(f'{paths.models}/ModelWrapper_finetuning_best.pth') # Loads best model

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

model.eval()
X, Y = next(iter(test_loader))
susbset = torch.sum(Y==1, axis=(1, 2)) > 0
X, Y = X[susbset], Y[susbset]

with torch.no_grad():
    Yh = model(X.to(device)).cpu()
Y, Yh  = Y.squeeze(), Yh.squeeze()

threshold = 0.5
status = torch.where(torch.isnan(Y), torch.nan, torch.eq(Y, Yh > threshold))

for i in np.random.choice(range(len(X)), 2, replace=False):
    titles = [f'{s:.0f}\nY: {y:.0f} - Yh: {yh > threshold:.0f} ({yh:.2f})' for s, y, yh in zip(status[i], Y[i], Yh[i])]
    display_sequence(X[i], titles, grid_size=(5,5))
del titles

#%%
