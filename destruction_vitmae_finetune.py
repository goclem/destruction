#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Fine-tunes the vision transformer on destruction sequences
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
    batch_size=8,
    seq_len=10,
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

    def __init__(self, datafiles:list, datasets:list, label_map:dict, batch_size:int, seq_len:int):
        self.datafiles    = datafiles
        self.datasets     = datasets
        self.label_map    = label_map
        self.batch_size   = batch_size
        self.seq_len      = seq_len
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
    
    def crop_sequence(self, x:torch.Tensor, y:torch.Tensor, seq_len:int, dim:int=1) -> torch.Tensor:
        start = torch.randint(low=0, high=x.size(dim) - seq_len + 1, size=(1, )).item()
        x = x.narrow(dim=dim, start=start, length=seq_len)
        y = y.narrow(dim=dim, start=start, length=seq_len)
        return x, y

    def pad_sequence(self, x:torch.Tensor, y:torch.Tensor, seq_len:int, dim:int=1) -> torch.Tensor:
        x_pad = torch.zeros(2 * x.ndim, dtype=int).index_fill(0, -torch.tensor(2 * dim + 1), seq_len - x.size(dim))
        y_pad = torch.zeros(2 * y.ndim, dtype=int).index_fill(0, -torch.tensor(2 * dim + 1), seq_len - y.size(dim))
        x = nn.functional.pad(x, pad=x_pad.tolist(), value=0)
        y = nn.functional.pad(y, pad=y_pad.tolist(), value=255)
        return x, y
    
    def __len__(self):
        return len(self.data_indices) - 1
    
    def __iter__(self):
        self.batch_index = 0
        for city in self.datafiles:
            print(f'Shuffling {city}')
            shuffle_zarr(self.datafiles[city]['images_zarr'])
            shuffle_zarr(self.datafiles[city]['labels_zarr'])
        return self

    def __next__(self):
        if self.batch_index == len(self):
            raise StopIteration 
        # Loads sequences
        X, Y = list(), list()
        for dataset, indices in zip(self.datasets, self.data_indices.T):
            start = indices[self.batch_index]
            end   = indices[self.batch_index + 1]
            if start != end: # Skips empty batches
                X_ds, Y_ds = dataset[start:end]
                X.append(X_ds), 
                Y.append(Y_ds)
        # Normalises sequences
        for i in range(len(X)):
            if X[i].size(1) > self.seq_len:
                X[i], Y[i] = self.crop_sequence(x=X[i], y=Y[i], seq_len=self.seq_len)
            elif X[i].size(1) < self.seq_len:
                X[i], Y[i] = self.pad_sequence(x=X[i], y=Y[i], seq_len=self.seq_len)
        X = torch.cat(X)
        Y = torch.cat(Y)
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
train_loader = ZarrDataLoader(datafiles=train_datafiles, datasets=train_datasets, batch_size=params.batch_size, seq_len=params.seq_len, label_map=params.label_map)
valid_loader = ZarrDataLoader(datafiles=valid_datafiles, datasets=valid_datasets, batch_size=params.batch_size, seq_len=params.seq_len, label_map=params.label_map)
test_loader  = ZarrDataLoader(datafiles=test_datafiles,  datasets=test_datasets,  batch_size=params.batch_size, seq_len=params.seq_len, label_map=params.label_map)

del train_datafiles, valid_datafiles, test_datafiles, train_datasets, valid_datasets, test_datasets

''' Checks data loaders
X, Y = next(train_loader)
for i in range(5):
    display_sequence(X[i], Y[i], grid_size=(4, 5))
del X, Y
''' 

#%% INITIALISES MODEL

class ModelWrapper(nn.Module):
    def __init__(self, preprocessor:nn.Module, image_encoder:nn.Module, sequence_encoder:nn.Module, prediction_head:nn.Module):
        super().__init__()
        self.preprocessor     = preprocessor
        self.image_encoder    = image_encoder
        self.sequence_encoder = sequence_encoder
        self.prediction_head  = prediction_head

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        # Encodes images
        H = X.view(-1, 3, 128, 128) # n x t x c x h x w > nt x c x h x w
        H = self.preprocessor(H, return_tensors='pt', do_resize=True)
        H = self.image_encoder(**H.to(device)) # nt x c x h x w > nt x d x k
        H = H.last_hidden_state[:, 0, :] # nt x d x k > nt x k
        H = H.view(X.size(0), X.size(1), H.size(-1)) # nt x k > n x t x k
        # Encodes sequence
        H = self.sequence_encoder(H) # n x t x k > n x t x k
        Y = self.prediction_head(H)  # n x t x k > n x t x 1
        return Y

# Initialises model components #! Change model path
preprocessor     = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
image_config     = transformers.ViTMAEConfig.from_pretrained('facebook/vit-mae-base')
image_encoder    = transformers.ViTMAEModel.from_pretrained('facebook/vit-mae-base', config=image_config)
sequence_config  = dict(input_dim=768, max_length=23, n_heads=4, hidden_dim=768, n_layers=2, dropout=0.0)
sequence_encoder = SequenceEncoder(**sequence_config)
prediction_head  = PredictionHead(input_dim=768, output_dim=1)

# Initialises model
model = ModelWrapper(preprocessor, image_encoder, sequence_encoder, prediction_head)
model = model.to(device)
count_parameters(model)

del preprocessor, image_encoder, sequence_encoder, prediction_head

#%% ALIGNING PARAMETERS

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
set_trainable(model.image_encoder, False)
count_parameters(model)

optimiser = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
criterion = BceLoss(focal=True, drop_nan=True, alpha=0.25, gamma=2.0)

# Training
train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device,
      criterion=criterion, 
      optimiser=optimiser, 
      model_path='../models/vitmae_align.pth',
      n_epochs=100, 
      patience=3,
      accumulate=4)

# Clears GPU memory
empty_cache(device)

#%% FINE TUNES MODEL

# Unfreezes image encoder's parameters
set_trainable(model.image_encoder, True)
count_parameters(model)

optimiser = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
criterion = BceLoss(focal=True, drop_nan=True, alpha=0.25, gamma=2.0)

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
