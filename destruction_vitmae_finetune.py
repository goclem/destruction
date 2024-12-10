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

from destruction_utilities import *

# Utilities
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
params = argparse.Namespace(
    cities=['aleppo', 'moschun'],
    batch_size=16,
    seq_len=20,
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

    def __init__(self, datafiles:list, datasets:list, label_map:dict, batch_size:int, seq_len:int, preprocessor=None):
        self.datafiles    = datafiles
        self.datasets     = datasets
        self.label_map    = label_map
        self.batch_size   = batch_size
        self.seq_len      = seq_len
        self.batch_index  = 0
        self.data_sizes   = np.array([len(dataset) for dataset in datasets])
        self.data_indices = self.compute_data_indices()
        self.preprocessor = preprocessor

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
        for datafile in self.datafiles:
            print(f'Shuffling {datafile}')
            shuffle_zarr(datafile)
        return self

    def crop_sequence(self, X:torch.Tensor, Y:torch.Tensor, length:int, dim:int=1) -> torch.Tensor:
        start = torch.randint(low=0, high=X.size(dim) - length + 1, size=(1, )).item()
        X = X.narrow(dim, start, length)
        Y = Y.narrow(dim, start, length)
        return X, Y

    def pad_sequence(self, X:torch.Tensor, Y:torch.Tensor, length:int, dim:int=1) -> torch.Tensor:
        x_pad = torch.zeros(2 * X.ndim, dtype=int).index_fill(0, - torch.tensor(2 * dim + 1), length - X.size(dim))
        y_pad = torch.zeros(2 * Y.ndim, dtype=int).index_fill(0, - torch.tensor(2 * dim + 1), length - Y.size(dim))
        X = nn.functional.pad(X, pad=x_pad.tolist(), value=0)
        Y = nn.functional.pad(Y, pad=y_pad.tolist(), value=255)
        return X, Y
    
    def __next__(self):
        if self.batch_index == len(self):
            raise StopIteration 
        # Loads sequences
        X, Y = list(), list()
        for dataset, indices in zip(self.datasets, self.data_indices.T):
            start = indices[self.batch_index]
            end   = indices[self.batch_index + 1]
            X_ds, Y_ds = dataset[start:end]
            X.append(X_ds), 
            Y.append(Y_ds)
        # Normalises sequences
        for i in range(len(X)):
            if X[i].size(1) > self.seq_len:
                X[i], Y[i] = self.crop_sequence(X[i], Y[i], length=self.seq_len)
            elif X[i].size(1) < self.seq_len:
                X[i], Y[i] = self.pad_sequence(X[i],  Y[i], length=self.seq_len)
        X = torch.cat(X)
        Y = torch.cat(Y)
        # Preprocesses sequences
        if self.preprocessor is not None:
            n, t, c, h, w = X.size()
            X  = self.preprocessor(X.reshape(-1, c, h, w), return_tensors='pt', do_resize=True)['pixel_values']
            X  = X.reshape(n * t, c, 224, 224)
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

# Initialises model components #! Change model path
config = transformers.ViTMAEConfig.from_pretrained('facebook/vit-mae-base')
model  = transformers.ViTMAEModel.from_pretrained('facebook/vit-mae-base', config=config)
model  = model.to(device)



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
        train_loss = optimise(model=model, loader=train_loader, device=device, criterion=criterion, optimiser=optimiser, accumulate=accumulate)
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
