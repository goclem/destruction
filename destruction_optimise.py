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

# Utilities
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
params = argparse.Namespace(tile_size=128, grid_size=20, batch_size=16, mapping={'0':0, '1':0, '2':0, '3':1, '255':255})

#%% READS DATA

def load_sequences(files:list, grid_size:int, tile_size:int, stride:int=None) -> torch.Tensor:
    window    = center_window(source=files[0], size=(grid_size*tile_size, grid_size*tile_size))
    sequences = [read_raster(file, window=window, dtype='uint8') for file in files]
    sequences = [torch.tensor(image).permute(2, 0, 1) for image in sequences]
    sequences = [image_to_tiles(image, tile_size=tile_size, stride=stride) for image in sequences]
    sequences = torch.stack(sequences).swapaxes(1, 0)
    return sequences

# Reads images
images = search_data(pattern('aleppo', 'image'))
images = load_sequences(images, grid_size=params.grid_size, tile_size=params.tile_size)
images = images / 255

# Reads labels
labels = search_data(pattern(city='aleppo', type='label'))
labels = load_sequences(labels, grid_size=params.grid_size, tile_size=1)
labels = labels.squeeze(2, 3) 

# Remaps label values
labels = labels.apply_(lambda val: params.mapping.get(str(val))).type(torch.float)
labels = torch.where(labels == 255, torch.tensor(float('nan')), labels)

# Reads samples
samples = search_data('aleppo_samples.tif$')
samples = load_sequences(samples, grid_size=params.grid_size, tile_size=1)
samples = samples.squeeze()

#? Balances sequences
def balance_sequences(labels:torch.Tensor) -> torch.Tensor:
    subset = np.any(labels.numpy().astype(bool), axis=(1,2))
    negobs = np.random.choice(np.argwhere(~subset).flatten(), subset.sum(), replace=False)
    np.put(subset, negobs, True)
    subset = torch.Tensor(subset).type(torch.bool)
    return subset

subset  = balance_sequences(labels)
images  = images[subset]
labels  = labels[subset]
samples = samples[subset]
del subset

#%% INITIALISES DATA LOADERS

class Dataset(utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    
# Split samples
_, images_train, images_valid, images_test = sample_split(images, samples)
_, labels_train, labels_valid, labels_test = sample_split(labels, samples)
del samples, _

train_loader = utils.data.DataLoader(Dataset(images_train, labels_train), batch_size=params.batch_size, shuffle=True)
valid_loader = utils.data.DataLoader(Dataset(images_valid, labels_valid), batch_size=params.batch_size, shuffle=True)
test_loader  = utils.data.DataLoader(Dataset(images_test,  labels_test),  batch_size=params.batch_size, shuffle=False)
del Dataset, images, labels, images_train, images_valid, images_test, labels_train, labels_valid, labels_test

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

#? Parameter alignment (freezes  feature extractor)
model.image_encoder = set_trainable(model.image_encoder, False)
optimiser = optim.AdamW(model.parameters(), lr=1e-4)
count_parameters(model)

''' #? Fine tuning (unfreezes feature extractor)
model.image_encoder = set_trainable(model.image_encoder, True)
optimiser = optim.AdamW(model.parameters(), lr=1e-5)
count_parameters(model)
'''

# Loss function
criterion = BceLoss(focal=False, drop_nan=True)

# Training
train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device, 
      criterion=criterion, 
      optimiser=optimiser, 
      n_epochs=10, 
      patience=3,
      accumulate=1)

# Restores best model
model = torch.load(f'{paths.models}/ModelWrapper_best.pth')

# Testing
validate(model=model, 
         loader=test_loader, 
         device=device, 
         criterion=criterion)

empty_cache(device)

#%% CHECKS PREDICTIONS

''' Checks predictions
with torch.no_grad():
    X, Y = next(iter(test_loader))
    Yh = model(X.to(device)).cpu()

for i in np.random.choice(range(len(Y)), 2, replace=False):
    titles = [f'Y: {y:.2f} | Yh: {np.round(yh):.2f} ({yh:.2f})'for y, yh in zip(Y[i].squeeze().tolist(), Yh[i].squeeze().tolist())]
    display_sequence(X[i], titles, grid_size=(5,5))
del titles

Yh = predict(model, loader=test_loader, device=device)
'''

#%%
