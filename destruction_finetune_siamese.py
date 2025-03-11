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
import torchvision

from destruction_models import *
from destruction_utilities import *
from pytorch_lightning import callbacks, loggers, profilers
from torch import optim
from torchmetrics import classification

# Utilities
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
params = argparse.Namespace(
    cities=['aleppo'],
    batch_size=64,
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})

#%% INITIALISES DATA MODULE

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

class ZarrDataModule(pl.LightningDataModule):
    
    def __init__(self, train_datafiles:list, valid_datafiles:list, test_datafiles:list, batch_size:int, label_map:dict, shuffle:bool=True) -> None:
        super().__init__()
        self.train_datafiles = train_datafiles
        self.valid_datafiles = valid_datafiles
        self.test_datafiles = test_datafiles
        self.batch_size = batch_size
        self.label_map = label_map
        self.shuffle = shuffle
    
    def setup(self, stage:str=None):
        self.train_datasets = [ZarrDataset(**train_datafiles[city]) for city in params.cities]
        self.valid_datasets = [ZarrDataset(**valid_datafiles[city]) for city in params.cities]
        self.test_datasets  = [ZarrDataset(**test_datafiles[city])  for city in params.cities]

    def train_dataloader(self):
        return ZarrDataLoader(datafiles=self.train_datafiles, datasets=self.train_datasets, label_map=self.label_map, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return ZarrDataLoader(datafiles=self.valid_datafiles, datasets=self.valid_datasets, label_map=self.label_map, batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self):
        return ZarrDataLoader(datafiles=self.test_datafiles, datasets=self.test_datasets,   label_map=self.label_map, batch_size=self.batch_size, shuffle=self.shuffle)

# Initialises datasets
train_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_train_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_train_balanced.zarr') for city in params.cities]))
valid_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_valid_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_valid_balanced.zarr') for city in params.cities]))
test_datafiles  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_test_balanced.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_test_balanced.zarr')  for city in params.cities]))
data_module = ZarrDataModule(train_datafiles=train_datafiles, valid_datafiles=valid_datafiles, test_datafiles=test_datafiles, batch_size=params.batch_size, label_map=params.label_map, shuffle=True)
del train_datafiles, valid_datafiles, test_datafiles

''' Check data module
data_module.setup()
X, Y = next(data_module.train_dataloader())
for idx in np.random.choice(range(len(X)), size=5, replace=False):
    display_sequence(X[idx], [0] + [int(Y[idx])])
del X, Y, idx
'''

#%% INITIALISES MODEL MODULE

def contrastive_loss(distance:torch.Tensor, label:torch.Tensor, margin:float=1.0) -> torch.Tensor:
    loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

class SiameseModel(nn.Module):
    
    def __init__(self, backbone:str):
        super().__init__()
        self.processor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
        self.encoder   = transformers.ViTMAEModel.from_pretrained(backbone)
        self.output    = nn.Linear(768, 1)

    def forward_branch(self, X:torch.Tensor) -> torch.Tensor:
        H = self.processor(X, return_tensors='pt', do_resize=True).to(device)
        H = self.image_encoder(**H.to(device))
        H = H.last_hidden_state[:,0,:]
        return H

    def forward(self, X0:torch.Tensor, X1:torch.Tensor, Y:torch.Tensor=None) -> torch.Tensor:
        H0 = self.forward_branch(X0)
        H1 = self.forward_branch(X1)
        D  = F.pairwise_distance(H0, H1, keepdim=True)
        Y  = self.output(D)
        return D, Y

class SiameseModule(pl.LightningModule):
    
    def __init__(self, backbone:str, learning_rate:float=1e-4, weight_decay:float=0.05):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = SiameseModule(backbone=backbone)
        self.contrastive_loss = contrastive_loss
        self.sigmoid_loss = torchvision.ops.sigmoid_focal_loss
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.trainable     = None

    def freeze_encoder(self):
        self.trainable = [param.requires_grad for param in self.model.encoder.parameters()]
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print('Encoder frozen')

    def unfreeze_encoder(self):
        for param, status in zip(self.model.encoder.parameters(), self.trainable):
            param.requires_grad = status
        self.trainer.strategy.setup_optimizers(self.trainer)
        print('Encoder unfrozen, optimisers reset')

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        return Y
    
    def training_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y   = batch
        D, Yh  = self.model(X).squeeze()
        C_loss = self.contrastive_loss(D, Y, reduction='mean')
        S_loss = self.sigmoid_loss(Yh, Y, reduction='mean')
        train_loss = C_loss + S_loss
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y   = batch
        D, Yh  = self.model(X).squeeze()
        C_loss = self.contrastive_loss(D, Y, reduction='mean')
        S_loss = self.sigmoid_loss(Yh, Y, reduction='mean')
        val_loss = C_loss + S_loss
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss
    
    def test_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y   = batch
        D, Yh  = self.model(X).squeeze()
        C_loss = self.contrastive_loss(D, Y, reduction='mean')
        S_loss = self.sigmoid_loss(Yh, Y, reduction='mean')
        test_loss = C_loss + S_loss
        self.log('test_loss', test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self) -> dict:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {'optimizer':optimizer}
    
# Initialises model
model_module = SiameseModule(backbone=f'{paths.models}/checkpoint-9920')
model_module.freeze_encoder()
model_name = 'destruction_finetune_siamese'

#%% TRAINS MODEL

logger = loggers.CSVLogger(
    save_dir='../data/models/logs', 
    name=model_name, 
    version=0
)

model_checkpoint = callbacks.ModelCheckpoint(
    dirpath='../data/models',
    filename=f'{model_name}-{{epoch:02d}}-{{step:05d}}',
    monitor='step',
    every_n_train_steps=1e3,
    save_top_k=1,
    save_last=True
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=1,
    verbose=True
)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator='mps',
    log_every_n_steps=1e3,
    logger=logger,
    callbacks=[model_checkpoint, early_stopping],
    profiler=profilers.SimpleProfiler()
)

# Fits model
trainer.fit(
    model=model_module, 
    datamodule=data_module,
    ckpt_path=model_checkpoint.last_model_path if model_checkpoint.last_model_path else None,
)

# Saves model
trainer.save_checkpoint(f'../data/models/{model_name}-e{trainer.current_epoch:02d}-s{trainer.global_step:05d}.ckpt')
empty_cache()
#%%
