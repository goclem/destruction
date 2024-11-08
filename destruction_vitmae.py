#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Vision transformer masked auto-encoder
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import accelerate
import argparse
import numpy as np
import transformers
import torch

from destruction_utilities import *
from matplotlib import colors, pyplot

# Utilities
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
params = argparse.Namespace(
    tile_size=224,
    batch_size=32,
    cities=['aleppo', 'moschun'])

#%% UTILITIES

class ZarrDatasetX(utils.data.Dataset):
    '''Zarr dataset for PyTorch'''
    def __init__(self, images_zarr:str):
        self.images = zarr.open(images_zarr, mode='r')
        self.length = len(self.images)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.images[idx])
        return X

class ZarrDataLoaderX:
    '''Zarr data loader for PyTorch'''
    def __init__(self, datasets:list, batch_size:int, preprocessor:callable=None):
        self.datasets     = datasets
        self.batch_size   = batch_size
        self.indices      = np.zeros(len(datasets), dtype=int)
        self.data_sizes   = np.array([len(dataset) for dataset in datasets]) 
        self.preprocessor = preprocessor

    def compute_slice_sizes(self):
        slice_sizes = np.array([np.log(len(dataset)) for dataset in self.datasets])
        slice_sizes = np.divide(slice_sizes, slice_sizes.sum())
        slice_sizes = np.random.multinomial(self.batch_size, slice_sizes)
        return slice_sizes

    def __iter__(self):
        while True:
            slice_sizes = self.compute_slice_sizes()
            if np.any(self.indices + slice_sizes > self.data_sizes):
                break
            X = torch.cat([dataset[indice:indice+slice_size] for dataset, indice, slice_size in zip(self.datasets, self.indices, slice_sizes) if slice_size > 0])
            X = X.moveaxis(1, -1)
            X = self.preprocessor(X, return_tensors='pt').pixel_values
            X = {'pixel_values':X}
            self.indices += slice_sizes
            yield X

# Custom Trainer to use PyTorch DataLoader
class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, train_dataloader, eval_dataloader, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        return self._train_dataloader
    
    def get_eval_dataloader(self, eval_dataset=None):
        return self._eval_dataloader
    
def unprocess_images(images:torch.Tensor, preprocessor) -> torch.Tensor:
    std    = torch.tensor(preprocessor.image_std).view(1, 3, 1, 1)
    mean   = torch.tensor(preprocessor.image_mean).view(1, 3, 1, 1)
    images = torch.clip((images * std + mean) * 255, 0, 255).to(torch.uint8)
    return images

#%% INITIALISES DATA LOADERS

# Initialises datasets
train_datasets = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_train.zarr') for city in params.cities]))
valid_datasets = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_valid.zarr') for city in params.cities]))
test_datasets  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_test.zarr')  for city in params.cities]))

train_datasets = [ZarrDatasetX(**train_datasets[city]) for city in params.cities]
valid_datasets = [ZarrDatasetX(**valid_datasets[city]) for city in params.cities]
test_datasets  = [ZarrDatasetX(**test_datasets[city])  for city in params.cities]

# Intialises data loaders
preprocessor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
train_loader = ZarrDataLoaderX(train_datasets, batch_size=params.batch_size, preprocessor=preprocessor)
valid_loader = ZarrDataLoaderX(valid_datasets, batch_size=params.batch_size, preprocessor=preprocessor)
test_loader  = ZarrDataLoaderX(test_datasets,  batch_size=params.batch_size, preprocessor=preprocessor)

# next(iter(train_loader)) # Checks data loader
del train_datasets, valid_datasets, test_datasets

#%% FINE-TUNES MODEL

# Loads pre-trained model
model = transformers.ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
model = model.to(device)

training_args = transformers.TrainingArguments(
    output_dir='../results',
    num_train_epochs=5,
    per_device_train_batch_size=params.batch_size,
    per_device_eval_batch_size=params.batch_size,
    warmup_steps=500,
    max_steps=10000, #! Required
    weight_decay=0.01,
    logging_dir='../logs',
    logging_steps=100,
    evaluation_strategy='epoch',
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataloader=train_loader,
    eval_dataloader=valid_loader
)

trainer.train()

model.save_pretrained(f'{paths.models}/mae')
del model, training_args, trainer

#%% CHECKS PREDICTIONS



''' Checks data
display(unprocess_images(X_train[0], preprocessor)[0], title=Y_train[0].item())

image=io.read_image('/Users/goclem/Downloads/morrisseau_close1.jpg')
display(image, title='Original image')

image = unprocess_images(X_train[0], preprocessor)[0]
image = image.permute(1, 2, 0)

%matplotlib widget
fig, ax = pyplot.subplots(1, figsize=(10, 10))
ax.imshow(image, cmap='gray')
pyplot.show()

''' 

#%%