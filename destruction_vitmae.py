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
import datasets
import numpy as np
import transformers
import torch

from destruction_utilities import *
from matplotlib import colors, pyplot
from PIL import Image
from torchvision import transforms

# Utilities
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

params = argparse.Namespace(
    tile_size=224, 
    batch_size_image=64,
    batch_size_sequence=32,
    max_size=896,
    cities=['aleppo'],
    batch_size=8,
    samples=[0.8, 0.1, 0.1],
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})

#%% INITIALISES DATA LOADERS

# Datasets
train_datasets = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_train.zarr') for city in params.cities]))
valid_datasets = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_valid.zarr') for city in params.cities]))
test_datasets  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_test.zarr')  for city in params.cities]))

# Intialises data loaders
train_loader = [ZarrDatasetX(**train_datasets[city]) for city in params.cities]
valid_loader = [ZarrDatasetX(**valid_datasets[city]) for city in params.cities]
test_loader  = [ZarrDatasetX(**test_datasets[city])  for city in params.cities]
train_loader = ZarrDataLoader(train_loader, batch_size=params.batch_size, label_map=params.label_map)
valid_loader = ZarrDataLoader(valid_loader, batch_size=params.batch_size, label_map=params.label_map)
test_loader  = ZarrDataLoader(test_loader,  batch_size=params.batch_size, label_map=params.label_map)

''' Checks data loaders
X, _ = next(iter(train_loader))
X    = X.view()
del X, _
''' 

batch_size  = 64
datasets    = dict(aleppo=20, raqqa=50, homs=100)
slice_sizes = np.array([np.log(dataset) for dataset in datasets.values()])
slice_sizes = np.divide(slice_sizes, slice_sizes.sum())
slice_sizes = np.random.choice(list(datasets.keys()), batch_size, p=slice_sizes)
slice_sizes = np.unique(slice_sizes, return_counts=True)
é = dict(zip(slice_sizes[0], slice_sizes[1]))

class ZarrDataLoaderX:
    '''Zarr data loader for PyTorch'''
    def __init__(self, datasets:list, batch_size:int):
        self.datasets    = datasets
        self.batch_size  = batch_size
        self.slice_sizes = None

    def compute_slice_sizes(self):
        slice_sizes = np.array([np.log(len(dataset)) for dataset in self.datasets])
        slice_sizes = np.divide(slice_sizes, slice_sizes.sum())
        slice_sizes = np.random.choice(datasets.keys(), self.batch_size, p=slice_sizes)
        slice_sizes = np.unique(slice_sizes, return_counts=True)
        self.slice_sizes = dict(zip(dataset_slices[0], dataset_slices[1]))

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


#%% FINE-TUNING

# Model configuration
preprocessor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')

def unprocess_images(images:torch.Tensor, preprocessor) -> torch.Tensor:
    std    = torch.tensor(preprocessor.image_std).view(1, 3, 1, 1)
    mean   = torch.tensor(preprocessor.image_mean).view(1, 3, 1, 1)
    images = torch.clip((images * std + mean) * 255, 0, 255).to(torch.uint8)
    return images

# Loads images and labels
images = search_data(pattern(city='aleppo', type='image'))[:2]
images = load_sequences(images, tile_size=224)
images = images.contiguous().view(-1, images.size(2), images.size(3), images.size(4))

# Pre-processes images
images = images.moveaxis(1, -1)
loader = utils.data.DataLoader(images, batch_size=1024, shuffle=False)
images = list()
for i, batch in enumerate(loader):
    print(f'Batch {i+1:02d}/{len(loader)}')
    images.append(preprocessor(batch, return_tensors='pt').pixel_values)
images = torch.cat(images, axis=0)
del loader, i, batch

# Splits samples
X_train, X_valid, X_test = sample_split(images, sizes=dict(train=0.8, valid=0.1, test=0.1), seed=0)
del images

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

#%% FINE-TUNES MODEL

# Custom Trainer to use PyTorch DataLoader
class CustomTrainer(transformers.Trainer):
    def get_train_dataloader(self):
        return train_loader
    def get_eval_dataloader(self, eval_dataset=None):
        return valid_loader

# Initialises datasets
dataset = datasets.DatasetDict({
    'train': datasets.Dataset.from_dict({'pixel_values':X_train}),
    'valid': datasets.Dataset.from_dict({'pixel_values':X_valid}),
    'test':  datasets.Dataset.from_dict({'pixel_values':X_test})
})

# Loads pre-trained model
model = transformers.ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
model = model.to(device)

# Training routine
training_args = transformers.TrainingArguments(
    output_dir='../results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../logs',
    logging_steps=100,
    evaluation_strategy='epoch',
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid']
)

trainer.train()

model.save_pretrained(f'{paths.models}/mae')
del model, dataset, training_args, trainer


#%%