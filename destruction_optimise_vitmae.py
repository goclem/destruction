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
import matplotlib.pyplot as plt
import numpy as np
import transformers
import torch

from destruction_utilities import *

# Utilities
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
params = argparse.Namespace(batch_size=64, cities=['aleppo', 'moschun'])

#%% TRAINING UTILITIES

class ZarrDataset(utils.data.Dataset):

    def __init__(self, dataset:str):
        self.images = zarr.open(dataset, mode='r')
        self.length = len(self.images)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.images[idx])
        return X

class ZarrDataLoader:

    def __init__(self, datafiles:list, datasets:list, batch_size:int, preprocessor=None):
        self.datafiles    = datafiles
        self.datasets     = datasets
        self.batch_size   = batch_size
        self.batch_index  = 0
        self.data_sizes   = np.array([len(dataset) for dataset in datasets])
        self.data_indices = self.compute_data_indices()
        self.preprocessor = preprocessor

    def compute_data_indices(self):
        slice_sizes  = np.cbrt(self.data_sizes) #! Large impact
        slice_sizes  = np.divide(slice_sizes, slice_sizes.sum())
        slice_sizes  = np.random.multinomial(self.batch_size, slice_sizes, size=int(np.max(self.data_sizes / self.batch_size)))
        data_indices = np.row_stack((np.zeros(len(self.data_sizes), dtype=int), np.cumsum(slice_sizes, axis=0)))
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

    def __next__(self):
        if self.batch_index == len(self):
            raise StopIteration 
        X = list()
        for dataset, indices in zip(self.datasets, self.data_indices.T):
            start = indices[self.batch_index]
            end   = indices[self.batch_index + 1]
            X.append(dataset[start:end])
        X = torch.cat(X)
        if self.preprocessor is not None:
            X = self.preprocessor(X.moveaxis(1, -1), return_tensors='pt', do_resize=True)
        self.batch_index += 1 
        return X

class ZarrTrainer(transformers.Trainer):

    def __init__(self, *args, train_dataloader, eval_dataloader, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_dataloader = train_dataloader
        self._eval_dataloader  = eval_dataloader

    def get_train_dataloader(self):
        return self._train_dataloader
    
    def get_eval_dataloader(self, eval_dataset=None):
        return self._eval_dataloader

#%% INITIALISES DATA LOADERS

# Initialises datasets
train_datafiles = [f'{paths.data}/{city}/zarr/images_train_vitmae.zarr' for city in params.cities]
valid_datafiles = [f'{paths.data}/{city}/zarr/images_valid_vitmae.zarr' for city in params.cities]
test_datafiles  = [f'{paths.data}/{city}/zarr/images_test_vitmae.zarr'  for city in params.cities]
train_datasets  = [ZarrDataset(datafile) for datafile in train_datafiles]
valid_datasets  = [ZarrDataset(datafile) for datafile in valid_datafiles]
test_datasets   = [ZarrDataset(datafile) for datafile in test_datafiles]

# Intialises data loaders
preprocessor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
train_loader = ZarrDataLoader(datafiles=train_datafiles, datasets=train_datasets, batch_size=params.batch_size, preprocessor=preprocessor)
valid_loader = ZarrDataLoader(datafiles=valid_datafiles, datasets=valid_datasets, batch_size=params.batch_size, preprocessor=preprocessor)
test_loader  = ZarrDataLoader(datafiles=test_datafiles,  datasets=test_datasets,  batch_size=params.batch_size, preprocessor=preprocessor)

# X = next(test_loader) # Checks data loader
del train_datafiles, valid_datafiles, test_datafiles, train_datasets, valid_datasets, test_datasets

#%% FINE-TUNES VITMAE

# Loads pre-trained model
model = transformers.ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
model = model.to(device)

# Training
training_args = transformers.TrainingArguments(
    output_dir='../models',
    num_train_epochs=100,
    per_device_train_batch_size=params.batch_size,
    per_device_eval_batch_size=params.batch_size,
    learning_rate=1e-4,
    weight_decay=0.01,
    lr_scheduler_type='linear',
    warmup_steps=100,
    save_strategy='epoch',
    eval_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_runtime',
    logging_strategy='no'
)

trainer = ZarrTrainer(
    model=model,
    args=training_args,
    train_dataloader=train_loader,
    eval_dataloader=valid_loader
)

history = trainer.train()

# Clears GPU memory
empty_cache(device)

#%% CHECKS RECONSTRUCTIONS

def unprocess_images(images:torch.Tensor, preprocessor:nn.Module) -> torch.Tensor:
    means  = torch.Tensor(preprocessor.image_mean).view(3, 1, 1)
    stds   = torch.Tensor(preprocessor.image_std).view(3, 1, 1)
    images = (images * stds + means) * 255
    images = torch.clip(images, 0, 255).to(torch.uint8)
    return images

# Predicts images
images = next(iter(test_loader)).pixel_values
with torch.no_grad():
    outputs = model(images.to(device))

# Unprocesses outputs
masks = outputs.mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)
masks = model.unpatchify(masks).cpu().int()
preds = model.unpatchify(outputs.logits.cpu())

images  = unprocess_images(images, preprocessor)
inputs  = images * (1 - masks)
preds   = unprocess_images(preds, preprocessor) * masks
outputs = inputs + preds
figdata = torch.stack([images, inputs, preds, outputs], dim=1)
del images, outputs, masks, preds, inputs

# Plots images, inputs, predictions and outputs 
for i in torch.randperm(figdata.shape[0])[:10]:
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, image, title in zip(axs.ravel(), figdata[i], ['Image', 'Input', 'Prediction', 'Output']):
        ax.imshow(image.moveaxis(0, -1))
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
    plt.show()

#%%