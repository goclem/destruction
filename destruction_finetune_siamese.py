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
parser = argparse.ArgumentParser()
parser.add_argument('--cities', nargs='+', type=str, default=['aleppo', 'moschun'], help='List of city names')
args = parser.parse_args()

params = argparse.Namespace(
    cities=args.cities,
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
        self.train_datasets = [ZarrDataset(**self.train_datafiles[city]) for city in params.cities]
        self.valid_datasets = [ZarrDataset(**self.valid_datafiles[city]) for city in params.cities]
        self.test_datasets  = [ZarrDataset(**self.test_datafiles[city])  for city in params.cities]

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

''' Notes
- We average the last hidden state, it's supposed to be more robust than selecting the first token
- The pairwise distance function actually sums the tensors (64 x 768) (64 x 768) > (64 x 1)
- It's good practice to have a single unit (output) turning the distance into a logit score
- We don't apply sigmoid because sigmoid_focal_loss requires logit scores
- I implemented contrastive loss as a function rather than a class so it works in the same way as sigmoid_focal_loss
'''

def contrastive_loss(distance:torch.Tensor, label:torch.Tensor, margin:float) -> torch.Tensor:
    loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

class SiameseModel(nn.Module):
    
    def __init__(self, backbone:str):
        super().__init__()
        self.processor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
        self.encoder   = transformers.ViTMAEModel.from_pretrained(backbone)
        self.output    = nn.Linear(1, 1)

    def forward_branch(self, Xt:torch.Tensor) -> torch.Tensor:
        Ht = self.processor(Xt, return_tensors='pt', do_resize=True)
        Ht = self.encoder(**Ht.to(device))
        Ht = Ht.last_hidden_state.mean(dim=1) # Alternative H = H.last_hidden_state[:,0,:] 
        return Ht

    def forward(self, X:torch.Tensor, Y:torch.Tensor=None) -> torch.Tensor:
        H0 = self.forward_branch(X[:,0])
        H1 = self.forward_branch(X[:,1])
        D = F.pairwise_distance(H0, H1, keepdim=True)
        Y = self.output(D)
        return D, Y

class SiameseModule(pl.LightningModule):
    
    def __init__(self, model:str, model_name:str, learning_rate:float=1e-4, weight_decay:float=0.05, weigh_contrast:float=0.0, margin_contrast=1.0):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.model_name      = model_name
        self.contrast_loss   = contrastive_loss
        self.sigmoid_loss    = torchvision.ops.sigmoid_focal_loss
        self.learning_rate   = learning_rate
        self.weight_decay    = weight_decay
        self.weigh_contrast  = weigh_contrast
        self.margin_contrast = 1.0
        self.trainable       = None
        self.accuracy_metric = classification.BinaryAccuracy()
        self.auroc_metric    = classification.BinaryAUROC()

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
        X, Y  = batch
        D, Yh = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        train_loss = loss_S + self.weigh_contrast * loss_C
        self.log('train_loss', train_loss, prog_bar=True)
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('train_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.auroc_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y  = batch
        D, Yh = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        val_loss = loss_S + self.weigh_contrast * loss_C
        self.log('val_loss', val_loss, prog_bar=True)
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('val_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.auroc_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        return val_loss
    
    def test_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y  = batch
        D, Yh = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        test_loss = loss_S + self.weigh_contrast * loss_C
        self.log('test_loss', test_loss, prog_bar=True)
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('test_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_auroc', self.auroc_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        return test_loss

    def configure_optimizers(self) -> dict:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {'optimizer':optimizer}
    
    def on_train_epoch_end(self) -> None:
        self.accuracy_metric.reset()
        self.auroc_metric.reset()
    
    def on_validation_epoch_end(self) -> None:
        self.accuracy_metric.reset()
        self.auroc_metric.reset()
    
    def on_test_epoch_end(self) -> None:
        self.accuracy_metric.reset()
        self.auroc_metric.reset()

# Initialises model module
model_module = SiameseModule(
    model=SiameseModel(backbone=f'{paths.models}/checkpoint-9920'), 
    model_name='destruction_finetune_siamese', 
    learning_rate=1e-4, 
    weight_decay=0.05,
    weigh_contrast=0.0)

#%% TRAINS MODEL

# Initialises logger
logger = loggers.CSVLogger(
    save_dir=f'{paths.models}/logs', 
    name=model_module.model_name, 
    version=0
)

# Define a callback that will unfreeze the encoder when the new trainer begins fit.
class UnfreezeCallback(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        # Check if any encoder parameter is still frozen
        if any(not param.requires_grad for param in pl_module.model.encoder.parameters()):
            pl_module.unfreeze_encoder()
            print("Encoder unfrozen in on_fit_start callback.")
            
# Initialises callbacks
model_checkpoint = callbacks.ModelCheckpoint(
    dirpath=paths.models,
    filename=f'{model_module.model_name}-{{epoch:02d}}-{{step:05d}}',
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

# Initialises trainer
trainer1 = pl.Trainer(
    max_epochs=100,
    accelerator=device,
    log_every_n_steps=1e3,
    logger=logger,
    callbacks=[model_checkpoint, early_stopping],
    profiler=profilers.SimpleProfiler()
)

# Optimisation step 1: Aligns output layer
model_module.freeze_encoder()
trainer1.fit(model=model_module, datamodule=data_module)
trainer1.save_checkpoint(f'{paths.models}/{model_module.model_name}_stage1.ckpt')
empty_cache(device=device)

"""trainer.fit(
    model=model_module, 
    datamodule=data_module,
    ckpt_path=model_checkpoint.last_model_path if model_checkpoint.last_model_path else None,
)
trainer.save_checkpoint(f'{paths.models}/{model_module.model_name}_stage1.ckpt')
empty_cache(device=device)
"""

# Optimisation step 2: Fine-tunes full model
# Manually load the weights from stage 1 into the current model
checkpoint = torch.load(f'{paths.models}/{model_module.model_name}_stage1.ckpt', map_location=device)
model_module.load_state_dict(checkpoint['state_dict'])

# Instantiate a new trainer for stage 2, adding the UnfreezeCallback.
trainer2 = pl.Trainer(
    max_epochs=100,
    accelerator=device,
    log_every_n_steps=1000,
    logger=logger,
    callbacks=[model_checkpoint, early_stopping, UnfreezeCallback()],
    profiler=profilers.SimpleProfiler()
)

# Run training from the current state (with ckpt_path=None so no checkpoint reload happens)
trainer2.fit(model=model_module, datamodule=data_module, ckpt_path=None)
trainer2.save_checkpoint(f'{paths.models}/{model_module.model_name}_stage2.ckpt')
empty_cache(device=device)

"""
model_module.unfreeze_encoder()
trainer.fit(
    model=model_module, 
    datamodule=data_module,
    ckpt_path=model_checkpoint.last_model_path if model_checkpoint.last_model_path else None,
)

trainer.save_checkpoint(f'{paths.models}/{model_module.model_name}_stage2.ckpt')
empty_cache(device=device)
"""
#%%
