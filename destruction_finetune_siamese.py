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
    batch_size=32,
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})

#%% INITIALISES DATA MODULE

class ZarrDataset(utils.data.Dataset):

    def __init__(self, images_zarr:str, labels_zarr:str) -> None:
        self.images = zarr.open(images_zarr, mode='r')
        self.labels = zarr.open(labels_zarr, mode='r')
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx:int) -> tuple:
        x = torch.from_numpy(self.images[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y

class Formatter:
    
    def __init__(self, processor, label_map:dict, image_size:int=224) -> None:
        self.processor  = processor
        self.label_map  = label_map
        self.image_size = image_size

    def __call__(self, X:torch.Tensor, Y:torch.Tensor):
        X = X.view(-1, 3, self.image_size, self.image_size)
        X = self.processor(X, return_tensors='pt')['pixel_values']
        X = X.view(-1, 2, 3, self.image_size, self.image_size)
        for k, v in self.label_map.items():
            Y = Y.squeeze(1) # Removes channel dimension
            Y = torch.where(Y == k, v, Y)
        return X, Y

class ZarrDataLoader:

    def __init__(self, datafiles:list, datasets:list, formatter, batch_size:int, shuffle:bool=True):
        self.datafiles    = datafiles
        self.datasets     = datasets
        self.formatter    = formatter
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
                x, y = dataset[start:end]
                X.append(x), 
                Y.append(y)
        X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
        X, Y = self.formatter(X, Y)
        self.batch_index += 1 # Updates batch index
        return X, Y

class ZarrDataModule(pl.LightningDataModule):
    
    def __init__(self, train_datafiles:list, val_datafiles:list, test_datafiles:list, formatter, batch_size:int, shuffle:bool=True) -> None:
        super().__init__()
        self.train_datafiles = train_datafiles
        self.val_datafiles = val_datafiles
        self.test_datafiles = test_datafiles
        self.formatter = formatter
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage:str=None):
        self.train_datasets = [ZarrDataset(**self.train_datafiles[city]) for city in params.cities]
        self.val_datasets   = [ZarrDataset(**self.val_datafiles[city]) for city in params.cities]
        self.test_datasets  = [ZarrDataset(**self.test_datafiles[city])  for city in params.cities]

    def train_dataloader(self):
        return ZarrDataLoader(datafiles=self.train_datafiles, datasets=self.train_datasets, formatter=self.formatter, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return ZarrDataLoader(datafiles=self.val_datafiles, datasets=self.val_datasets, formatter=self.formatter, batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self):
        return ZarrDataLoader(datafiles=self.test_datafiles, datasets=self.test_datasets, formatter=self.formatter, batch_size=self.batch_size, shuffle=self.shuffle)

def unprocess_image(image:torch.Tensor, processor) -> torch.Tensor:
    means = torch.tensor(processor.image_mean).view(3, 1, 1)
    stds  = torch.tensor(processor.image_std).view(3, 1, 1)
    image = image * stds + means
    image = (image * 255.0).clamp(0, 255).to(torch.uint8)
    return image

# Initialises datasets
train_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_train_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_train_balanced.zarr') for city in params.cities]))
val_datafiles   = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_val_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_val_balanced.zarr')   for city in params.cities]))
test_datafiles  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_test_balanced.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_test_balanced.zarr')  for city in params.cities]))
processor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
formatter = Formatter(processor=processor, label_map=params.label_map, image_size=processor.size['height'])
data_module = ZarrDataModule(train_datafiles=train_datafiles, val_datafiles=val_datafiles, test_datafiles=test_datafiles, formatter=formatter, batch_size=params.batch_size, shuffle=True)
del train_datafiles, val_datafiles, test_datafiles

''' Check data module
data_module.setup()
loader = iter(data_module.train_dataloader())
X, Y   = next(loader)
for i in np.random.choice(len(X), size=5, replace=False):
    x    = torch.stack([unprocess_image(x, processor) for x in X[i]])
    y    = F.interpolate(Y[i].view(1, 1, 14, 14), size=(224, 224), mode='nearest').squeeze(0).bool()
    x[1] = torchvision.utils.draw_segmentation_masks(x[1], y, alpha=0.25, colors=['red'])
    display_sequence(x, titles=['x0', 'x1'])
del loader, X, Y, x, y, i
'''

#%% INITIALISES MODEL MODULE

def contrastive_loss(distance:torch.Tensor, label:torch.Tensor, margin:float, reduction:str=None) -> torch.Tensor:
    loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    if reduction == 'mean':
        loss = loss.mean()
    return loss

class SiameseModel(nn.Module):
    
    def __init__(self, backbone:str='facebook/vit-mae-base') -> None:
        super().__init__()
        self.encoder   = transformers.ViTModel.from_pretrained(backbone) # If using VitMAE loader, provide bool_masked_pos
        self.model_dim = self.encoder.config.hidden_size
        self.patch_dim = self.encoder.config.image_size // self.encoder.config.patch_size
        self.project0  = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim//2),
            nn.GELU(),
            nn.Linear(self.model_dim//2, self.model_dim//2)
        )
        self.project1 = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim//2),
            nn.GELU(),
            nn.Linear(self.model_dim//2, self.model_dim//2)
        )
        self.output = nn.Linear(1, 1)

    def forward_branch(self, Xt:torch.Tensor) -> torch.Tensor:
        return self.encoder(Xt).last_hidden_state[:, 1:, :]

    def forward(self, X:torch.Tensor, Y:torch.Tensor=None) -> torch.Tensor:
        H0 = self.forward_branch(X[:,0])
        H1 = self.forward_branch(X[:,1])
        H0 = self.project0(H0)
        H1 = self.project1(H1)
        D  = (H0 - H1).norm(dim=-1) # L2 distance
        Yh = self.output(D.unsqueeze(-1)).squeeze(-1)
        D  = D.reshape(-1,  self.patch_dim, self.patch_dim)
        Yh = Yh.reshape(-1, self.patch_dim, self.patch_dim)
        return D, Yh

class SiameseModule(pl.LightningModule):
    
    def __init__(self, model:str, downscale:int, model_name:str, learning_rate:float=1e-4, weight_decay:float=0.05, weight_contrast:float=0.0, margin_contrast=1.0):
        
        super().__init__()
        self.save_hyperparameters()
        self.model           = model
        self.downscale       = downscale # 1=8m, 2=16m, 7=56m, 14=112m
        self.model_name      = model_name
        self.contrast_loss   = contrastive_loss
        self.sigmoid_loss    = torchvision.ops.sigmoid_focal_loss
        self.learning_rate   = learning_rate
        self.weight_decay    = weight_decay
        self.weight_contrast = weight_contrast
        self.margin_contrast = margin_contrast
        self.trainable       = None
        self.accuracy_metric = classification.BinaryAccuracy()
        self.auroc_metric    = classification.BinaryAUROC()

    def count_parameters(self):
        '''Counts the number of parameters in a model'''
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        nontrain  = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f'Trainable parameters: {trainable:,} | Non-trainable parameters: {nontrain:,}')

    def freeze_encoder(self):
        self.trainable = [param.requires_grad for param in self.model.encoder.parameters()]
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.count_parameters()

    def unfreeze_encoder(self):
        for param, status in zip(self.model.encoder.parameters(), self.trainable):
            param.requires_grad = status
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.count_parameters()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        return Y
    
    def training_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y  = batch
        D, Yh = self.model(X)
        mask  = torch.isnan(Y)
        if self.downscale > 1:
            Y    = F.max_pool2d(torch.nan_to_num(Y, nan=0.0), kernel_size=self.downscale, stride=self.downscale)
            mask = F.max_pool2d(mask.int(), kernel_size=self.downscale, stride=self.downscale)
            mask = mask.bool() & ~Y.bool()
            D    = F.avg_pool2d(D,  kernel_size=self.downscale, stride=self.downscale)
            Yh   = F.avg_pool2d(Yh, kernel_size=self.downscale, stride=self.downscale)
        loss_S = self.sigmoid_loss(Yh[~mask], Y[~mask], reduction='mean')
        loss_C = self.contrast_loss(D[~mask], Y[~mask], margin=self.margin_contrast, reduction='mean')
        loss   = loss_S + self.weight_contrast * loss_C
        self.log('train_loss', loss, prog_bar=True)
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs[~mask], Y[~mask])
        self.auroc_metric.update(probs[~mask], Y[~mask])
        self.log('train_acc',   self.accuracy_metric.compute(), on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.auroc_metric.compute(),    on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y  = batch
        D, Yh = self.model(X)
        mask  = torch.isnan(Y)
        if self.downscale > 1:
            Y    = F.max_pool2d(torch.nan_to_num(Y, nan=0.0), kernel_size=self.downscale, stride=self.downscale)
            mask = F.max_pool2d(mask.int(), kernel_size=self.downscale, stride=self.downscale)
            mask = mask.bool() & ~Y.bool()
            D    = F.avg_pool2d(D,  kernel_size=self.downscale, stride=self.downscale)
            Yh   = F.avg_pool2d(Yh, kernel_size=self.downscale, stride=self.downscale)
        loss_S = self.sigmoid_loss(Yh[~mask], Y[~mask], reduction='mean')
        loss_C = self.contrast_loss(D[~mask], Y[~mask], margin=self.margin_contrast, reduction='mean')
        loss   = loss_S + self.weight_contrast * loss_C
        self.log('val_loss', loss, prog_bar=True)
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs[~mask], Y[~mask])
        self.auroc_metric.update(probs[~mask], Y[~mask])
        self.log('val_acc',   self.accuracy_metric.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.auroc_metric.compute(),    on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y  = batch
        D, Yh = self.model(X)
        mask  = torch.isnan(Y)
        if self.downscale > 1:
            Y    = F.max_pool2d(torch.nan_to_num(Y, nan=0.0), kernel_size=self.downscale, stride=self.downscale)
            mask = F.max_pool2d(mask.int(), kernel_size=self.downscale, stride=self.downscale)
            mask = mask.bool() & ~Y.bool()
            D    = F.avg_pool2d(D,  kernel_size=self.downscale, stride=self.downscale)
            Yh   = F.avg_pool2d(Yh, kernel_size=self.downscale, stride=self.downscale)
        loss_S = self.sigmoid_loss(Yh[~mask], Y[~mask], reduction='mean')
        loss_C = self.contrast_loss(D[~mask], Y[~mask], margin=self.margin_contrast, reduction='mean')
        loss   = loss_S + self.weight_contrast * loss_C
        self.log('test_loss', loss, prog_bar=True)
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs[~mask], Y[~mask])
        self.auroc_metric.update(probs[~mask], Y[~mask])
        self.log('test_acc',   self.accuracy_metric.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_auroc', self.auroc_metric.compute(),    on_step=False, on_epoch=True, prog_bar=True)
        return loss

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
    downscale=7, # 1=8m, 2=16m, 7=56m, 14=112m  #! Should be set
    model_name='destruction_finetune_siamese', 
    learning_rate=1e-4,
    weight_decay=0.05,
    weight_contrast=0.1) #! Should be tuned

#%% TRAINS MODEL

# Initialises logger
logger = loggers.CSVLogger(
    save_dir=f'{paths.models}/logs', 
    name=model_module.model_name, 
    version=0
)

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

# Aligns output layer
model_module.freeze_encoder()

trainer = pl.Trainer(
    max_epochs=1,
    accelerator=device)

trainer.fit(
    model=model_module,
    datamodule=data_module)

# Fine-tunes full model
model_module.unfreeze_encoder()

trainer = pl.Trainer(
    max_epochs=100,
    accelerator=device,
    log_every_n_steps=1e3,
    logger=logger,
    callbacks=[model_checkpoint, early_stopping],
    profiler=profilers.SimpleProfiler()
)

trainer.fit(
    model=model_module, 
    datamodule=data_module,
    ckpt_path=model_checkpoint.last_model_path if model_checkpoint.last_model_path else None,
)

# Saves model
trainer.save_checkpoint(f'{paths.models}/{model_module.model_name}.ckpt')
empty_cache(device=device)
#%%
