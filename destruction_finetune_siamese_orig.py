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
import datetime

from destruction_models import *
from destruction_utilities import *
from pytorch_lightning import callbacks, loggers, profilers
from torch import optim
from torchmetrics import classification

seed = 42 # Or any integer
np.random.seed(seed)
torch.manual_seed(seed)

# Utilities
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--cities', nargs='+', type=str, default=['aleppo', 'moschun'], help='List of city names')
parser.add_argument('--run_name', type=str, default=None, help='Unique name/ID for this training run')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval_per_city'], help='Operation mode: train a new model or evaluate an existing one per city.')
parser.add_argument('--checkpoint_to_eval', type=str, default=None, help='Path to .ckpt file for evaluation, or a run_name to derive best checkpoint path (requires consistent saving).')

# After parsing args:
args = parser.parse_args()
if args.mode == 'train' and args.run_name is None:
    args.run_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') # Default run_name if training

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
        self.processor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
    
    def __len__(self):
        return self.length
    
    '''
    # Getitem CNN
    def __getitem__(self, idx):
        X = torch.from_numpy(self.images[idx]).float()
        Y = torch.from_numpy(self.labels[idx]).float()
        X = X / 255.0
        return X, Y
    '''    
        
    # Getitem Transformer
    def __getitem__(self, idx):
        X = self.images[idx]
        X = torch.stack([self.processor(x, return_tensors='pt')['pixel_values'] for x in X])
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
        if slice_sizes.sum() <= 0:
            raise ValueError("Sum of slice_sizes is zero or negative; check your dataset lengths.")
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
        X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
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

#def contrastive_loss(distance:torch.Tensor, label:torch.Tensor, margin:float) -> torch.Tensor:
#    loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
#    return loss.mean()

# New contrastive loss for cosine similarity
def contrastive_loss(similarity: torch.Tensor, label: torch.Tensor, margin: float = 1) -> torch.Tensor:
    # For similar pairs (label=0), push similarity towards 1 (or margin_similar)
    # We want (margin_similar - similarity)^2 if similarity < margin_similar, else 0
    # Or simply (1-similarity)^2
    loss_similar = (1 - label) * torch.pow((1.0 - similarity)/2, 2)

    # For dissimilar pairs (label=1), push similarity below margin_dissimilar
    loss_dissimilar = label * torch.pow((similarity + 1.0)/2, 2)

    return (loss_similar + loss_dissimilar).mean()
### Old Model - CNN based
'''
class DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        # He (Kaiming) normal initialization
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_filters: int = 32, dropout: float = 0.1, 
                 n_convs: int = 1, n_blocks: int = 3):
        """
        Creates a CNN encoder.
        For each block i (i from 0 to n_blocks-1), we set:
          block_filters = (base_filters * 2) // (i+1)
        Each block runs n_convs convolution operations, each doing:
            Conv2d (3x3, padding=1, no bias) -> ReLU -> BatchNorm2d -> MaxPool2d(2) -> Dropout2d.
        """
        super().__init__()
        layers_list = []
        current_channels = in_channels

        for i in range(n_blocks):
            block_filters = (base_filters * 2) // (i + 1)
            # Repeat the convolutional operation n_convs times in this block:
            for j in range(n_convs):
                conv = nn.Conv2d(current_channels, block_filters, kernel_size=3, padding=1, bias=False)
                layers_list.append(conv)
                layers_list.append(nn.ReLU())
                layers_list.append(nn.BatchNorm2d(block_filters))
                layers_list.append(nn.MaxPool2d(kernel_size=2))
                layers_list.append(nn.Dropout2d(dropout))
                # Update the number of channels for the next layer in the same block
                current_channels = block_filters

        layers_list.append(nn.Flatten())
        self.model = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SiameseCNNModel(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 img_size: int = 224, 
                 base_filters: int = 32, 
                 dropout: float = 0.1, 
                 n_convs: int = 1, 
                 n_blocks: int = 3, 
                 dense_units: int = 128):
        """
        This model accepts a tensor X of shape (B, 2, C, H, W) and processes each image 
        in the pair with a shared CNN encoder. The resulting flattened features are concatenated,
        then passed through three dense blocks and a final linear layer to output a single logit.
        """
        super().__init__()
        self.encoder = CNNEncoder(in_channels, base_filters, dropout, n_convs, n_blocks)
        # Determine the encoder’s flattened feature dimension by a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            feat_dim = self.encoder(dummy).shape[1]
        # After processing a pair, the concatenated feature vector has dimension 2*feat_dim.
        self.dense_block1 = DenseBlock(2 * feat_dim, dense_units, dropout)
        self.dense_block2 = DenseBlock(dense_units, dense_units, dropout)
        self.dense_block3 = DenseBlock(dense_units, dense_units, dropout)
        self.output = nn.Linear(dense_units, 1)
        # (Optionally, initialize self.output if needed)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity='sigmoid')

    def forward_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Expects input X of shape (B, 2, C, H, W). The two images are processed separately,
        their encoded (flattened) features concatenated, and then run through dense layers.
        """
        # Split the two images
        x1 = X[:, 0]  # shape: (B, C, H, W)
        x2 = X[:, 1]  # shape: (B, C, H, W)
        feat1 = self.forward_branch(x1)
        feat2 = self.forward_branch(x2)
        # Concatenate along the feature dimension
        concat = torch.cat([feat1, feat2], dim=1)
        D = F.pairwise_distance(feat1, feat2, keepdim=True)
        out = self.dense_block1(concat)
        out = self.dense_block2(out)
        out = self.dense_block3(out)
        out = self.output(out)
        # We leave the final activation (sigmoid) out because the loss (sigmoid focal loss) expects logits.

        return D, out


class SiameseCNNModule(pl.LightningModule):
    def __init__(self, model: nn.Module, model_name: str, learning_rate: float = 1e-4, weight_decay: float = 0.05, weigh_contrast:float=0.0):
        """
        This Lightning module wraps the SiameseCNNModel.
        It uses torchvision’s sigmoid focal loss (which expects logits) and computes 
        binary accuracy and AUROC as metrics.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.model_name = model_name
        self.contrast_loss   = contrastive_loss
        self.sigmoid_loss    = torchvision.ops.sigmoid_focal_loss
        self.weigh_contrast  = weigh_contrast
        self.margin_contrast = 1.0
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.accuracy_metric = classification.BinaryAccuracy()
        self.auroc_metric = classification.BinaryAUROC()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        X, Y = batch  # X: (B,2,C,H,W), Y: (B,1) (or (B,))
        D, Yh = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        loss = loss_S + self.weigh_contrast * loss_C
        self.log('train_loss', loss, prog_bar=True)
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('train_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.auroc_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        X, Y = batch
        D, Yh = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        loss = loss_S + self.weigh_contrast * loss_C
        self.log('val_loss', loss, prog_bar=True)
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('val_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.auroc_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        X, Y = batch
        D, Yh = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        loss = loss_S + self.weigh_contrast * loss_C
        self.log('test_loss', loss, prog_bar=True)
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('test_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_auroc', self.auroc_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def on_train_epoch_end(self) -> None:
        self.accuracy_metric.reset()
        self.auroc_metric.reset()

    def on_validation_epoch_end(self) -> None:
        self.accuracy_metric.reset()
        self.auroc_metric.reset()

    def on_test_epoch_end(self) -> None:
        self.accuracy_metric.reset()
        self.auroc_metric.reset()

cnn_model = SiameseCNNModel(in_channels=3, img_size=128, base_filters=64, dropout=0.1, n_convs=1, n_blocks=3, dense_units=64)

# Wrap it in the Lightning module.
model_module = SiameseCNNModule(
    model=cnn_model,
    model_name='destruction_cnn_siamese',
    #learning_rate=1e-4,
    learning_rate=3e-5,
    weight_decay=0.05
)

'''

### New Model - Transformer based
class SiameseModel(nn.Module):
    
    def __init__(self, backbone:str):
        super().__init__()
        self.encoder   = transformers.ViTMAEModel.from_pretrained(backbone)
        self.model_dim = self.encoder.config.hidden_size
        self.project0  = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim//2),
            nn.GELU(),
            nn.Linear(self.model_dim//2, self.model_dim//2)
        )
        self.project1  = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim//2),
            nn.GELU(),
            nn.Linear(self.model_dim//2, self.model_dim//2)
        )
        self.output = nn.Linear(1, 1)

    def forward_branch(self, Xt:torch.Tensor) -> torch.Tensor:
        Ht = self.encoder(Xt)
        Ht = Ht.last_hidden_state[:, 0, ...]
        return Ht

    def forward(self, X:torch.Tensor, Y:torch.Tensor=None) -> torch.Tensor:
        H0 = self.forward_branch(X[:,0])
        H0 = self.project0(H0)
        H1 = self.forward_branch(X[:,1])
        H1 = self.project1(H1)
        D  = F.cosine_similarity(H0, H1, dim=1, eps=1e-8).unsqueeze(1)
        Yh = self.output(D)
        return D, Yh

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
        X, Y   = batch
        D, Yh  = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        train_loss = loss_S + self.weigh_contrast * loss_C
        self.log('train_loss', train_loss, prog_bar=True)
        
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('train_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.auroc_metric.compute(),  on_step=True, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y   = batch
        D, Yh  = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        val_loss = loss_S + self.weigh_contrast * loss_C
        self.log('val_loss', val_loss, prog_bar=True)
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('val_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.auroc_metric.compute(),  on_step=True, on_epoch=True, prog_bar=True)
        return val_loss
    
    def test_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        X, Y   = batch
        D, Yh  = self.model(X)
        loss_S = self.sigmoid_loss(Yh, Y, reduction='mean')
        loss_C = self.contrast_loss(D, Y, margin=self.margin_contrast)
        test_loss = loss_S + self.weigh_contrast * loss_C
        self.log('test_loss', test_loss, prog_bar=True)
        # Metrics
        probs = torch.sigmoid(Yh)
        self.accuracy_metric.update(probs, Y)
        self.auroc_metric.update(probs, Y)
        self.log('test_acc', self.accuracy_metric.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_auroc', self.auroc_metric.compute(),  on_step=True, on_epoch=True, prog_bar=True)
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
    weigh_contrast=0.1) #! Should be tuned

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

# Aligns output layer (1 unit)
model_module.freeze_encoder()
count_parameters(model_module)

#trainer = pl.Trainer(max_epochs=1, accelerator=device)
alignment_trainer = pl.Trainer(
    max_epochs=args.max_epochs_align, # e.g., 1-5 epochs
    accelerator=device,
    logger=False,  # No separate logger for this transient stage
    enable_checkpointing=False # No checkpoints for this stage
)
alignment_trainer.fit(model=model_module, datamodule=data_module)


# Fine-tunes full model
model_module.unfreeze_encoder()
count_parameters(model_module)

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
