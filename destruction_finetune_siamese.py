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
import csv
import json
import pandas as pd
import yaml
import os

from destruction_models import *
from destruction_utilities import *
from pytorch_lightning import callbacks, loggers, profilers
from torch import optim
from torchmetrics import classification
#%%

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

# --- MODIFIED ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument('--cities', nargs='+', type=str, default=['aleppo', 'moschun'], help='List of city names for training and default for evaluation.')
parser.add_argument('--run_name', type=str, default=None, help='Unique name/ID for this training run. If None for training, a timestamp will be generated.')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval_per_city'], help='Operation mode: train a new model or evaluate an existing one per city.')
parser.add_argument('--checkpoint_to_eval', type=str, default=None, help='Path to .ckpt file for evaluation.')
parser.add_argument('--eval_cities', nargs='+', type=str, default=None, help='Cities on which we want to evaluate the model.')

# hyperparameters
parser.add_argument('--max_epochs_align', type=int, default=1, help='Max epochs for the alignment (frozen encoder) training stage.')
parser.add_argument('--max_epochs_ft', type=int, default=100, help='Max epochs for the fine-tuning (unfrozen encoder) stage.')
parser.add_argument('--patience_ft', type=int, default=2, help='Early stopping patience for the fine-tuning stage.') # Increased default
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation.')
parser.add_argument('--weight_contrast', type=float, default=0.1, help='Weight for the contrastive loss component.')
parser.add_argument('--weight_decay', type=float, default=0.05, help='Penalizes large weights to prevent overfitting.')
parser.add_argument('--margin_contrast', type=float, default=1, help='Value that explains how strict the contrastive loss is.')
parser.add_argument('--backbone_model', type=str, default='checkpoint-9920', help='Name of the checkpoint of the pretrained encoder.')
# Add any other hyperparameters you want to control via CLI

args = parser.parse_args()

# Generate run_name if training and not provided
if args.mode == 'train' and args.run_name is None:
    args.run_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Set default for eval_cities if not provided
if args.eval_cities is None:
    args.eval_cities = args.cities # Default to the training cities

# Define your label_map (this is the one you want as default)
default_label_map = {0:0, 1:0, 2:0, 3:1, 255:torch.tensor(float('nan'))}
args.label_map = default_label_map # Now args.label_map exists

BACKBONE_PATH = f'{paths.models}/{args.backbone_model}'

# --- PRINT ALL ARGUMENTS ---
print(f"\n--- Script Starting with the following arguments: ---\n")
for arg_name, arg_value in vars(args).items():
    print(f"{arg_name}: {arg_value}")
print(f"\n-----------------------------------------------------\n")


#%% --- FUNCTION DEFINITIONS (Place these early in the script) ---

def update_experiment_overview_csv(overview_filepath: str, dict_list):
    """Appends a new run's summary to the overview CSV file."""
    fieldnames = np.concatenate([list(dict.keys()) for dict in dict_list])
    
    value_lists = []
    for value_list in [list(dict.values()) for dict in dict_list]:
        value_lists.append([str(val) for val in value_list]) 
    fieldvalues = np.concatenate(value_lists)  
      
    save_dict = {}
    for i in range(len(fieldvalues)):
        save_dict[fieldnames[i]] = fieldvalues[i]
      
    ordered_columns = [
    'checkpoint_to_eval',
    'epoch',
    'run_name',
    'mode',
    'training_start_actual_time',
    'evaluation_timestamp',
    'cities',
    'test_auroc_epoch',
    'test_acc_epoch',
    'test_loss',
    'val_auroc_epoch',
    'val_acc_epoch',
    'val_loss',
    'train_auroc_epoch',
    'train_acc_epoch',
    'per_city_test_aurocs',
    'average_test_auroc',
    'per_city_length_test',
    'backbone_model',
    'learning_rate',
    'batch_size',
    'weight_decay',
    'margin_contrast',
    'weight_contrast',
    'max_epochs_align',
    'max_epochs_ft',
    'patience_ft',
    'label_map'
    ]
      
    subset_dict = {}
    for col in ordered_columns:
        subset_dict[col] = save_dict[col]
    
    file_exists = os.path.isfile(overview_filepath)
    try:
        with open(overview_filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ordered_columns, extrasaction='ignore')
            if not file_exists or os.path.getsize(overview_filepath) == 0: # Check if file is empty too
                writer.writeheader()
            writer.writerow(subset_dict)
        print(f"Run summary appended to: {overview_filepath}")
    except IOError as e:
        print(f"Error writing to overview CSV {overview_filepath}: {e}")



def run_per_city_evaluation(
    checkpoint_to_eval_path: str,
    current_args: argparse.Namespace,
    global_paths_obj: object, # Explicitly pass the paths object
    base_model_name_str: str
):
    """
    Loads a trained model and evaluates it on the test set for each specified city
    """
    print(f"\n--- Starting Evaluation for Checkpoint: {checkpoint_to_eval_path.split('/')[-1]} ---\n")
    cities_for_evaluation = current_args.eval_cities 
    print(f"\nEvaluating on cities: {cities_for_evaluation}\n")

    try:
        model_to_eval = SiameseModule.load_from_checkpoint(
            checkpoint_path=checkpoint_to_eval_path,
            model=SiameseModel(backbone=BACKBONE_PATH), # Pass nn.Module
            # Pass other __init__ args for SiameseModule if they are not in hparams and needed for instantiation
            learning_rate=current_args.learning_rate, # Example: if it's part of __init__ and not just for optimizer
            weight_decay=current_args.weight_decay, # Example
            weight_contrast=current_args.weight_contrast, # Example
            model_name=base_model_name_str # Ensure model_name is consistent
        )
    except Exception as e:
        print(f"Error loading model from checkpoint '{checkpoint_to_eval_path}': {e}")
        return

    model_to_eval.eval()
    model_to_eval.to(device)

    eval_trainer = pl.Trainer(accelerator=device, logger=False)
    all_city_results = {}
    all_city_length = {}

    for city_name in cities_for_evaluation:
        print(f"--- Evaluating city: {city_name} ---")
        city_test_datafile_spec = {
            'images_zarr': f'{global_paths_obj.data}/{city_name}/zarr/images_prepost_test_balanced.zarr',
            'labels_zarr': f'{global_paths_obj.data}/{city_name}/zarr/labels_prepost_test_balanced.zarr'
        }
        if not os.path.exists(city_test_datafile_spec['images_zarr']) or \
           not os.path.exists(city_test_datafile_spec['labels_zarr']):
            print(f"Warning: Data files for city {city_name} not found. Skipping.")
            all_city_results[city_name] = {"error": "data not found"}
            continue

        try:
            city_dataset = ZarrDataset(**city_test_datafile_spec)
            if len(city_dataset) == 0:
                print(f"Warning: Dataset for city {city_name} is empty. Skipping.")
                all_city_results[city_name] = {"error": "empty dataset"}
                continue
            
            all_city_length[city_name] = len(city_dataset)
            
            city_test_loader = ZarrDataLoader(
                datafiles={city_name: city_test_datafile_spec},
                datasets=[city_dataset],
                label_map=current_args.label_map,
                batch_size=current_args.batch_size, # Use batch_size from args
                shuffle=False
            )
            city_metrics_list = eval_trainer.test(model=model_to_eval, dataloaders=city_test_loader, verbose=False)
            if city_metrics_list and isinstance(city_metrics_list, list) and len(city_metrics_list) > 0:
                all_city_results[city_name] = city_metrics_list[0]
                all_city_results[city_name]["test_set_size"] = len(city_dataset)
                print(f"Metrics for {city_name}: {city_metrics_list[0]}")
            else:
                print(f"Warning: No metrics returned from eval_trainer.test() for city {city_name}.")
                all_city_results[city_name] = {"error": "no metrics returned"}
        except Exception as e:
            print(f"Error during evaluation for city {city_name}: {e}")
            all_city_results[city_name] = {"error": str(e)}


    print(f"\n--- Overall Per-City Test Results (from checkpoint: {checkpoint_to_eval_path.split('/')[-1]}) ---")
    for city, metrics in all_city_results.items():
        print(f"City: {city}, Metrics: {metrics}, Size test set: {all_city_length[city]}")

    return all_city_results, all_city_length
    




#%% DEFINE DATA MODULE

class ZarrDataset(utils.data.Dataset):

    def __init__(self, images_zarr:str, labels_zarr:str):
        self.images = zarr.open(images_zarr, mode='r')
        self.labels = zarr.open(labels_zarr, mode='r')
        self.length = len(self.images)
        self.processor = transformers.ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
    
    def __len__(self):
        return self.length
            
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
        self.train_datasets = [ZarrDataset(**self.train_datafiles[city]) for city in args.cities]
        self.valid_datasets = [ZarrDataset(**self.valid_datafiles[city]) for city in args.cities]
        self.test_datasets  = [ZarrDataset(**self.test_datafiles[city])  for city in args.cities]

    def train_dataloader(self):
        return ZarrDataLoader(datafiles=self.train_datafiles, datasets=self.train_datasets, label_map=self.label_map, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return ZarrDataLoader(datafiles=self.valid_datafiles, datasets=self.valid_datasets, label_map=self.label_map, batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self):
        return ZarrDataLoader(datafiles=self.test_datafiles, datasets=self.test_datasets,   label_map=self.label_map, batch_size=self.batch_size, shuffle=self.shuffle)


''' Check data module
data_module.setup()
X, Y = next(data_module.train_dataloader())
for idx in np.random.choice(range(len(X)), size=5, replace=False):
    display_sequence(X[idx], [0] + [int(Y[idx])])
del X, Y, idx
'''

#%% DEFINE MODEL MODULE


# New contrastive loss for cosine similarity
def contrastive_loss(similarity: torch.Tensor, label: torch.Tensor, margin: float = 1) -> torch.Tensor:
    # For similar pairs (label=0), push similarity towards 1 (or margin_similar)
    # We want (margin_similar - similarity)^2 if similarity < margin_similar, else 0
    # Or simply (1-similarity)^2
    loss_similar = (1 - label) * torch.pow((1.0 - similarity)/2, 2)

    # For dissimilar pairs (label=1), push similarity below margin_dissimilar
    loss_dissimilar = label * torch.pow((similarity + 1.0)/2, 2)

    return (loss_similar + loss_dissimilar).mean()


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
    
    def __init__(self, model:str, model_name:str, learning_rate:float=1e-4, weight_decay:float=0.05, weight_contrast:float=0.0, margin_contrast=1.0):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.model_name      = model_name
        self.contrast_loss   = contrastive_loss
        self.sigmoid_loss    = torchvision.ops.sigmoid_focal_loss # nn.crossentropy
        self.learning_rate   = learning_rate
        self.weight_decay    = weight_decay
        self.weight_contrast  = weight_contrast
        self.margin_contrast = margin_contrast
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
        train_loss = loss_S + self.weight_contrast * loss_C
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
        val_loss = loss_S + self.weight_contrast * loss_C
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
        test_loss = loss_S + self.weight_contrast * loss_C
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


#%% INITIALISE DATA AND MODEL (Needed for both modes) ---

# Initialises datasets
train_datafiles = dict(zip(args.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_train_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_train_balanced.zarr') for city in args.cities]))
valid_datafiles = dict(zip(args.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_valid_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_valid_balanced.zarr') for city in args.cities]))
test_datafiles  = dict(zip(args.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_test_balanced.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_test_balanced.zarr')  for city in args.cities]))
data_module = ZarrDataModule(train_datafiles=train_datafiles, 
                             valid_datafiles=valid_datafiles, 
                             test_datafiles=test_datafiles, 
                             batch_size=args.batch_size, 
                             label_map=args.label_map, 
                             shuffle=True)
del train_datafiles, valid_datafiles, test_datafiles

# Initialises model
if not os.path.exists(BACKBONE_PATH):
    print(f"Warning: Backbone path {BACKBONE_PATH} does not exist. Ensure it's correct.")

siamese_nn_model = SiameseModel(backbone=BACKBONE_PATH)
model_module = SiameseModule(
    model= siamese_nn_model, 
    model_name='destruction_finetune_siamese', 
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    weight_contrast=args.weight_contrast, #! Should be tuned
    margin_contrast=args.margin_contrast) 


    
#%% MAIN SCRIPT LOGIC DISPATCHER
    # --- MAIN SCRIPT LOGIC ---
if args.mode == 'train':
            
    print(f"\n--- Starting Training Mode for Run ID: {args.run_name} ---\n")
    print(f"Training with cities: {args.cities}")
    training_start_actual_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Log actual start

    # --- ALIGNMENT STAGE ---
    print("\n--- Stage 1: Alignment Training (Encoder Frozen) ---\n")
    model_module.freeze_encoder()
    if 'count_parameters' in globals() and callable(globals()['count_parameters']): count_parameters(model_module)

    alignment_trainer = pl.Trainer(
        max_epochs=args.max_epochs_align,
        accelerator=device,
        logger=False,
        enable_checkpointing=False
    )
    alignment_trainer.fit(model=model_module, datamodule=data_module)
    print("--- Alignment Stage Complete ---")

    # --- FINE-TUNING STAGE ---
    print("\n--- Stage 2: Fine-Tuning (Encoder Unfrozen) ---\n")
    model_module.unfreeze_encoder()
    if 'count_parameters' in globals() and callable(globals()['count_parameters']): count_parameters(model_module)

    fine_tune_logger = loggers.CSVLogger(
        save_dir=f'{paths.models}/logs',
        name=model_module.model_name,
        version=args.run_name
    )

    # -----------------------------------------------------------------------------------------------
    # Logging of hyperparameters in hparams.yaml 
    
    # Log all parameters in args
    hparams_for_logging = vars(args).copy()
    
    # Properly save the label map
    if 'label_map' in hparams_for_logging and isinstance(hparams_for_logging['label_map'], dict):
        serializable_label_map = {}
        for k, v_obj in hparams_for_logging['label_map'].items():
            if isinstance(v_obj, torch.Tensor):
                if v_obj.isnan().any():
                    serializable_label_map[k] = "NaN" # Represent as string "NaN"
                elif v_obj.numel() == 1:
                    serializable_label_map[k] = v_obj.item()
                else:
                    serializable_label_map[k] = str(v_obj.tolist())
            else:
                serializable_label_map[k] = v_obj
        hparams_for_logging['label_map'] = serializable_label_map
    
    # Add any other important parameters not in args, e.g. when the training was started
    hparams_for_logging['training_start_actual_time'] = training_start_actual_time
    
    # Save the hyperparameters 
    if hasattr(fine_tune_logger, 'log_hyperparams'):
        fine_tune_logger.log_hyperparams(hparams_for_logging)
    # Fallback for older versions of PyTorch Lightning
    elif hasattr(model_module, 'hparams'): # Fallback
            model_module.hparams.update(hparams_for_logging)

    # -----------------------------------------------------------------------------------------------
    # Logging of checkpoints 
    
    fine_tune_checkpoint_dir = os.path.join(fine_tune_logger.log_dir, 'checkpoints')
    os.makedirs(fine_tune_checkpoint_dir, exist_ok=True)

    fine_tune_model_checkpoint = callbacks.ModelCheckpoint(
        dirpath=fine_tune_checkpoint_dir,
        filename=f"{model_module.model_name}-FT-{{epoch:02d}}-{{val_auroc_epoch:.4f}}", # Make sure val_auroc_epoch is logged
        monitor='val_auroc_epoch',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    
    
    # -----------------------------------------------------------------------------------------------
    # Defining the early stopping
     
    fine_tune_early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=args.patience_ft, verbose=True
    )
    
    
    # -----------------------------------------------------------------------------------------------
    # Start training
     
    trainer_callbacks_list = [fine_tune_model_checkpoint, fine_tune_early_stopping]

    fine_tune_trainer = pl.Trainer(
        max_epochs=args.max_epochs_ft,
        accelerator=device,
        logger=fine_tune_logger,
        callbacks=trainer_callbacks_list,
        profiler=profilers.SimpleProfiler() if 'profilers' in globals() and callable(globals()['profilers'].SimpleProfiler) else None
    )
    
    initial_ft_ckpt_path = None # For a new run_name, usually start fresh fine-tuning
    print(f"\n Starting fine-tuning. Initial FT checkpoint: {initial_ft_ckpt_path if initial_ft_ckpt_path else 'None (after alignment)'} \n")
    fine_tune_trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=initial_ft_ckpt_path
    )
    print(f"\n--- Fine-Tuning Stage Complete for Run ID: {args.run_name} ---\n")

    # Identifying if the checkpoint of the best/last run was created
    best_ft_checkpoint_for_eval = None
    last_ckpt_in_run_dir = os.path.join(fine_tune_checkpoint_dir, "last.ckpt")

    if hasattr(fine_tune_model_checkpoint, 'best_model_path') and \
        fine_tune_model_checkpoint.best_model_path and \
        os.path.exists(fine_tune_model_checkpoint.best_model_path):
        best_ft_checkpoint_for_eval = fine_tune_model_checkpoint.best_model_path
        print(f"Best fine-tuned checkpoint identified: {best_ft_checkpoint_for_eval}")
    elif os.path.exists(last_ckpt_in_run_dir):
        best_ft_checkpoint_for_eval = last_ckpt_in_run_dir
        print(f"Using last fine-tuned checkpoint: {best_ft_checkpoint_for_eval}")
    else:
        print("Error: No fine-tuned checkpoint (best or last) was found after training.")

    if 'empty_cache' in globals() and callable(globals()['empty_cache']): empty_cache(device=device)


    if best_ft_checkpoint_for_eval:
        # -----------------------------------------------------------------------------------------------
        # Evaluate the model with the test set of the entire data
        print(f"\n--- Test Set Evaluation Results for checkpoint: {best_ft_checkpoint_for_eval.split('/')[-1]} ---\n")
        test_results = fine_tune_trainer.test(model=model_module,
                                              datamodule=data_module,
                                                   ckpt_path=best_ft_checkpoint_for_eval)

        if test_results: # trainer.test() returns a list of dictionaries
            for result_dict in test_results:
                for key, value in result_dict.items():
                    print(f"{key}: {value:.4f}") # Print test metrics
        else:
            print("No test results returned. Ensure metrics are logged in test_step of your LightningModule.")


        # -----------------------------------------------------------------------------------------------
        # Per city evaluation if a checkpoint of the model was saved

        print(f"\n--- Automatically proceeding to per-city evaluation for checkpoint: {best_ft_checkpoint_for_eval.split('/')[-1]} ---\n")
        
        # Prepare args for evaluation function by adding the best checkpoint of the model
        args.checkpoint_to_eval = best_ft_checkpoint_for_eval
        
        all_city_results, all_city_length = run_per_city_evaluation(
            checkpoint_to_eval_path=args.checkpoint_to_eval,
            current_args=args,
            global_paths_obj=paths,   # global paths object
            base_model_name_str=model_module.model_name
        )
    else:
        print("\nSkipping automatic per-city evaluation as no valid checkpoint was identified.")

# -----------------------------------------------------------------------------------------------
# Directly run per city evaluation
elif args.mode == 'eval_per_city':
    if not args.checkpoint_to_eval or not os.path.exists(args.checkpoint_to_eval):
        raise ValueError("Must provide a valid --checkpoint_to_eval path for 'eval_per_city' mode when run manually.")
    
    all_city_results, all_city_length = run_per_city_evaluation(
        checkpoint_to_eval_path=args.checkpoint_to_eval,
        current_args=args, # The command-line args passed for evaluation
        global_paths_obj=paths,
        base_model_name_str=model_module.model_name # Assuming model_module is instantiated globally
    )
else:
    print(f"Unknown mode: {args.mode}. Choose 'train' or 'eval_per_city'.")


# -----------------------------------------------------------------------------------------------
# Gather information for the overview CSV
#%%
# --- Gather info for overview CSV ---
'''run_id_for_overview = "N/A"
model_name_overview = "N/A" #model_module.model_name
training_cities_overview_str = "N/A"
key_hyperparams_overview_json = "{}"
best_val_metric_value_overview = "N/A"
epoch_of_best_val_overview = "N/A"
train_auroc_at_best_val_overview = "N/A"
train_loss_at_best_val_overview = "N/A"
training_start_ts_overview = "N/A" #datetime.datetime.strptime(args.run_name, '%Y%m%d-%H%M%S')
training_best_checkpoint_ts = "N/A"'''
#if os.path.exists(args.checkpoint_to_eval):
#    training_best_checkpoint_ts = datetime.datetime.fromtimestamp(os.path.getmtime(args.checkpoint_to_eval)).strftime('%Y-%m-%d %H:%M:%S')


# Open hyperparameter and metrics files
path_parts = args.checkpoint_to_eval.split(os.sep)
try:
    checkpoints_idx = path_parts.index("checkpoints")
    run_id_for_overview = path_parts[checkpoints_idx - 1]
    model_name_overview = path_parts[checkpoints_idx - 2]
    run_log_dir_for_csv = os.path.join(f'{paths.models}/logs', model_name_overview, run_id_for_overview)
except (ValueError, IndexError):
    print(f"Warning: Could not parse run_id/model_name from checkpoint path: {args.checkpoint_to_eval}")
    run_log_dir_for_csv = os.path.dirname(os.path.dirname(args.checkpoint_to_eval)) # Fallback

hparams_file = os.path.join(run_log_dir_for_csv, 'hparams.yaml')
metrics_file = os.path.join(run_log_dir_for_csv, 'metrics.csv')

#%%
# This is important if eval is run manually and needs to reference the original training cities
if os.path.exists(hparams_file):
    with open(hparams_file, 'r') as f:
        hparams = yaml.unsafe_load(f)

#%%
if os.path.exists(metrics_file):
    try:
        metrics_df = pd.read_csv(metrics_file)

        # Extract the validation metrics for the best epoch
        best_validation_df = metrics_df.iloc[[metrics_df["val_auroc_epoch"].idxmax()]]
        best_validation_df = best_validation_df[["epoch", "val_acc_epoch", "val_auroc_epoch", "val_loss"]].reset_index(drop=True)
        
        # Extract the train metrics for the best epoch
        best_train_df = metrics_df.loc[(metrics_df["epoch"] == best_validation_df["epoch"].values[0]) & (~metrics_df["train_auroc_epoch"].isna()), ["train_acc_epoch", "train_auroc_epoch"]].reset_index(drop=True)
        
        # Extract the test metrics for the best epoch
        test_df = metrics_df.loc[~metrics_df["test_acc_epoch"].isna(), ["test_acc_epoch", "test_auroc_epoch", "test_loss"]].reset_index(drop=True)
        
        # Join extracted metrics        
        extracted_metrics_df = pd.concat([best_validation_df, best_train_df, test_df], axis=1)
        extracted_metrics_df["epoch"] = extracted_metrics_df["epoch"].astype(int)
        
        extracted_metrics_dict = {}
        for col in extracted_metrics_df.columns:
            extracted_metrics_dict[col] = extracted_metrics_df[col].values[0]
        
    except Exception as e:
        print(f"Warning: Could not read/select fine-tuning metrics from {metrics_file}: {e}")
        
#%%        
per_city_test_aurocs_dict = {
    city: results.get('test_auroc_epoch', results.get('test_auroc', "N/A")) # Ensure key matches test_step log
    for city, results in all_city_results.items() if "error" not in results
}
per_city_test_aurocs_json = json.dumps(per_city_test_aurocs_dict)
auroc_values = [val for val in per_city_test_aurocs_dict.values() if isinstance(val, (float, int))]
average_test_auroc_overview = sum(auroc_values) / len(auroc_values) if auroc_values else "N/A"

per_city_eval_data = {
    'checkpoint_to_eval': args.checkpoint_to_eval,
    'evaluation_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'per_city_test_aurocs': per_city_test_aurocs_json,
    'per_city_length_test': str(all_city_length),
    'average_test_auroc': f"{average_test_auroc_overview:.4f}" if isinstance(average_test_auroc_overview, float) else average_test_auroc_overview,
}

dicts_list = [hparams, extracted_metrics_dict, per_city_eval_data]

overview_csv_filepath = os.path.join(f'{paths.models}/logs', "experiment_overview.csv")
update_experiment_overview_csv(overview_csv_filepath, dicts_list)

if 'empty_cache' in globals() and callable(globals()['empty_cache']): empty_cache(device)

# %%
