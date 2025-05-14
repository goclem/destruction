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
parser.add_argument('--training_cities_list', nargs='+', type=str, default=None, help='For eval_per_city mode: comma-separated list of cities used during training of the checkpoint (for overview logging).')
parser.add_argument('--max_epochs_align', type=int, default=1, help='Max epochs for the alignment (frozen encoder) training stage.')
parser.add_argument('--max_epochs_ft', type=int, default=100, help='Max epochs for the fine-tuning (unfrozen encoder) stage.')
parser.add_argument('--patience_ft', type=int, default=5, help='Early stopping patience for the fine-tuning stage.') # Increased default
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation.')
parser.add_argument('--weigh_contrast', type=float, default=0.1, help='Weight for the contrastive loss component.')
# Add any other hyperparameters you want to control via CLI

args = parser.parse_args()

# Generate run_name if training and not provided
if args.mode == 'train' and args.run_name is None:
    args.run_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

params = argparse.Namespace(
    cities=args.cities,
    batch_size=args.batch_size,
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})



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
        self.train_datasets = [ZarrDataset(**self.train_datafiles[city]) for city in params.cities]
        self.valid_datasets = [ZarrDataset(**self.valid_datafiles[city]) for city in params.cities]
        self.test_datasets  = [ZarrDataset(**self.test_datafiles[city])  for city in params.cities]

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



# --- INITIALISE DATA AND MODEL (Needed for both modes) ---

# Initialises datasets
train_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_train_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_train_balanced.zarr') for city in params.cities]))
valid_datafiles = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_valid_balanced.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_valid_balanced.zarr') for city in params.cities]))
test_datafiles  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_prepost_test_balanced.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_prepost_test_balanced.zarr')  for city in params.cities]))
data_module = ZarrDataModule(train_datafiles=train_datafiles, 
                             valid_datafiles=valid_datafiles, 
                             test_datafiles=test_datafiles, 
                             batch_size=params.batch_size, 
                             label_map=params.label_map, 
                             shuffle=True)
del train_datafiles, valid_datafiles, test_datafiles


# Initialises model
BACKBONE_PATH = f'{paths.models}/checkpoint-9920'
if not os.path.exists(BACKBONE_PATH):
    print(f"Warning: Backbone path {BACKBONE_PATH} does not exist. Ensure it's correct.")

siamese_nn_model = SiameseModel(backbone=BACKBONE_PATH)
model_module = SiameseModule(
    model= siamese_nn_model, 
    model_name='destruction_finetune_siamese', 
    learning_rate=args.learning_rate,
    weight_decay=0.05,
    weigh_contrast=args.weigh_contrast, #! Should be tuned
    margin_contrast=1.0) 

# --- END INITIALISE DATA AND MODEL ---


# Function to update overview CSV (define this at the top level of your script or import it)
def update_experiment_overview_csv(overview_filepath: str, run_data_dict: dict):
    """Appends a new run's summary to the overview CSV file."""
    fieldnames = [
        'run_id', 'training_start_timestamp', 'training_end_timestamp', 'training_cities',
        'key_hyperparameters', 'best_val_metric_name', 'best_val_metric_value',
        'epoch_of_best_val_metric', 'train_metric_at_best_val_auroc', 'train_metric_at_best_val_loss',
        'best_checkpoint_path', 'evaluation_timestamp',
        'per_city_test_aurocs', 'average_test_auroc'
    ]
    file_exists = os.path.isfile(overview_filepath)
    with open(overview_filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # extrasaction='ignore' is safer
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_data_dict)
    print(f"Run summary appended to: {overview_filepath}")
    
#%% MAIN SCRIPT LOGIC DISPATCHER

if args.mode == 'train':
    # ==============================================================================
    # TRAINING MODE (Alignment Stage + Fine-Tuning Stage)
    # ==============================================================================
    print(f"--- Starting Training Mode for Run ID: {args.run_name} ---")
    print(f"Training with cities: {args.cities}")


    # --- 1. ALIGNMENT STAGE ---
    print("--- Stage 1: Alignment Training (Encoder Frozen) ---")
    model_module.freeze_encoder()
    if 'count_parameters' in globals(): count_parameters(model_module)

    alignment_trainer = pl.Trainer(
        max_epochs=args.max_epochs_align,
        accelerator=device,
        logger=False,             # No separate logger for this stage
        enable_checkpointing=False # No checkpoints for this stage
    )
    alignment_trainer.fit(model=model_module, datamodule=data_module)
    print("--- Alignment Stage Complete ---")



    # --- 2. FINE-TUNING STAGE ---
    print("--- Stage 2: Fine-Tuning (Encoder Unfrozen) ---")
    model_module.unfreeze_encoder()
    if 'count_parameters' in globals(): count_parameters(model_module)

    # Logger for the fine-tuning stage (associated with run_name)
    fine_tune_logger = loggers.CSVLogger(
        save_dir=f'{paths.models}/logs',
        name=model_module.model_name, # Subdirectory for model type
        version=args.run_name         # Subdirectory for this specific run
    )

    # Log all command-line arguments that were used for this run
    # vars(args) converts the argparse Namespace to a dict
    # PTL usually logs LightningModule's hparams. Ensure relevant args are in model_module.hparams
    # If SiameseModule calls self.save_hyperparameters() including args like learning_rate, etc., they will be logged.
    # Explicitly logging all args can be useful:
    if hasattr(fine_tune_logger, 'log_hyperparams'):
         # Create a clean dict of hparams from args for logging
        hparams_to_log = vars(args).copy()
        # Remove args not relevant as hyperparameters for this run's core logic if desired
        # e.g. del hparams_to_log['mode'], del hparams_to_log['checkpoint_to_eval']
        fine_tune_logger.log_hyperparams(hparams_to_log)
    elif hasattr(model_module, 'hparams'): # Fallback for older PTL or direct manipulation
        model_module.hparams.update(vars(args))


    # Checkpoints for the fine-tuning stage
    # PTL 2.x: logger.log_dir usually is the versioned path like ".../logs/model_name/run_name/"
    # For PTL <2.0, logger.save_dir/logger.name/logger.version might be needed.
    # Assuming logger.log_dir points to the correct versioned directory.
    fine_tune_checkpoint_dir = os.path.join(fine_tune_logger.log_dir, 'checkpoints')
    os.makedirs(fine_tune_checkpoint_dir, exist_ok=True)

    fine_tune_model_checkpoint = callbacks.ModelCheckpoint(
        dirpath=fine_tune_checkpoint_dir,
        filename=f"{model_module.model_name}-FT-{{epoch:02d}}-{{val_auroc_epoch:.4f}}",
        monitor='val_auroc_epoch',  # Ensure this metric is logged by your SiameseModule.validation_epoch_end
        mode='max',
        save_top_k=1,
        save_last=True
    )

    fine_tune_early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', # Or 'val_auroc_epoch' if you prefer, then mode='max'
        mode='min',
        patience=args.patience_ft,
        verbose=True
    )

    fine_tune_trainer = pl.Trainer(
        max_epochs=args.max_epochs_ft,
        accelerator=device,
        logger=fine_tune_logger,
        callbacks=[fine_tune_model_checkpoint, fine_tune_early_stopping],
        profiler=profilers.SimpleProfiler() if 'profilers' in globals() else None # Conditional profiler
    )

    # Resuming logic for fine-tuning (simplified: assumes a new run_name means fresh FT)
    initial_ft_ckpt_path = None
    last_ckpt_in_run_dir = os.path.join(fine_tune_checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt_in_run_dir): # Simple check if current run_name dir has a last.ckpt
        print(f"Found existing last.ckpt for run {args.run_name}. To resume, re-run with --checkpoint_to_eval pointing to it, or manage resumption externally.")
        # For a truly new run with a new run_name, this should not exist initially.
        # If you want to enable *automatic* resumption of the *same run_name*, you could set:
        # initial_ft_ckpt_path = last_ckpt_in_run_dir

    print(f"Starting fine-tuning for run: {args.run_name}. Initial FT checkpoint: {initial_ft_ckpt_path if initial_ft_ckpt_path else 'None (training after alignment)'}")
    fine_tune_trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=initial_ft_ckpt_path # Typically None for a new run_name
    )

    print(f"--- Fine-Tuning Stage Complete for Run ID: {args.run_name} ---")
    best_ft_checkpoint_for_eval = None
    if hasattr(fine_tune_model_checkpoint, 'best_model_path') and fine_tune_model_checkpoint.best_model_path and os.path.exists(fine_tune_model_checkpoint.best_model_path):
        print(f"Best fine-tuned checkpoint saved at: {fine_tune_model_checkpoint.best_model_path}")
        best_ft_checkpoint_for_eval = fine_tune_model_checkpoint.best_model_path
    elif os.path.exists(last_ckpt_in_run_dir):
        print(f"No 'best' checkpoint found (or save_top_k=0). Using last checkpoint: {last_ckpt_in_run_dir}")
        best_ft_checkpoint_for_eval = last_ckpt_in_run_dir
    else:
        print("Error: No fine-tuned checkpoint (best or last) was found after training.")

    # Optional: Save a final generic checkpoint if needed, though ModelCheckpoint handles it well.
    # final_explicit_save_path = os.path.join(fine_tune_checkpoint_dir, f"{model_module.model_name}-FT-final_explicit.ckpt")
    # fine_tune_trainer.save_checkpoint(final_explicit_save_path)
    # print(f"Final model from trainer state explicitly saved to: {final_explicit_save_path}")

    if 'empty_cache' in globals(): empty_cache(device=device)

    # Automatically run evaluation if a checkpoint was produced
    if best_ft_checkpoint_for_eval:
        print(f"\n--- Automatically proceeding to per-city evaluation for checkpoint: {best_ft_checkpoint_for_eval} ---")
        # Set up args for evaluation mode (some might need to be passed from original args)
        eval_args_dict = args.__dict__.copy()
        eval_args_dict['mode'] = 'eval_per_city'
        eval_args_dict['checkpoint_to_eval'] = best_ft_checkpoint_for_eval
        # 'cities' for evaluation will be args.cities by default
        # 'training_cities_list' for overview CSV should be args.cities from this training run
        eval_args_dict['training_cities_list'] = args.cities 
        
        # Simulate running the eval mode (or call a function)
        # For simplicity, directly call the core logic of eval mode here
        # This avoids re-parsing args and re-initializing everything if structured as a function.
        # For now, this implies duplication or refactoring eval logic into a callable function.
        # Let's assume for this example, the user will run eval as a separate command.
        print("\nTo evaluate this model per city and update the overview CSV, run:")
        print(f"python your_script_name.py --mode eval_per_city --checkpoint_to_eval \"{best_ft_checkpoint_for_eval}\" --cities {' '.join(args.cities)} --training_cities_list {' '.join(args.cities)}")
    else:
        print("\nSkipping automatic per-city evaluation as no valid checkpoint was identified.")


elif args.mode == 'eval_per_city':
    # ==============================================================================
    # EVALUATION MODE (Per-City on Test Set)
    # ==============================================================================
    if not args.checkpoint_to_eval or not os.path.exists(args.checkpoint_to_eval):
        raise ValueError("Must provide a valid --checkpoint_to_eval path for 'eval_per_city' mode.")

    print(f"--- Starting Evaluation Mode for Checkpoint: {args.checkpoint_to_eval} ---")
    print(f"Evaluating on cities: {args.cities}") # These are test cities

    # Load the fine-tuned model from the specified checkpoint
    # Ensure the model architecture (e.g. backbone path) matches the saved model
    # The SiameseModule requires the 'model' (nn.Module) to be passed if it was ignored in save_hyperparameters
    try:
        model_to_eval = SiameseModule.load_from_checkpoint(
            checkpoint_path=args.checkpoint_to_eval,
            model=SiameseModel(backbone=BACKBONE_PATH) # Provide the nn.Module instance
            # You might need to pass other args if __init__ changed and they are not in hparams.yaml
            # For example, if learning_rate was needed by __init__ but not used in eval:
            # learning_rate=args.learning_rate (or a default, or from loaded hparams)
        )
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Please ensure the model architecture (e.g., backbone) and required __init__ arguments match the saved checkpoint.")
        exit(1)

    model_to_eval.eval()
    model_to_eval.to(device)

    eval_trainer = pl.Trainer(accelerator=device, logger=False) # No training logger needed for eval

    all_city_results = {}
    # data_module.setup('test') # Call this if your DataModule requires explicit setup for test stage

    for city_name in args.cities: # Iterate over cities provided for evaluation
        print(f"--- Evaluating: {city_name} ---")
        city_test_datafile_spec = {
            'images_zarr': f'{paths.data}/{city_name}/zarr/images_prepost_test_balanced.zarr',
            'labels_zarr': f'{paths.data}/{city_name}/zarr/labels_prepost_test_balanced.zarr'
        }
        # Check if city data files exist
        if not os.path.exists(city_test_datafile_spec['images_zarr']) or \
           not os.path.exists(city_test_datafile_spec['labels_zarr']):
            print(f"Warning: Data files for city {city_name} not found. Skipping.")
            all_city_results[city_name] = {"error": "data not found"}
            continue

        city_dataset = ZarrDataset(**city_test_datafile_spec)
        city_test_loader = ZarrDataLoader(
            datafiles={city_name: city_test_datafile_spec},
            datasets=[city_dataset],
            label_map=params.label_map, # from global params
            batch_size=args.batch_size,
            shuffle=False # Crucial for testing
        )

        # .test() calls test_step in your LightningModule. Ensure it logs 'test_auroc_epoch'.
        city_metrics_list = eval_trainer.test(model=model_to_eval, dataloaders=city_test_loader, verbose=False)
        
        if city_metrics_list and isinstance(city_metrics_list, list) and len(city_metrics_list) > 0:
            all_city_results[city_name] = city_metrics_list[0] 
            print(f"Metrics for {city_name}: {city_metrics_list[0]}")
        else:
            print(f"Warning: No metrics returned from eval_trainer.test() for city {city_name}.")
            all_city_results[city_name] = {"error": "no metrics returned"}


    print(f"\n--- Overall Per-City Test Results (from checkpoint: {args.checkpoint_to_eval}) ---")
    for city, metrics in all_city_results.items():
        print(f"City: {city}, Metrics: {metrics}")

    # --- Gather all information for the overview CSV ---
    run_id_for_overview = "N/A"
    model_name_overview = model_module.model_name # Use from instantiated model_module
    training_cities_overview_str = "N/A"
    key_hyperparams_overview_json = "{}"
    best_val_metric_name_overview = "N/A"
    best_val_metric_value_overview = "N/A"
    epoch_of_best_val_overview = "N/A"
    train_auroc_at_best_val_overview = "N/A"
    train_loss_at_best_val_overview = "N/A"
    training_start_ts_overview = "N/A"
    training_end_ts_overview = datetime.datetime.fromtimestamp(os.path.getmtime(args.checkpoint_to_eval)).strftime('%Y-%m-%d %H:%M:%S')


    # Infer run_id and model_name from checkpoint path structure
    # Path: .../logs/<model_name>/<run_id>/checkpoints/<checkpoint_file.ckpt>
    path_parts = args.checkpoint_to_eval.split(os.sep)
    try:
        checkpoints_idx = path_parts.index("checkpoints")
        run_id_for_overview = path_parts[checkpoints_idx - 1]
        model_name_overview = path_parts[checkpoints_idx - 2] 
        run_log_dir_for_csv = os.path.join(f'{paths.models}/logs', model_name_overview, run_id_for_overview)
    except (ValueError, IndexError):
        print(f"Warning: Could not reliably determine run_id/model_name from checkpoint path for CSV logging: {args.checkpoint_to_eval}")
        run_log_dir_for_csv = os.path.dirname(os.path.dirname(args.checkpoint_to_eval)) # Fallback

    hparams_file = os.path.join(run_log_dir_for_csv, 'hparams.yaml')
    metrics_file = os.path.join(run_log_dir_for_csv, 'metrics.csv')

    if os.path.exists(hparams_file):
        with open(hparams_file, 'r') as f:
            hparams = yaml.safe_load(f)
            if args.training_cities_list: # Prefer explicitly passed list if available
                 training_cities_overview_str = ",".join(args.training_cities_list)
            elif 'cities' in hparams and isinstance(hparams['cities'], list): # Fallback to hparams
                 training_cities_overview_str = ",".join(hparams['cities'])
            
            key_hparams_to_log = {
                k: hparams.get(k) for k in [
                    'learning_rate', 'weight_decay', 'weigh_contrast', 'batch_size', 
                    'max_epochs_ft', 'patience_ft' # Add other key FT hparams
                ] if k in hparams
            }
            key_hyperparams_overview_json = json.dumps(key_hparams_to_log)
            if 'run_name' in hparams and run_id_for_overview == "N/A": # If CLI run_name was in hparams
                run_id_for_overview = hparams['run_name']


    if os.path.exists(metrics_file):
        try:
            metrics_df = pd.read_csv(metrics_file)
            metrics_df.dropna(subset=['epoch'], inplace=True)
            metrics_df['epoch'] = metrics_df['epoch'].astype(int)

            monitored_val_metric_col = 'val_auroc_epoch' # This must match ModelCheckpoint monitor and logged name
            train_auroc_col = 'train_auroc_epoch' # Assuming this is logged
            train_loss_col = 'train_loss_epoch'   # Assuming this is logged (or train_loss for epoch)


            if monitored_val_metric_col in metrics_df.columns:
                valid_metrics_df = metrics_df.dropna(subset=[monitored_val_metric_col])
                if not valid_metrics_df.empty:
                    # Assuming mode 'max' for val_auroc_epoch
                    best_epoch_idx = valid_metrics_df[monitored_val_metric_col].idxmax()
                    best_row = valid_metrics_df.loc[best_epoch_idx]

                    best_val_metric_name_overview = monitored_val_metric_col
                    best_val_metric_value_overview = best_row[monitored_val_metric_col]
                    epoch_of_best_val_overview = int(best_row['epoch'])
                    
                    if train_auroc_col in best_row:
                        train_auroc_at_best_val_overview = best_row[train_auroc_col]
                    if train_loss_col in best_row: # Check for existence before accessing
                        train_loss_at_best_val_overview = best_row[train_loss_col]
                    
                    if 'timestamp' in metrics_df.columns:
                        training_start_ts_overview = metrics_df['timestamp'].min() 
                        # training_end_ts_overview already set by checkpoint mod time
        except Exception as e:
            print(f"Warning: Could not read or parse fine-tuning metrics from {metrics_file}: {e}")
            
    # Format per-city test AUROC and calculate average
    per_city_test_aurocs_dict = {
        city: results.get('test_auroc_epoch', results.get('test_auroc', "N/A")) # Check your test_step log name
        for city, results in all_city_results.items() if "error" not in results
    }
    per_city_test_aurocs_json = json.dumps(per_city_test_aurocs_dict)
    
    auroc_values = [val for val in per_city_test_aurocs_dict.values() if isinstance(val, (int, float))]
    average_test_auroc_overview = sum(auroc_values) / len(auroc_values) if auroc_values else "N/A"

    overview_data = {
        'run_id': run_id_for_overview,
        'training_start_timestamp': training_start_ts_overview,
        'training_end_timestamp': training_end_ts_overview,
        'training_cities': training_cities_overview_str,
        'key_hyperparameters': key_hyperparams_overview_json,
        'best_val_metric_name': best_val_metric_name_overview,
        'best_val_metric_value': f"{best_val_metric_value_overview:.4f}" if isinstance(best_val_metric_value_overview, float) else best_val_metric_value_overview,
        'epoch_of_best_val_metric': epoch_of_best_val_overview,
        'train_metric_at_best_val_auroc': f"{train_auroc_at_best_val_overview:.4f}" if isinstance(train_auroc_at_best_val_overview, float) else train_auroc_at_best_val_overview,
        'train_metric_at_best_val_loss': f"{train_loss_at_best_val_overview:.4f}" if isinstance(train_loss_at_best_val_overview, float) else train_loss_at_best_val_overview,
        'best_checkpoint_path': args.checkpoint_to_eval,
        'evaluation_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'per_city_test_aurocs': per_city_test_aurocs_json,
        'average_test_auroc': f"{average_test_auroc_overview:.4f}" if isinstance(average_test_auroc_overview, float) else average_test_auroc_overview,
    }

    overview_csv_filepath = os.path.join(f'{paths.models}/logs', "experiment_overview.csv")
    update_experiment_overview_csv(overview_csv_filepath, overview_data)

    if 'empty_cache' in globals(): empty_cache(device=device)

else:
    print(f"Unknown mode: {args.mode}. Choose 'train' or 'eval_per_city'.")

#%%


#%%
