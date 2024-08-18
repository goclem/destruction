#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Model optimisation for the destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr, dominik.wielath@upf.edu
'''

#%% HEADER

# Packages
import numpy as np
import torch
import typing

from destruction_models import *
from destruction_utilities import *
from torch import optim, nn, utils
from torch.utils.data import ConcatDataset, Subset
from os import path
from torcheval import metrics
from sklearn.metrics import roc_curve

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--cities', nargs='+', type=str, default=['aleppo'], help='List of city names')

# Parse command-line arguments
args = parser.parse_args()

print(args.cities)

# Utilities
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
params = argparse.Namespace(
    cities=args.cities,
    tile_size=128, 
    batch_size=8,  # Slice sizes
    label_map={0:0, 1:0, 2:1, 3:1, 255:torch.tensor(float('nan'))})


max_seq_len = 0
for city in params.cities:
    max_seq_len = max(len(search_data(pattern=pattern(city=city, type='image'))), max_seq_len)

print(f"Device: {device}, tile size: {params.tile_size}, batch size: {params.batch_size}, max sequence length: {max_seq_len}")
#%% INITIALISES DATA LOADERS
# Datasets
train_datasets = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_train.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_train.zarr') for city in params.cities]))
valid_datasets = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_valid.zarr', labels_zarr=f'{paths.data}/{city}/zarr/labels_valid.zarr') for city in params.cities]))
test_datasets  = dict(zip(params.cities, [dict(images_zarr=f'{paths.data}/{city}/zarr/images_test.zarr',  labels_zarr=f'{paths.data}/{city}/zarr/labels_test.zarr')  for city in params.cities]))



for city in params.cities:
    train_datasets[city]['seq_len'] = max_seq_len
    valid_datasets[city]['seq_len'] = max_seq_len
    test_datasets[city]['seq_len'] = max_seq_len

# Intialises data loaders
train_dataset_list = [ZarrDataset(**train_datasets[city]) for city in params.cities]
valid_dataset_list = [ZarrDataset(**valid_datasets[city]) for city in params.cities]
test_dataset_list  = [ZarrDataset(**test_datasets[city])  for city in params.cities]

valid_dataset_city_end_index = {}
test_dataset_city_end_index = {}

for city in params.cities:
    valid_dataset_city_end_index[city] = len(ZarrDataset(**valid_datasets[city]))
    test_dataset_city_end_index[city] = len(ZarrDataset(**test_datasets[city]))

train_dataset_list_concatenated = [ConcatDataset(train_dataset_list)]  # Add all your datasets here
valid_dataset_list_concatenated = [ConcatDataset(valid_dataset_list)]  # Add all your datasets here
test_dataset_list_concatenated = [ConcatDataset(test_dataset_list)]  # Add all your datasets here

train_loader = ZarrDataLoader(train_dataset_list_concatenated, batch_size=params.batch_size, label_map=params.label_map)
valid_loader = ZarrDataLoader(valid_dataset_list_concatenated, batch_size=params.batch_size, label_map=params.label_map)
test_loader  = ZarrDataLoader(test_dataset_list_concatenated,  batch_size=params.batch_size, label_map=params.label_map)



# Checks the indices for the validation and test set to later identify subsets correspoding to individual cities
valid_dataset_city_index = {}
test_dataset_city_index = {}

start_index_val = 0
end_index_val = 0
start_index_test = 0
end_index_test = 0
for city in params.cities:
    
    print(f"\n{city.capitalize()}")
    
    # Validation Set
    # Load dataset corresponding to the city
    city_loader_val = ZarrDataLoader([ZarrDataset(**valid_datasets[city])], batch_size=params.batch_size, label_map=params.label_map, training=False)

    end_index_val += valid_dataset_city_end_index[city]
    print(f"Validation set range: {start_index_val} - {end_index_val}")
    
    city_index_list_val = list(range(start_index_val, end_index_val))
    valid_dataset_city_index[city] = city_index_list_val
    
    # Subset the conatenated dataset to extract the city specific dataset
    subset_loader_val = ZarrDataLoader([Subset(valid_dataset_list_concatenated[0],city_index_list_val)], batch_size=params.batch_size, label_map=params.label_map, training=False)
    
    # Compare the two dataloaders
    assert compare_dataloaders(city_loader_val, subset_loader_val), f"Error with indexing of the validation dataset for {city.capitalize()}"
    start_index_val = end_index_val
    
    # Test Set
    # Load dataset corresponding to the city
    city_loader_test = ZarrDataLoader([ZarrDataset(**test_datasets[city])], batch_size=params.batch_size, label_map=params.label_map, training=False)

    end_index_test += test_dataset_city_end_index[city]
    print(f"Test set range: {start_index_test} - {end_index_test}")
    
    city_index_list_test = list(range(start_index_test, end_index_test))
    test_dataset_city_index[city] = city_index_list_test
    
    # Subset the conatenated dataset to extract the city specific dataset
    subset_loader_test = ZarrDataLoader([Subset(test_dataset_list_concatenated[0],city_index_list_test)], batch_size=params.batch_size, label_map=params.label_map, training=False)
    
    # Compare the two dataloaders
    assert compare_dataloaders(city_loader_test, subset_loader_test), f"Error with indexing of the test dataset for {city.capitalize()}"
    start_index_test = end_index_test

print(f"\n")

            

''' Checks data loaders
X, Y = next(iter(train_loader))
for i in range(5):
    display_sequence(X[i], Y[i], grid_size=(5,5))
del X, Y
''' 

# Prints excluded cities
[print(f'Excluding: {city}') for city, size in zip(params.cities, train_loader.slice_sizes) if size == 0]

#%% INTIALISES MODEL

# Initialises model components
#? feature_extractor = torch.load(f'{paths.models}/Aerial_SwinB_SI.pth')
feature_extractor = ResNextExtractor(dropout=0.0)
image_encoder     = ImageEncoder(feature_extractor=feature_extractor)
sequence_encoder  = dict(input_dim=512, max_length=25, n_heads=4, hidden_dim=512, n_layers=2, dropout=0.0)
sequence_encoder  = SequenceEncoder(**sequence_encoder)
prediction_head   = PredictionHead(input_dim=512, output_dim=1)

# Initialises model wrapper
model = ModelWrapper(image_encoder, sequence_encoder, prediction_head)
model.to(device)

print(model.__class__.__name__)
# Checks model parameters
count_parameters(model)
count_parameters(model.image_encoder)
count_parameters(model.sequence_encoder)

del image_encoder, sequence_encoder, prediction_head

#%% ALIGNING PARAMETERS

''' #? Loads previous checkpoint
if path.exists(f'{paths.models}/ModelWrapper_best.pth'):
    model = torch.load(f'{paths.models}/ModelWrapper_best.pth')
'''

#? Parameter alignment i.e. freezes image encoder's parameters
# set_trainable(model.image_encoder.feature_extractor, False)
# count_parameters(model)

optimiser = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
criterion = BceLoss(focal=False, drop_nan=True, alpha=0.25, gamma=2.0)

def train(model:nn.Module, train_loader, valid_loader, device:torch.device, criterion, optimiser, n_epochs:int=1, patience:int=1, accumulate:int=1, label:str=''):
    '''Trains a model using a training and validation sample'''
    best_loss, counter = torch.tensor(float('inf')), 0
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1:03d}/{n_epochs:03d}')
        train_loss = optimise(model=model, loader=train_loader, device=device, criterion=criterion, optimiser=optimiser, accumulate=accumulate)
        # Early stopping
        valid_loss = validate(model=model, loader=valid_loader, device=device, criterion=criterion)
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model, f'{paths.models}/{model.__class__.__name__}_{label}_best.pth')
            # Shuffles zarr datasets
            for city in params.cities:
                shuffle_zarr(**train_datasets[city])
                #shuffle_zarr(**valid_datasets[city])
        else:
            counter += 1
            if counter >= patience:
                print('- Early stopping')
                break

# Test
# Training
train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device, 
      criterion=criterion, 
      optimiser=optimiser, 
      n_epochs=25, 
      patience=3,
      accumulate=4,
      label='resnext_encoder')

# Clears GPU memory
empty_cache(device)


model = torch.load(f'{paths.models}/ModelWrapper_resnext_encoder_best.pth') # Loads best model

"""print("Per citiy validation")
for city in valid_dataset_city_index.keys():
    print(f"\n{city.capitalize()}")
    subset_loader = ZarrDataLoader([Subset(valid_dataset_list[0],valid_dataset_city_index[city])], batch_size=params.batch_size, label_map=params.label_map, training=False)
    validate(model=model, loader=subset_loader, device=device, criterion=criterion)
"""
print("Per citiy validation")
for i, city in enumerate(params.cities):
    print(f"\n{city.capitalize()}")
    
    city_data_loader = ZarrDataLoader([valid_dataset_list[i]], batch_size=params.batch_size, label_map=params.label_map, training=False)
    val_loss = validate(model=model, loader=city_data_loader, device=device, criterion=criterion)
    print(val_loss)  
      
print(f"\nPer citiy test")
for i, city in enumerate(params.cities):
    print(f"\n{city.capitalize()}")
    
    city_data_loader = ZarrDataLoader([test_dataset_list[i]], batch_size=params.batch_size, label_map=params.label_map, training=False)
    test_loss = validate(model=model, loader=city_data_loader, device=device, criterion=criterion)
    print(test_loss)


# Clears GPU memory
empty_cache(device)

#%% FINE TUNES ENTIRE MODEL

'''
#? Fine tuning i.e. unfreezes image encoder's parameters
model = torch.load(f'{paths.models}/ModelWrapper_aligmnent_best.pth') # Loads best model
set_trainable(model.image_encoder.feature_extractor, True)

optimiser = optim.AdamW(model.parameters(), lr=1e-5)
count_parameters(model)

train(model=model, 
      train_loader=train_loader, 
      valid_loader=valid_loader, 
      device=device, 
      criterion=criterion, 
      optimiser=optimiser, 
      n_epochs=25, 
      patience=3,
      accumulate=4,
      label='finetuning')

# Clears GPU memory
empty_cache(device)
'''

#%% ESTIMATES THRESHOLD

model = torch.load(f'{paths.models}/ModelWrapper_resnext_encoder_best.pth') # Loads best model

def compute_threshold(model:nn.Module, loader, device:torch.device, n_batches:int=None) -> float:
    '''Estimates threshold for binary classification'''
    Y, Yh  = predict(model, loader=train_loader, device=device, n_batches=n_batches)
    subset = ~Y.isnan()
    fpr, tpr, thresholds = roc_curve(Y[subset].cpu(), Yh[subset].cpu())
    threshold = thresholds[np.argmax(tpr - fpr)]
    return threshold

# Threshold estimation  
threshold = compute_threshold(model=model, loader=train_loader, device=device)
print(f'Threshold: {threshold:.2f}')

# Testing
validate(model=model, loader=test_loader, device=device, criterion=criterion, threshold=threshold)

#%% CHECKS PREDICTIONS

model.eval()
X, Y = next(iter(test_loader))
susbset = torch.sum(Y==1, axis=(1, 2)) > 0
X, Y = X[susbset], Y[susbset]

with torch.no_grad():
    Yh = model(X.to(device)).cpu()
Y, Yh  = Y.squeeze(), Yh.squeeze()

threshold = 0.5
status = torch.where(torch.isnan(Y), torch.nan, torch.eq(Y, Yh > threshold))

for i in np.random.choice(range(len(X)), 2, replace=False):
    titles = [f'{s:.0f}\nY: {y:.0f} - Yh: {yh > threshold:.0f} ({yh:.2f})' for s, y, yh in zip(status[i], Y[i], Yh[i])]
    display_sequence(X[i], titles, grid_size=(5,5))
del titles

#%%
