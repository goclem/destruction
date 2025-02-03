#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Model structures for the destruction project
@authors: Clement Gorin, Dominik Wielath
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import math
import torch

from torch import nn

#%% SEQUENCE ENCODER

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim:int, max_length:int, n_heads:int, hidden_dim:int, n_layers:int, dropout:float=0.0):
        super().__init__()
        self.input_dim  = input_dim
        self.max_length = max_length
        self.positional_encoding = self.positional_encoder()
        self.attention_mask = self.attention_mask()
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layers, num_layers=n_layers)
        
    def positional_encoder(self) -> torch.Tensor:
        pe = torch.zeros(self.max_length, self.input_dim)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_dim, 2).float() * (-math.log(10000.0) / self.input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def attention_mask(self) -> torch.Tensor:
        mask = torch.ones(self.max_length, self.max_length)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, H:torch.Tensor) -> torch.Tensor:
        t = H.size(1) # n x t x d
        P = self.positional_encoding[:, :t, :].to(H.device)
        M = self.attention_mask[:t, :t].to(H.device)
        H = self.transformer(H + P, M)
        return H
    
#%% PREDITION HEAD

class PredictionHead(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, H:torch.Tensor) -> torch.Tensor:
        Y = self.output_layer(H)
        return Y 

#%% TESTS MODEL COMPONENTS

'''
from destruction_utilities import *

# Initialises model components
feature_extractor = torch.load(f'{paths.models}/Aerial_SwinB_SI.pth')
feature_extractor = ResNextExtractor()
image_encoder     = ImageEncoder(feature_extractor)
sequence_encoder  = dict(input_dim=512, max_length=25, n_heads=4, hidden_dim=512, n_layers=2, dropout=0.0)
sequence_encoder  = SequenceEncoder(**sequence_encoder)
prediction_head   = PredictionHead(input_dim=512, output_dim=1)
model = ModelWrapper(image_encoder, sequence_encoder, prediction_head)

# Checks model parameters
count_parameters(model)
count_parameters(model.image_encoder)
count_parameters(model.sequence_encoder)
count_parameters(model.prediction_head)

# Loads batch
X, Y = next(iter(train_loader))
print(X.size(), Y.size())
display_sequence(X[0], Y[0], grid_size=(5,5))

# Tests model components
with torch.no_grad():
    H = image_encoder(X)
    H = sequence_encoder(H)
    Y = prediction_head(H)

# Tests full model
with torch.no_grad():
    Y = model(X)
'''
