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

#%% FEATURE EXTRACTOR

class ResNeXtBlock(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, stride=2, cardinality=32, base_width=4, dropout=0.0):
        super(ResNeXtBlock, self).__init__()
        hidden_dim = cardinality * base_width
        # 1x1 convolutional layer to reduce dimensions
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        # 3x3 grouped convolutional layer
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        # 1x1 convolutional layer to expand dimensions
        self.conv3 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(output_dim)
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_dim)
            )
        self._initialize_weights()
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, X):
        H = self.conv1(X)
        H = self.norm1(H)
        H = torch.relu(H)
        H = self.conv2(H)
        H = self.norm2(H)
        H = torch.relu(H)
        H = self.conv3(H)
        H = self.norm3(H)
        H += self.shortcut(X)
        Y = torch.relu(H)
        Y = self.dropout(Y)
        return Y

class ResNextExtractor(nn.Module):
    def __init__(self, dropout=0.0):
        super(ResNextExtractor, self).__init__()
        self.block0 = ResNeXtBlock(input_dim=3,   output_dim=128, stride=2, cardinality=32, dropout=dropout) # 128x128x3 > 64x64x64
        self.block1 = ResNeXtBlock(input_dim=128, output_dim=128, stride=2, cardinality=32, dropout=dropout) # 64x64x64  > 32x32x128
        self.block2 = ResNeXtBlock(input_dim=128, output_dim=128, stride=2, cardinality=32, dropout=dropout) # 32x32x128 > 16x16x128
        self.block3 = ResNeXtBlock(input_dim=128, output_dim=128, stride=2, cardinality=32, dropout=dropout) # 16x16x128 > 8x8x128
        self.block4 = ResNeXtBlock(input_dim=128, output_dim=128, stride=2, cardinality=32, dropout=dropout) # 8x8x128   > 4x4x128
        
    def forward(self, X):
        H0 = self.block0(X)
        H1 = self.block1(H0)
        H2 = self.block2(H1)
        H3 = self.block3(H2)
        H4 = self.block4(H3)
        return H1, H2, H3, H4

#%% IMAGE ENCODER

class ImageEncoder(nn.Module):
    def __init__(self, feature_extractor:nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.downscale_layers  = nn.ModuleList([nn.Conv2d(128, 8, kernel_size=ks, stride=ks) for ks in [8, 4, 2, 1]])
        self.adaptive_pooling  = nn.AdaptiveMaxPool2d((1, 1))
        self.layer_norm        = nn.LayerNorm(512)
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        n, t, d, h, w = X.size()
        H = X.view(n*t, d, h, w)
        H = self.feature_extractor(H) # 32x32x128 16x16x128 8x8x128 4x4x128
        H = [layer(h) for layer, h in zip(self.downscale_layers, H)]
        H = [h.view(n, t, -1) for h in H]
        H = torch.cat(H, dim=-1)
        H = self.layer_norm(H)
        return H

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
    
#%% MODEL WRAPPER

class ModelWrapper(nn.Module):
    def __init__(self, image_encoder:nn.Module, sequence_encoder:nn.Module, prediction_head:nn.Module):
        super().__init__()
        self.image_encoder    = image_encoder
        self.sequence_encoder = sequence_encoder
        self.prediction_head  = prediction_head

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        H = self.image_encoder(X)    # Mapping: n x t x d x h x w > n x t x k
        H = self.sequence_encoder(H) # Mapping: n x t x k > n x t x k
        Y = self.prediction_head(H)  # Mapping: n x t x k > n x t x 1
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
