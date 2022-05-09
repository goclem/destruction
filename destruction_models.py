#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Initlaises model structures
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.09
'''

# Modules
from tensorflow.keras import layers, models

# Convolution block
def convolution_block(inputs, filters:int, dropout:float, name:str):
    convolution   = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution')(inputs)
    pooling       = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation')(pooling)
    outputs       = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(normalisation)
    return outputs

# Dense block
def dense_block(inputs, units:int=1, dropout:float=0, name:str=''):
    dense         = layers.Dense(units=units, activation='relu', use_bias=False, kernel_initializer='he_normal', name=f'{name}_dense')(inputs)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation')(dense)
    outputs       = layers.Dropout(rate=dropout, name=f'{name}_dropout')(normalisation)
    return outputs

# Convolutional network
def convolutional_network(shape:tuple, filters:int, units:int, dropout:float):
    # Input layer
    inputs = layers.Input(shape=shape, name='inputs')
    # Hidden convolutional layers
    tensor = convolution_block(inputs, filters=filters*1, dropout=dropout, name='conv_block1')
    tensor = convolution_block(tensor, filters=filters*2, dropout=dropout, name='conv_block2')
    tensor = convolution_block(tensor, filters=filters*3, dropout=dropout, name='conv_block3')
    tensor = convolution_block(tensor, filters=filters*4, dropout=dropout, name='conv_block4')
    tensor = convolution_block(tensor, filters=filters*5, dropout=dropout, name='conv_block5')
    # Hidden dense layers
    tensor = layers.Flatten(name='flatten')(tensor)
    tensor = dense_block(tensor, units=units, dropout=dropout, name='dense_block1')
    tensor = dense_block(tensor, units=units, dropout=dropout, name='dense_block2')
    # Output layer
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(tensor)
    # Model
    model   = models.Model(inputs=inputs, outputs=outputs, name='convolutional_network')
    return model

# Encoder block with shared parameters
def encoder_block_shared(shape:tuple, filters:int=1, dropout=0):
    inputs  = layers.Input(shape=shape, name='inputs')
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name='block1')
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout, name='block2')
    tensor  = convolution_block(tensor, filters=filters*3, dropout=dropout, name='block3')
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout, name='block4')
    tensor  = convolution_block(tensor, filters=filters*5, dropout=dropout, name='block5')
    outputs = layers.Flatten(name='encoder_flatten')(tensor)
    encoder = models.Model(inputs=inputs, outputs=outputs, name='encoder')
    return encoder

# Siamese convolutional network
def siamese_convolutional_network(shape:tuple, args_encode:dict, args_dense:dict):
    # Input layers
    images1 = layers.Input(shape=shape, name='images1')
    images2 = layers.Input(shape=shape, name='images2')
    # Hidden convolutional layers (shared parameters)
    encoder_block = encoder_block_shared(shape=shape, **args_encode)
    encode1 = encoder_block(images1)
    encode2 = encoder_block(images2)
    # Hidden dense layers
    concat  = layers.Concatenate(name='concatenate')(inputs=[encode1, encode2])
    dense   = dense_block(concat, **args_dense, name='dense_block1')
    dense   = dense_block(dense,  **args_dense, name='dense_block2')
    dense   = dense_block(dense,  **args_dense, name='dense_block3')
    dense   = dense_block(dense,  **args_dense, name='dense_block4')
    dense   = dense_block(dense,  **args_dense, name='dense_block5')
    # Output layer
    outputs = layers.Dense(units=1, activation='tanh', name='outputs')(dense)
    # Model
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='siamese_convolutional_network')
    return model