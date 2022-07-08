#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Computes prediction statistics
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.09
'''

#%% HEADER

# Modules
import numpy as np
from destruction_utilities import *

#%% FUNCTIONS

# ! Update for patches

# Computes prediction sets
def compute_sets(label:np.ndarray, predict:np.ndarray) -> np.ndarray:
    # Formats labels
    label   = label.astype(bool)
    predict = predict.astype(bool)
    # Computes sets
    tp   = np.logical_and(label, predict)
    tn   = np.logical_and(np.invert(label), np.invert(predict))
    fp   = np.logical_and(np.invert(label), predict)
    fn   = np.logical_and(label, np.invert(predict))
    sets = np.array([tp, tn, fp, fn])
    return sets

# Computes prediction statistics
def compute_statistics(sets:np.ndarray) -> dict:
    tp, tn, fp, fn = np.sum(sets, axis=(1, 2, 3))
    with np.errstate(divide='ignore', invalid='ignore'): # Returns Inf when dividing by 0
        recall    = np.divide(tp, (tp + fn))
        precision = np.divide(tp, (tp + fp))
        accuracy  = np.divide((tp + tn), (tp + tn + fp + fn))
    statistics = dict(tp=tp, tn=tn, fp=fp, fn=fn, recall=recall, precision=precision, accuracy=accuracy)
    return statistics

# Displays prediction masks
def display_statistics(image:np.ndarray, sets:np.ndarray) -> None:
        # Formats titles
        counts = np.sum(sets, axis=(1, 2, 3))
        titles = ['True positive ({:d})', 'True negative ({:d})', 'False positive ({:d})', 'False negative ({:d})']
        titles = list(map(lambda title, count: title.format(count), titles, counts))
        # Formats images
        image  = (image * 255).astype(int)
        colour = (255, 255, 0)
        images = [np.where(np.tile(mask, (1, 1, 3)), colour, image) for mask in sets]
        # Displays images
        fig, axs = pyplot.subplots(2, 2, figsize=(10, 10))
        for image, title, ax in zip(images, titles, axs.ravel()):
            ax.imshow(image)
            ax.set_title(title, fontsize=20)
            ax.axis('off')
        pyplot.tight_layout(pad=2.0)
        pyplot.show()