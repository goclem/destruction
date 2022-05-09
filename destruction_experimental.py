#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Experimental functions
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.09
'''

#%% SEQUENCE GENERATOR

# Modules
import numpy as np
import itertools

# Generates sequences of an arbitatry length
def generate_sequences(sequence:bool, size:int=2) -> list:
    index0  = np.where(sequence==0)[0]
    index1  = np.where(sequence==1)[0]
    assert len(index0) >= size, 'Sequence smaller than size'
    assert len(index1) >= size, 'Sequence smaller than size'
    sequence0 = itertools.combinations(index0, size)
    sequence1 = itertools.combinations(index1, size)
    sequences = list(itertools.product(sequence0, sequence1))
    return sequences

# Simulated data
sequence = np.array((False, False, False, True, True, True))
sequence = np.tile(sequence, 9).reshape(3, 3, len(sequence))

# Sequences
sequence = sequence.reshape(3 * 3, 6)
sequence = np.apply_along_axis(generate_sequences, 1, sequence)
# %%

import re
from destruction_utilities import search_files


def make_pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif'):
    pattern = f'^.*{city}/.*/{type}_{date}\.{ext}$'
    return pattern

search_files('../data', pattern=pattern)

def search_files(directory:str, pattern:dict='.') -> list:
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files
