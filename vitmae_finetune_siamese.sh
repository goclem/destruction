#!/bin/bash
#
# Script: vitmae_finetune_siamese.sh
# Description: The script calles the python script to train and evaluate the model.
# Author: Dominik Wielath
# Date: 2025-02-03
# Version: 1.0

#python3 destruction_finetune_siamese.py --cities moschun volnovakha

# There seems to be an error when including deirezzor or homs, this has to be investigated
python3 destruction_finetune_siamese.py --cities hostomel irpin livoberezhnyi moschun volnovakha aleppo daraa hama idlib raqqa

