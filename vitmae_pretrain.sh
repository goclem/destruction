#!/bin/bash
#
# Script: vitmae_pretrain.sh
# Description: The script calles the python script to train and evaluate the model.
# Author: Dominik Wielath
# Date: 2025-02-03
# Version: 1.0

# python3 ddestruction_vitmae_pretrain.py --cities moschun aleppo volnovakha
# Ther seems to be an error when including deirezzor, this has to be investigated

# python3 ddestruction_vitmae_pretrain.py --cities hostomel irpin livoberezhnyi moschun volnovakha aleppo damascus daraa deirezzor hama homs idlib raqqa
python3 destruction_vitmae_pretrain.py --cities hostomel irpin livoberezhnyi moschun volnovakha aleppo daraa hama homs idlib raqqa

