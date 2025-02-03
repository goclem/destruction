#!/bin/bash
#
# Script: vitmae_finetune_sequence.sh
# Description: The script calles the python script to train and evaluate the model.
# Author: Dominik Wielath
# Date: 2025-02-03
# Version: 1.0

# python3 destruction_vitmae_finetune_sequence.py --cities moschun aleppo volnovakha
# Ther seems to be an error when including deirezzor, this has to be investigated

# python3 destruction_vitmae_finetune_sequence.py --cities hostomel irpin livoberezhnyi moschun volnovakha aleppo damascus daraa deirezzor hama homs idlib raqqa
python3 destruction_vitmae_finetune_sequence.py --cities hostomel irpin livoberezhnyi moschun volnovakha aleppo daraa hama homs idlib raqqa

