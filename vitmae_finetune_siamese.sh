#!/bin/bash
#
# Script: vitmae_finetune_siamese.sh
# Description: The script calles the python script to train and evaluate the model.
# Author: Dominik Wielath
# Date: 2025-02-03
# Version: 1.0

#python3 destruction_finetune_siamese.py --cities moschun volnovakha

# There seems to be an error when including deirezzor or homs, this has to be investigated

# Looks like Homs and Aleppo didn't work
#python3 destruction_finetune_siamese.py --cities hostomel irpin livoberezhnyi moschun volnovakha daraa hama idlib raqqa
python3 destruction_finetune_siamese.py --cities aleppo

# Define an array of cities
#cities=(hostomel aleppo homs)

# Loop over the cities array
#for city in "${cities[@]}"; do
#    echo "Processing city: $city"
#    python3 destruction_finetune_siamese.py --cities "$city"
#done