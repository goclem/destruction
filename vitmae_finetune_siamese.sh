#!/bin/bash
#
# Script: vitmae_finetune_siamese.sh
# Description: The script calles the python script to train and evaluate the model.
# Author: Dominik Wielath
# Date: 2025-02-03
# Version: 1.0

#python3 destruction_finetune_siamese.py --cities damascus
python3 destruction_finetune_siamese.py --cities hostomel irpin livoberezhnyi moschun rubizhne volnovakha aleppo daraa deirezzor hama homs idlib raqqa

# Define an array of cities
#cities=(hostomel aleppo homs)

# Loop over the cities array
#for city in "${cities[@]}"; do
#    echo "Processing city: $city"
#    python3 destruction_finetune_siamese.py --cities "$city"
#done