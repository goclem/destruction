#!/bin/bash
#
# Script: destruction_cities_optimise.sh
# Description: The script calles the python script to train and evaluate the model.
# Author: Dominik Wielath
# Date: 2024-05-01
# Version: 1.0

# python3 destruction_optimise.py --cities moschun aleppo volnovakha
# Ther seems to be an error when including deirezzor, this has to be investigated
# python3 destruction_optimise.py --cities hostomel irpin livoberezhnyi moschun volnovakha aleppo damascus daraa deirezzor hama homs idlib raqqa
python3 destruction_optimise.py --cities hostomel irpin livoberezhnyi moschun volnovakha aleppo damascus daraa hama homs idlib raqqa