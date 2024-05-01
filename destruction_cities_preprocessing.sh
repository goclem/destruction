#!/bin/bash
#
# Script: destruction_cities_preprocessing.sh
# Description: The script calles the python script to preprocess the city data.
# Author: Dominik Wielath
# Date: 2024-05-01
# Version: 1.0

declare -a Cities=("aleppo" "moschun" "volnovakha")

for city in "${Cities[@]}"; do
    echo "Preprocessing" $city
    python3 destruction_data.py --city $city
    echo "Preprocessing done for" $city
    echo " "
done

# End of script
