#!/bin/bash
#
# Script: destruction_cities_preprocessing.sh
# Description: The script calles the python script to preprocess the city data.
# Author: Dominik Wielath
# Date: 2024-05-01
# Version: 1.0

#declare -a Cities=("hostomel" "irpin" "livoberezhnyi" "moschun" "volnovakha" "aleppo" "damascus" "daraa" "deirezzor" "hama" "homs" "idlib" "raqqa")
#declare -a Cities=("damascus" "daraa" "deirezzor" "hama" "homs" "idlib" "raqqa")
declare -a Cities=("damascus") # "volnovakha" "moschun")

for city in "${Cities[@]}"; do
    echo "Preprocessing" $city
    python3 destruction_data.py --city $city
    echo "Preprocessing done for" $city
    echo " "
done

# End of script
