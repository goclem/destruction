#!/bin/bash
#
# Script: vitmae_finetune_siamese.sh
# Description: The script calles the python script to train and evaluate the model.
# Author: Dominik Wielath
# Date: 2025-02-03
# Version: 1.0

#python3 destruction_finetune_siamese.py --cities moschun #hostomel

# --- Define Hyperparameter Grids ---
learning_rates=(1e-4 2e-4)
weights_contrast=(0.05 0.1 0.25)
weights_decay=(0.01 0.05 0.1)

# --- Define Fixed Parameters ---
CITIES="hostomel irpin livoberezhnyi moschun rubizhne volnovakha aleppo daraa deirezzor hama homs idlib raqqa"
MODE="train"
MAX_EPOCHS_ALIGN=1
MAX_EPOCHS_FT=100
PATIENCE_FT=2 # Increased default from your original script based on my previous advice, adjust if 1 is intentional.
BATCH_SIZE=64
MARGIN_CONTRAST=1 # As discussed, this isn't actively used by your current loss function
BASE_RUN_NAME="grid_search_$(date +%Y%m%d-%H%M%S)" # A base name for this set of grid search runs

# --- Iterate over Hyperparameters ---

for lr in "${learning_rates[@]}"; do
  for wc in "${weights_contrast[@]}"; do
    for wd in "${weights_decay[@]}"; do
      # Construct a unique run name for this combination
      RUN_NAME="${BASE_RUN_NAME}_lr${lr}_wc${wc}_wd${wd}"

      echo "-------------------------------------------------------------------------"
      echo "Starting run: ${RUN_NAME}"
      echo "Parameters: LR=${lr}, WC=${wc}, WD=${wd}"
      echo "Full command:"
      
      # Construct the command
      COMMAND="python3 destruction_finetune_siamese.py \
          --cities ${CITIES} \
          --mode ${MODE} \
          --run_name ${RUN_NAME} \
          --max_epochs_align ${MAX_EPOCHS_ALIGN} \
          --max_epochs_ft ${MAX_EPOCHS_FT} \
          --patience_ft ${PATIENCE_FT} \
          --learning_rate ${lr} \
          --batch_size ${BATCH_SIZE} \
          --weight_contrast ${wc} \
          --weight_decay ${wd} \
          --margin_contrast ${MARGIN_CONTRAST}"
      
      echo "${COMMAND}"
      echo "-------------------------------------------------------------------------"
      
      # Execute the command
      eval "${COMMAND}"
      
      # Optional: Add a small delay if needed, or error checking
      if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Run ${RUN_NAME} failed."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # Decide if you want to stop the grid search on error or continue
        # exit 1 # Uncomment to stop on first error
      fi
      echo "Finished run: ${RUN_NAME}"
      echo "-------------------------------------------------------------------------"
      echo "" # Add a blank line for readability
    done
  done
done

echo "Grid search completed."


#python3 destruction_finetune_siamese.py --cities hostomel irpin livoberezhnyi moschun rubizhne volnovakha aleppo daraa deirezzor hama homs idlib raqqa

# Define an array of cities
#cities=(hostomel aleppo homs)

# Loop over the cities array
#for city in "${cities[@]}"; do
#    echo "Processing city: $city"
#    python3 destruction_finetune_siamese.py --cities "$city"
#done