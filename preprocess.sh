#!/bin/sh

echo("Starting preprocessing for SNN")
python destruction_preprocess.py --city aleppo --mode snn --pre_image_index 0,1
python destruction_preprocess.py --city damascus --mode snn --pre_image_index 0
python destruction_preprocess.py --city daraa --mode snn --pre_image_index 0,1
python destruction_preprocess.py --city hama --mode snn --pre_image_index 0
python destruction_preprocess.py --city homs --mode snn --pre_image_index 0
python destruction_preprocess.py --city idlib --mode snn --pre_image_index 0
python destruction_preprocess.py --city raqqa --mode snn --pre_image_index 0
python destruction_preprocess.py --city deir-ez-zor --mode snn --pre_image_index 0
python destruction_concatenate.py