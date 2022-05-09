# Monitoring war destruction from space using machine learning (continued)

- Authors: Hannes Mueller, Andre Groeger, Clement Gorin
- Updated: 2022.05.09

Follow-up project using Mueller et al. Monitoring war destruction from space using machine learning. Proceedings of the National Academy of Sciences, 118(23), 2021.

## Scripts

```
./destruction
	/destruction_utilities.py
	/destruction_preprocess.py
	/destruction_models.py
	/destruction_optimise.py
	/destruction_predict.py
	/destruction_postprocess.py
	/destruction_statistics.py
	/destruction_environment.yml
```

## Data

```
./data
	/aleppo
		/images
			/image_2011_01_01.tif
			/...
		/labels
			/label_2011_01_01.tif
			/...
		/predicts
			/predict_2011_01_01.tif
			/...
		/others
			/aleppo_damage.gpkg
			/aleppo_noanalysis.gpkg
			/aleppo_settlement.gpkg
			/aleppo_samples.tif
	/...
```