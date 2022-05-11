# Data guidelines

Updated 2022-05-11 by Clément

## Structure

- Each city is separated for modularity
- One-to-one correspondence between images, labels and predictions
- See below for naming conventions
- Tile size `128 x 128 x 3` (`ts x ts x d`)

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

## Rasters

**All rasters**

- Geotiff format (.tif)
- EPSG:4326 projection

**Images**

- Name: `image_[yyyy-mm-dd].tif`
- Size: `h x w x d` with `h` and `w` multiples of 256
- Type: 8-byte unsigned integer
- Values: `0-255`

**Labels**

- Name: `label_[yyyy-mm-dd].tif`
- Size: `(h / ts) x (w / ts)`
- Type: 8-byte unsigned integer
- Values:
	- `1` low destruction
	- `2` medium destruction
	- `3` high destruction
	- `0` otherwise

**Predictions**

- Name: `predict_[yyyy-mm-dd].tif`
- Size: `(h / ts) x (w / ts)`
- Type: Float 32
- Values:	`0-1` conditional probabilities

**Samples**

- Name: `[city]_samples.tif`
- Size: `(h / ts) x (w / ts)`
- Type: 8-byte unsigned integer
- Values:
	- `1` training
	- `2` validation
	- `3` test
	- `0` otherwise

## Vectors

**All vectors**

- Geopackage format 
- EPSG:4326 projection
- Validate geometries
- Separate cities

**Damage**

- Name: `[city]_damage.gpkg`
- Variables (integer):
	- Names: `[yyyy-mm-dd]` (one per census)
	- Values:
		- `1` low destruction
		- `2` medium destruction
		- `3` high destruction
		- `0` otherwise

**Noanalysis**

- Name: `[city]_noanalysis.gpkg`
- Variables (string): `reason`

**Settlement**

- Name: `[city]_settlement.gpkg`
- Variables: `None`