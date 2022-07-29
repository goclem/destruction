# Scripts guidelines

Updated in 2022-05-11 by Clément

## Structure

All scripts should be in a single folder used as a GitHub repository for easy synchronisation. There should be one script for each step of the project. Scripts should use a common prefix like `damage_` with self-explanatory names. We should only use a few well-maintained libraries i.e. `tensoflow`, `numpy`, `pandas`, `os`,  `matplotlib`, `rasterio`, `shutil` should be enough. A Python 3.9 environment with the GPU version of `tensorflow` and the other packages are saved in an environment file for portability. 

```
./damage_tensorflow
	/damage.code-workspace
	/damage_environment.yml
	/damage_utilities.py
	/damage_preprocess.py
	/damage_models.py
	/damage_optimise.py
	/damage_predict.py
	/damage_postprocess.py
	/damage_statistics.py

```
**Details on `destruction_preprocess.py`**

The script should be used via the command line and has the following options.


* **city**, the city for which to run the preprocessing script on
* **mode**, one of: snn (or) cnn (or) all
* **pre_image_index**, index of images to use as pre image; for SNN only; first image is passed as index 0
* **dataset**, one of: all (or) train (or) validate (or) test
* **refresh_sample**, regenrate sample? default=False

_Usage example_

The following snippet prepares train, test, and validation data for SNN for the city of aleppo using the first two images as pre image; `sample.tif` will be generated from scratch and old `sample.tif` will be replaced.

```
python destruction_preprocess.py --city aleppo --mode snn --pre_image_index 0,1 --dataset all --refresh_sample
```


**Details on `utilities.py`**

Most functions are already programmed. We should use four-dimensional `numpy.ndarray` (i.e. `n x w x h x d`) whenever possible. This results in shorter (i.e. few for loops) and faster code since (i.e. linear algebra). List comprehension and vectorised functions (i.e. `map`, `filter` or `reduce`) seamlessly apply.

```python
# utilities.py

# File utilities
ef search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''

def extract_dates(files:list, pattern:str='\d{4}-\d{2}-\d{2}') -> list:

# Raster utilities

def read_raster(source:str, band:int=None, window=None, dtype:str=None) -> np.ndarray:
    '''Reads a raster as a numpy array'''

def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='uint8') -> None:
    '''Writes a numpy array as a raster'''

def rasterise(source, profile, attribute:str=None, dtype:str='uint8') -> np.ndarray:
    '''Tranforms vector data into raster'''

# Array utilities

def tile_sequences(images:np.ndarray, tile_size:tuple=(128, 128)) -> np.ndarray:
    '''Converts images to sequences of tiles'''
    
# Display utilities

def display(image:np.ndarray, title:str='', cmap:str='gray') -> None:
    '''Displays an image'''

def compare(images:list, titles:list=['Image'], cmaps:list=['gray']) -> None:
    '''Displays multiple images'''

def display_structure(model, path:str) -> None:
    '''Displays keras model structure'''
```

**Details on `models.py` and `optimise.py`**

The script `models.py` should contains uncompiled model structures that can be called into `optimise.py` using `from damage_models import convolutional_network`, and compiled. Optimised models should be stored in a dedicated folder, which can be loaded into `predict.py`.

```
./models
	/convolutional_network_optimised.h5
	/convolutional_network_structure.html
	/...
```

Regarding model structures, batch-normalisation and (spatial) dropout should be used after each layer. We could define a convolutional block structure that performs the following sequence of operations: convolution, activation, pooling, normalisation, spatial dropout. Likewise, a dense block structure could perform the following operations dot product, activation, normalisation, dropout. Each block could be called within each model structure.

Regarding optimisation, the binary focal cross-entropy loss function should be used, which works well with the Adam optimiser. To prevent overfitting, a dedicated validation sample should be used, along with early-stopping callbacks. Data generators in `tensorflow` work very well and should be used. There is no need to optimise the filter size, pooling size and the number of epoch. Instead, we should focus on the number of convolutional filters. The number of convolutional blocks can be computed from the image size. We could also have a strong baseline model trained on all the available data, which is re-estimated on specific cities before prediction. The optimisation script should also contain an image augmentation section, we can use the `tensorflow` utilities for this.

```python
# models.py

def convolutional_block(inputs:tensor, filters:int, dropout:float, name:str) -> tensor:

def dense_block(inputs:tensor, units:int, dropout:float, name:str) -> tensor:

def convolutional_network(input_shape, filters) -> keras.model:

def siamese_convolutional_network(input_shape, filters) -> keras.model:

def recurrent_convolutional_network(input_shape, filters) -> keras.model:
```

Idea for later: Since the tile size is rather small we lack spatial context, and do not account for spatial dependence in the predictions. We could feed the model with overlapping tiles of 192x192x3 with a response of 3x3x1, so that each tile would be predicted 9 times depending on the spatial context. Predicted probabilities could then be averaged for each tile.

**Details on `statistics.py`**

This script computes the prediction statistics from the labels and the predictions. The functions should work for a single tile, but can be called on 4D arrays with `map` to produce global prediction statistics.

```python
# statistics.py

def compute_sets(label:np.ndarray, predict:np.ndarray) -> np.ndarray:
	'''Computes TP, TN, FP, FN rasters'''

def compute_statistics(sets:np.ndarray) -> dict:
	'''Computes precision, recall statistics from sets'''
	
def display_statistics(image:np.ndarray, sets:np.ndarray) -> None:
	'''Displays an image with TP, TN, FP, FN'''
```


## Previous scripts > New scripts

**Grouped into `./utilities.py`**

- `./damage/constants.py`
- `./damage/data/reading.py`
- `./damage/data/storing.py`
- `./damage/features/raster_splitter.py`
- `./damage/data/ensemble_splitter.py`
- `./damage/features/raster_pair_maker.py`

**Grouped into `./data.py`**

- `./damage/data/data_sources.py`
- `./damage/features/annotation_preprocessor.py`
- `./damage/features/annotation_maker_fillzeros.py`
- `./damage/features/annotation_maker.py`
- `./scripts/compute_features.py`

**Grouped into `./optimise.py`**

- `./damage/models/base.py`
- `./damage/models/losses.py`
- `./damage/models/cnn.py`
- `./damage/data/data_stream.py` (generators)
- `./damage/models/random_search.py`
- `./scripts/validate.py`
- `./scripts/validate_best_parameters.py`
- `./scripts/validate_with_upsampling.py`

**Grouped into `predict.py`**

- `./scripts/generate_dense_prediction_single_city.py`
- `./scripts/multi_predict.py`

**May be depreciated?**

- `.setup.py`
- `./damage/utils.py`
- `./damage/features/base.py`
- `./damage/features/pipeline.py`
- `./test/utils.py`
- `./test/features/test_annotation_maker.py`
- `./test/features/test_raster_pair_maker.py`

**Depreciated dependencies**

- `sklearn`

**Depreciated folders**

- Is `./syria/data/building damage` the current version?
- Is `./data-old/building damage` depreciated?
- Is `./data/building damage` depreciated?
- Is `./syria/data-old/building damage` depreciated?