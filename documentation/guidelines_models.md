# Models guidelines

Updated on 2022.05.09 by Clément

**Data format**

Storing image data in a 5-diemnsional array is flexible and can be used for all models structures with a single reshape operation. This format suits the recurrent structure, and flattening the `n x t` dimensions suits the simple convolutional network, shifting the `t` dimension suits the siamese network.

- Images as numpy array of dimensions `n x t x h x w x d`
- Labels as numpy array are of dimensions `n x t`
- Tiles outside the analysis zone are removed

**Samples**

- Split entire tile sequences:
	- Training sample (70%)
	- Validation sample (15%)
	- Test sample (15%)
	- 
**Filling labels (?)**

- Confirm that only label `3` are considered as destroyed
- Tile labelled as `t1=0` and `t3=1` e.g. `?|0|?|1|?|...`
- The imputation outputs `0|0|!0|1|1|...` although we don't know whether the building is destroyed in `t2`. The same applies to gaps of any length.
