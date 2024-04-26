# Destruction project

## To do

- Artemisa implementation (Dominik)

## Questions

**Data**

- How to aggregate buildings with different levels of destruction on a label tile?
- Should destruction 1 and 2 be labelled as 0?
- Should patches overlap for more robust-smooth predictions?

**Model**

- Should the sequence encoder access the post data?
- Should the ourput layer be an ordinal regression?

**Training**

- Should we fine-tune the image encoder?
- Should the sequence encoder be trained directly in the embedding space?

## Experiments

Experiments performed on Aleppo

**Destruction labels**

| Destroyed | Test Accuracy |
|-----------|---------------|
| 3         | 83.00         |
| 2, 3      | 90.18         |
| 1, 2, 3   | 91.75         |

**Subsamples**

- Keeping the sequences with some destruction &rarr; 0.95 test accuracy
- Does not generalise to full sample

**Focal loss**

- Focal loss + subsampling &rarr; decision threshold 0.5

## Proposed changes

**Model**

- PyTorch implementation, more training control
- Modular architecture with independent components 
	- Image encoder (n x t x d x h x w &rarr; n x t x k)
	- Sequence encoder (n x t x k  &rarr; n x t x k)
	- Classification head (n x t x k &rarr; n x t x 1)
- Pre-trained foundation model (i.e. SatlasPretrain) as image encoder
	- Domain transfer for HR aerial images (e.g. trained on snow)
	- Extract images features at different scales
	- Decreased computational cost, fine-tuning optionnal
	- Parametrised dimensionality reduction
- Transformer as sequence encoder
	- Processes sequences of varying lengths
	- Captures temporal dependence between images and labels
	- Multi-head attention using causal mask

**Optimisation**

- Masked focal cross-entropy 
	- Training using entire image sequences and the non-missing labels
	- Manages class imbalances and weights difficult examples
- Gradient accumulation to stabilise training
- Model checkpoints to avoid training from scratch

**Data & loaders**

- One zarr file per
	- City e.g. Aleppo
	- Sample i.e. train, valid, test
	- Dataset i.e. images, labels
- Data loader using multiple city datasets
- Shuffle training and validation datasets on epoch end

**Predictions**

- Predict using moving windows
- Estimate the threshold that maximises the F-score

## Resources

**Combine CNN and RNN**

- [Chavhan et al. 2023](https://ieeexplore.ieee.org/document/10192592)
- [Brock Abdallah 2023](https://arxiv.org/pdf/2204.08461.pdf)

**Transfer learning**

- [SATMAE](https://github.com/sustainlab-group/SatMAE?tab=readme-ov-file)
- [BigEarthNet](https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models)
- [SatlasPretrain](https://github.com/allenai/satlaspretrain_models?tab=readme-ov-file)
- [PRESTO](https://arxiv.org/pdf/2304.14065.pdf)
- [SITS-Former](https://www.sciencedirect.com/science/article/pii/S0303243421003585)

**Other**

- [Conflict Ecology Lab](https://www.conflict-ecology.org/research)