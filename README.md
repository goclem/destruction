# Destruction project

## Todo

- Artemisa implementation (Dominik)
- Multi-city training (Clément)

## Experimentations

**Destruction labels**

- Destroyed 3 -> 0.83 test accuracy
- Destroyed 2, 3 -> 90.18 test accuracy
- Destroyed 1, 2, 3 -> 91.75 test accuracy

**Subsamples**

- Keeping the sequences with some destruction -> 0.95 test accuracy

## Proposed changes

**Approach**

- Separate image encoder (foundation model) and sequence encoder
- Combined into a single model
- Predictions use all pre-images with or without labels
- Better domain transfer (e.g. trained on snow)
- Model in PyTorch (i.e more control)

**Image encoder**

- Extract features from individual images at different scales
- Pre-trained ViT (i.e. SatlasPretrain)
- Fine-tuning using contrastive loss?

**Sequence encoder**

- Transformer encoder
- Processes sequences of varying lengths in parallel
- Captures temporal dependence between images and labels
- Multi-head attention using causal mask

**Optimisation**

- Masked cross-entropy loss using only the non-missing labels
- Gradient accumulation to stabilise training
- Use model checkpoints rather than train from scratch

**Data & loaders**

- Format with fast block reading and in-place shuffling
- Saving three files per city containing the train, test, and validation sequences
- Need labels at the same resolution as the image
- Augmentation using subsets of sequences

**Predictions**

- Predict using moving windows
- Estimate the threshold that maximises the F-score

## Questions

- Which summary function for the feature sets at different resolutions?
- How to aggregate buildings with different levels of destruction on a label tile?
- Should destruction 1 and 2 be labelled as 0? Ordinal regression?
- Should the sequence encoder access the post data?
- Should patches overlap for more robust-smooth predictions? Which labels?
- Should we fine-tune the image encoder?

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