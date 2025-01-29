# Destruction project

## To do

- Artemisa implementation (Dominik)
- Add relevant classification metrics (Clément)

## Questions

## **General Project Structure**  

Our model follows a three-step routine:  
1. **Pretraining:** Train an encoder-decoder model to reconstruct images (*image → image*).  
2. **Fine-tuning:** Adapt the encoder to predict destruction labels (*image → destruction label*).  
3. **Sequence Encoding:** Apply a sequence encoder on top of encoded image sequences (*sequence of images → sequence of labels*).  

---  

## **Dataloader Workflow**  

1. Compute the probability for each city based on the share of tiles per city.  
2. Using these probabilities, randomly sample the number of tiles per city and batch, ensuring the total number of tiles matches the batch size.  
3. Generate a large list of city combinations for multiple batches.  
4. Define slice indices for each batch.  
5. Stop when the batch slicing index reaches the tile limit for the first city.  

### **Batch Creation & Shuffling**  
- **Batch Composition:** Assemble batches according to the number of tiles per city from the generated list.  
- **Shuffling:** Previously, images and labels were shuffled separately. This has now been corrected to ensure proper alignment.  

---  

## **Problem: Accuracy Improves, but AUC Remains at 0.5 (Despite Balanced Samples)**  

### **Investigation & Findings**  

#### **Dataset Analysis**  
- **Resizing issue when reshaping from (n × t × c × h × w) to ((n × t) × c × h × w):** ❌ No problem found.  
- **Data type issue:** ❌ No problem found.  
- **Class balance:** Initially, the dataset had a 2:1 ratio instead of being fully balanced.  
- **Data quality:** Some tiles are assigned a label of 1 in ways that are difficult to interpret. Overall, the dataset is not very clean.  

#### **Routine Step 1: Pretraining**  
- **Encoder-decoder performance:** The decoded images roughly resemble the original images, indicating the encoder is functioning as expected.  

#### **Routine Step 2: Fine-tuning for Destruction Prediction**  
- **Dataloader loading only 0s or 1s?** ❌ No, both labels (0 and 1) are present.  
- **Shuffling issue?** ❌ No, originally, images and labels were shuffled separately, but this has been corrected.  
- **Model predicting only 0s or 1s?** ❌ No, the model predicts values around 0.4 but varies across different tiles.  
- **Metric issue?** ❌ No issue detected with the metric calculation.  

### **Root Causes Identified**  
1. **Shuffling issue:** Previously, images and labels were shuffled separately. This has now been fixed.  
2. **Frozen encoder weights:** The encoder remained frozen during tile-level classification, preventing feature adaptation for destruction prediction.  

**Note:** Early stopping is currently based on **loss**, not AUC.  

---  

## **Next Steps**  

- **Clement:** Update the dataset to include both balanced and unbalanced versions.  
- **Dominik:** Implement fine-tuning at the tile level.  
- **Next Phase:** Once initial results are available, proceed with fine-tuning for sequence classification.

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