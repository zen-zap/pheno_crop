# Crop Growth Stage Detection from Satellite Imagery

Sentinel-1 and Sentinel-2 are sourced from Google Earth Engine. The scripts for fetching them can be found in `fetch_data.ipynb` at project root.

The primary challenge in remote sensing lies in the nature of the data itself: agricultural environments are highly dynamic, and the satellite telemetry used to monitor them is inherently asynchronous and noisy. To capture the full spectrum of a winter wheat crop's growth cycle (Stages 0 through 4), this architecture relies on two independent data streams extracted via Google Earth Engine.

---

## Model 1: Dual Stream Hybrid Network

**Architecture Overview:**

- The data is passed through an async pipeline for pre-processing. It outputs Optical and Radar data. See `model_1.ipynb` Model 1 for details.
- The Optical network consists of a 1D CNN with time embedding and a CLS token with Multi-Head Attention.
- The Radar network consists of a 1D CNN with time embedding processed with a GRU.
- The outputs from both the networks are then concatenated (they run in parallel) in a fusion network for the final prediction.

### The Dual-Sensor Approach

The first stream is Sentinel-2 optical imagery, fetched via a custom automated pipeline that calculates specific vegetation indices such as NDVI, NDWI, NDRE, EVI, and SAVI at a 10-meter spatial resolution. While rich in chemical information regarding chlorophyll and leaf water content, this optical data is highly susceptible to atmospheric occlusion and cloud cover.

To counterbalance this, the pipeline integrates a second stream: Sentinel-1 Synthetic Aperture Radar (SAR). By capturing the Vertical-Vertical (VV) and Vertical-Horizontal (VH) backscatter, along with their derived ratio, the radar stream provides continuous, cloud-penetrating measurements of the physical canopy structure and biomass density.

### Asynchronous Temporal Alignment

To bridge the gap between this raw, asynchronous satellite telemetry and the rigid input requirements of standard neural networks, the data undergoes a temporal alignment process within the dataset generation phase. Traditional machine learning models often force multi-sensor data onto a shared, interpolated daily grid, which inevitably introduces hallucinated biological data during long cloudy periods.

Instead, this system utilizes a fully asynchronous timeline anchored directly to the ground truth observation, designated as Day 0. The pipeline looks backward over a 90-day window, extracting a maximum of 30 valid satellite events for each modality. Rather than interpolating, the system calculates a strict relative temporal distance, represented as a discrete `days_ago` integer, for every single reading.

Consequently, the dataset yields two entirely separate sets of tensors for every agricultural plot:

- A feature tensor containing the sensor values.
- A parallel time tensor containing the exact chronological placement of those values.

### The Optical Branch (Chemistry & Filtering)

Operating on this aligned temporal data, the first half of the architecture focuses on the chemical and biological signatures via the Optical Attention Branch.

The network ingests a tensor of shape $[\text{Batch}, 30, 6]$, representing the sequence of optical indices alongside the critical `cloud_pct` metric. In classical time-series analysis, researchers manually calculate temporal derivatives, such as the slope of NDVI, to determine how rapidly a crop is greening up. This approach is brittle and highly susceptible to sensor noise.

Instead, this architecture natively integrates a 1-Dimensional Convolutional Neural Network (1D CNN) with a sliding window of size 3 as its front-end feature extractor. As this convolutional window sweeps across the 30-day temporal sequence, it dynamically calculates local derivatives, smoothing minor atmospheric noise and extracting the biological momentum of the indices into a richer 64-dimensional latent space.

To ensure the network understands the massive, irregular gaps between these clear readings, the integer time tensor is passed through an $\text{Embedding}(120, 64)$ layer, generating a unique mathematical signature for the exact day the reading occurred. This temporal signature is added directly to the CNN output.

A learnable 64-dimensional classification token is then prepended to the sequence, expanding the temporal length to 31. This sequence is processed by a deep Transformer Encoder utilizing Multi-Head Attention. By analyzing the entire 90-day timeline simultaneously, the attention mechanism acts as a dynamic biological filter. When it encounters an event with a high cloud percentage and a consequently crashed NDVI, it mathematically drops the attention weight for that specific day to near zero.

The network actively routes around atmospheric corruption, mapping the long-term biological relationships of the clear days and compressing that knowledge entirely into the classification token, which serves as the final 64-dimensional optical summary vector.

> **Context Note:** The "Transformer Encoder" acts as a smart filter. Instead of stepping through the timeline day-by-day, it looks at the entire 3-month history at once. It learns to automatically mute data points that are corrupted by clouds, ensuring only clean biological trends are passed forward.

### The Radar Branch (Structure & Continuity)

In stark contrast to the optical stream, the second half of the network processes the Sentinel-1 SAR telemetry through a Recurrent Radar Branch.

The input here takes the shape of $[\text{Batch}, 30, 3]$, capturing the VV, VH, and VH/VV ratio. Because SAR data is inherently plagued by microwave interference known as speckle, a parallel 1D CNN acts as a learned speckle filter. It identifies short-term structural shapes, such as the steep drop in VH backscatter that reliably occurs just before harvest as the stalks dry out and lose physical volume.

After being fused with an identical 64-dimensional positional time embedding, the sequence is fed sequentially into a Gated Recurrent Unit (GRU) with 2 hidden layers. Because radar microwaves effortlessly punch through cloud cover, the Sentinel-1 timeline is highly continuous and rarely features the massive temporal gaps seen in the optical stream.

This continuous nature makes a recurrent memory network the mathematically optimal choice, allowing the GRU to step chronologically through the days, smoothly updating its internal hidden state to track the physical accumulation and eventual dry-down of the crop canopy.

The final hidden state is extracted as the 64-dimensional radar summary vector.

### Late-Fusion and Regularization

With the biological chemistry and physical structure independently encoded, the network employs a late-fusion strategy to make its final prediction. The optical and radar summary vectors are concatenated into a unified 128-dimensional multi-modal tensor.

This delayed integration is a critical architectural choice; by forcing the network to understand the optical curve and the radar curve entirely independently before combining them, the system prevents a highly noisy optical reading from preemptively corrupting the continuous radar memory state.

The fused manifold is finally passed through a Multi-Layer Perceptron classifier. This classifier scales the features down to 64 dimensions before outputting the final 5 class logits.

Crucially, this block utilizes Batch Normalization and a Dropout probability of 30 percent. This heavy regularization actively prevents the massive capacity of the dual-stream network from simply memorizing the specific layouts of the training plots, forcing it to generalize the underlying biophysical rules of crop phenology.

### Architectural Defenses and Physics Constraints

The structural integrity of this entire pipeline relies heavily on its ability to mathematically mirror physical realities, a trait that directly answers the most common technical critiques in satellite machine learning.

For instance, differing acquisition dates between sensors are handled flawlessly precisely because the sensors are never artificially forced onto the same temporal grid; the asynchronous embeddings ensure that an optical reading from 14 days ago and a radar reading from 12 days ago are treated as distinct physical events.

Furthermore, SAR backscatter is highly sensitive to the satellite's specific viewing geometry, particularly the incident angle and whether the orbit pass is ascending or descending. This architecture actively mitigates these geometry-induced variations by utilizing the VH/VV ratio as a core input feature, which mathematically cancels out much of the background soil moisture and baseline structural variation. Coupled with the local smoothing of the front-end CNN, this prevents the recurrent network from overreacting to sudden backscatter jumps caused purely by alternating orbit directions.

Finally, while lighter architectures like the Lightweight Temporal Attention Encoder rely on a single master query to extract temporal features, this system utilizes a full self-attention stack. By allowing every temporal observation to mathematically attend to every other observation within the lookback window, the network captures much deeper, non-linear chronological relationships, such as explicitly linking the exact day of peak vegetative green-up to the exact day the harvest senescence begins.

---

## Results

**Confusion Matrix:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/96b4a82a-cf53-4994-9064-2d371b3818cf" />

**Latent Space:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/64d13128-29eb-44d5-a677-4ca0bfae8dc6" />

---
<br>

> **Important Note on Above Metrics:** Some currently reported scores are likely inflated because of evaluation leakage risks, including event-level random splitting (instead of strict plot-level/group splits), temporal overlap within the same plot context between train/validation samples, and running inference over the full dataset used during model development. Treat those numbers as exploratory, not final generalization performance. Work on this is here: `model_2.ipynb`

## Model 2: Leakage-Aware Dual-Stream Network

**Architecture Overview:**

- Model 2 keeps the same multi-modal philosophy as Model 1, but restructures training and evaluation around strict plot-level isolation.
- The input pipeline builds two asynchronous sequences per event over a 90-day lookback, capped to 30 observations.
- The Optical branch uses a CNN + Transformer encoder stack with learned time embeddings and masked pooling.
- The Radar branch uses a CNN + bidirectional GRU stack with masked pooling and linear projection.
- The two modality embeddings are concatenated and passed through a regularized MLP classifier for 5-stage prediction.

### Leakage-Aware Data Protocol

Model 2 is built to reduce overly optimistic metrics caused by split leakage. Instead of random event-level splitting, the dataset is partitioned by `plot_id` using group-aware splitting logic. A plot can only appear in exactly one of train, validation, or test.

This structure directly targets the most common failure mode in agricultural time-series experiments, where near-identical temporal neighborhoods from the same field leak across splits. By forcing disjoint plot groups, Model 2 evaluates generalization on genuinely unseen fields rather than memorized local patterns.

### Temporal Representation

For each ground-truth event, the data loader creates separate Sentinel-1 and Sentinel-2 sequences anchored to the target date. Each observation is encoded with a relative `days_ago` index, preserving asynchronous sampling without interpolation.

The sequence builder also uses explicit masks to distinguish valid observations from padded positions. This allows downstream temporal layers to pool only real events and ignore synthetic padding.

### Optical Branch (Sentinel-2)

The optical encoder ingests a 9-feature tensor per time step: vegetation indices and cloud percentage metadata. A two-layer 1D CNN first extracts short-range temporal patterns and local trend structure.

The projected features are combined with a learned temporal embedding (`Embedding(120, 64)`) derived from `days_ago`, then passed through a 2-layer Transformer encoder (4 heads, feedforward 128, dropout 0.25).

The output is summarized with masked mean pooling, producing a fixed-size optical embedding that focuses on valid timestamps only.

### Radar Branch (Sentinel-1)

The radar encoder processes VV, VH, and VH/VV ratio through a two-layer 1D CNN front-end, then models sequence dynamics with a 2-layer bidirectional GRU.

As with the optical stream, masked mean pooling extracts a robust sequence summary from valid positions. A linear projection then maps the bidirectional GRU output back into a compact modality embedding.

### Fusion and Classifier Head

The optical and radar embeddings are concatenated into a fused feature vector and fed into a classifier head:

- Linear(128 -> 96)
- BatchNorm1d(96)
- ReLU
- Dropout(0.4)
- Linear(96 -> 5)

This late-fusion design preserves modality-specific temporal modeling before decision-level integration.

### Training Strategy

Model 2 uses class-balanced cross-entropy with label smoothing (`0.05`), AdamW optimization, ReduceLROnPlateau scheduling on validation macro-F1, and early stopping.

Validation and test reporting prioritize macro-F1 and per-class recall, alongside confusion matrices and detailed classification reports. This is better aligned with imbalanced stage distributions than relying on accuracy alone.

### Why Model 2 Is More Reliable

Model 2 improves trustworthiness through:

- strict group-based train/val/test partitioning by plot,
- explicit padded-step masking in both branches,
- stronger regularization in the classifier,
- evaluation centered on macro-F1 and class-wise behavior.

The result is a cleaner estimate of field-level generalization while preserving the dual-sensor architecture that performed well in this project.

## Results: 

- **Validation Accuracy**: 70.52% | **Macro-F1**: 0.6923

- **Test Accuracy**: 70.83% | **Macro-F1**: 0.6929

```
[Test (Unseen Plots)] Loss: 1.1031 | Accuracy: 70.83% | Macro-F1: 0.6929
Per-class recall:
  [0] Bare: 0.7741
  [1] Growth: 0.6705
  [2] Ripening: 0.7424
  [3] Seedling: 0.6426
  [4] Tillering: 0.7538
              precision    recall  f1-score   support

        Bare       0.28      0.77      0.41       332
      Growth       0.83      0.67      0.74       786
    Ripening       0.75      0.74      0.75       427
    Seedling       0.87      0.64      0.74       887
   Tillering       0.92      0.75      0.83      1133

    accuracy                           0.71      3565
   macro avg       0.73      0.72      0.69      3565
weighted avg       0.81      0.71      0.74      3565
```

**Confusion Matrix:**

<img width="658" height="586" alt="image" src="https://github.com/user-attachments/assets/b154a55f-a425-4a2d-88c0-2ca4cac3873b" />


<br>

---

## References

- [Temporal Self Attention and Multi-Sensor Fusion](https://github.com/ellaampy/CropTypeMapping)
- [Cloud Imputation - SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/)
- [U-Net with Temporal Attention Encoder](https://github.com/Many98/Crop2Seg)
- [Implementations of Cross-Attention Layers](https://github.com/likyoo/awesome-multimodal-remote-sensing-classification)
- [L-TAE: Lightweight Temporal Attention Encoder](https://arxiv.org/abs/2007.00586)
