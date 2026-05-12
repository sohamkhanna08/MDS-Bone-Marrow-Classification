# Interpretable Deep Learning for Dysplastic Cell Classification in Bone Marrow Smears

> Binary classification of Myelodysplastic Syndrome (MDS) from bone marrow cell images using DenseNet121 with Squeeze-and-Excitation attention and Grad-CAM interpretability.

---

## Overview

Myelodysplastic Syndromes (MDS) are clonal hematological disorders characterised by ineffective haematopoiesis and abnormal bone marrow cell morphology. Morphological assessment of bone marrow smears remains the clinical standard, yet it is inherently subjective and prone to inter-observer variability.

This project develops an interpretable deep learning pipeline for automated binary classification of bone marrow cells into **Normal** and **Abnormal (Dysplastic)** categories. The model combines transfer learning on DenseNet121 with a channel-attention mechanism, domain-specific preprocessing, and Grad-CAM visualisation to improve both performance and clinical trustworthiness.

---

## Results

Evaluation was performed on a **balanced held-out test set** (400 Normal + 300 Abnormal) to provide a fair assessment under controlled class distribution.

| Metric | Value |
|---|---|
| Accuracy | **85.14%** |
| AUC-ROC | **0.9296** |
| F1 — Normal | 0.8802 |
| F1 — Abnormal | 0.8045 |
| F2 Score | 0.7472 |
| Median Score Separation | 0.9334 |
| Decision Threshold | 0.7026 |

> The AUC-ROC of **0.9296** reflects strong discriminative capacity between normal and dysplastic cells despite significant morphological overlap.

### Per-Class Accuracy (Balanced Test Set)

| Cell Type | N | Label | Mean Prob | Accuracy |
|---|---|---|---|---|
| Blast, NOC | 3 | AB | 0.983 | **1.000** |
| Micromegakaryocyte | 28 | AB | 0.950 | **0.964** |
| Small megakaryocyte | 41 | AB | 0.910 | **0.878** |
| Myeloblast | 149 | AB | 0.716 | 0.685 |
| Dysplastic erythroblast | 50 | AB | 0.616 | 0.580 |
| Dysplastic granulocyte | 26 | AB | 0.586 | 0.577 |
| Segmented neutrophil | 38 | NL | 0.031 | **1.000** |
| Late erythroblast | 77 | NL | 0.041 | **0.987** |
| Band neutrophil | 48 | NL | 0.048 | **0.979** |
| Mature lymphocyte | 86 | NL | 0.150 | 0.953 |
| Intermediate erythroblast | 44 | NL | 0.119 | 0.955 |
| Monocyte | 16 | NL | 0.113 | 0.938 |
| Promyelocyte | 9 | NL | 0.108 | **1.000** |

Difficult cases — Dysplastic granulocytes and erythroblasts — show lower accuracy due to morphological similarity with normal counterparts, consistent with challenges faced by human morphologists.

---

## Dataset
 
The dataset is sourced from the publicly available bone marrow cell image repository introduced by **Shen et al. (2025)**:
 
> D. Shen et al., "A large dataset of bone marrow cells in myelodysplastic syndrome for classification systems," *Scientific Data*, vol. 12, p. 1849, 2025. DOI: [10.1038/s41597-025-06027-z](https://doi.org/10.1038/s41597-025-06027-z)
 
### Summary
 
| Split | Count |
|---|---|
| Total images (after exclusions) | 20,572 |
| Normal (label 0) | 18,451 |
| Abnormal (label 1) | 2,121 |
| Class imbalance ratio | ~8.7 : 1 |
 
Ambiguous categories — **Smudge Cell** and **Unclassified Cell** — were excluded from the analysis. Dataset split: **70 / 15 / 15** (train / val / test), stratified at cell-type level.
 
### Per-Class Distribution
 
| Cell Type | Label | Count |
|---|---|---|
| Blast, NOC | 1 | 22 |
| Dysplastic erythroblast | 1 | 354 |
| Dysplastic granulocyte | 1 | 178 |
| Dysplastic megakaryocyte | 1 | 1 |
| Megaloblastic late erythroblast | 1 | 4 |
| Micromegakaryocyte | 1 | 210 |
| Monoblast | 1 | 19 |
| Myeloblast | 1 | 1,045 |
| Small megakaryocyte | 1 | 288 |
| Band eosinophil | 0 | 59 |
| Band neutrophil | 0 | 2,208 |
| Early erythroblast | 0 | 740 |
| Eosinophilic metamyelocyte | 0 | 148 |
| Eosinophilic myelocyte | 0 | 45 |
| Histiocyte | 0 | 3 |
| Intermediate erythroblast | 0 | 2,084 |
| Late erythroblast | 0 | 3,555 |
| Mature lymphocyte | 0 | 3,958 |
| Megakaryocyte | 0 | 11 |
| Mitosis | 0 | 123 |
| Monocyte | 0 | 734 |
| Neutrophilic metamyelocyte | 0 | 1,291 |
| Neutrophilic myelocyte | 0 | 828 |
| Plasma cell | 0 | 228 |
| Proerythroblast | 0 | 47 |
| Promonocyte | 0 | 33 |
| Promyelocyte | 0 | 429 |
| Segmented basophil | 0 | 88 |
| Segmented eosinophil | 0 | 96 |
| Segmented neutrophil | 0 | 1,743 |
 
### Label Mapping
 
```python
BINARY_MAPPING = {
    'Blast, NOC'                     : 1,
    'Myeloblast'                     : 1,
    'Monoblast'                      : 1,
    'Dysplastic granulocyte'         : 1,
    'Dysplastic erythroblast'        : 1,
    'Dysplastic megakaryocyte'       : 1,
    'Micromegakaryocyte'             : 1,
    'Small megakaryocyte'            : 1,
    'Megaloblastic late erythroblast': 1,
    'Promonocyte'                    : 0,
    'Promyelocyte'                   : 0,
    'Band neutrophil'                : 0,
    'Band eosinophil'                : 0,
    'Early erythroblast'             : 0,
    'Intermediate erythroblast'      : 0,
    'Late erythroblast'              : 0,
    'Proerythroblast'                : 0,
    'Neutrophilic myelocyte'         : 0,
    'Neutrophilic metamyelocyte'     : 0,
    'Eosinophilic myelocyte'         : 0,
    'Eosinophilic metamyelocyte'     : 0,
    'Monocyte'                       : 0,
    'Mature lymphocyte'              : 0,
    'Plasma cell'                    : 0,
    'Histiocyte'                     : 0,
    'Megakaryocyte'                  : 0,
    'Segmented neutrophil'           : 0,
    'Segmented eosinophil'           : 0,
    'Segmented basophil'             : 0,
    'Mitosis'                        : 0,
}
 
EXCLUDED_CLASSES = {'Smudge cell', 'Unclassified cell'}
```
 
---
 

## Methodology

### Preprocessing Pipeline

**1. Stain Normalisation**
Reinhard colour normalisation applied in CIE LAB colour space. A reference image is selected as the sample with minimum mean Euclidean distance to the centroid of the LAB distribution computed over 20 randomly drawn training candidates. For each channel:

$$I_c^{\text{norm}} = \left(\frac{I_c - \mu_c^{\text{src}}}{\sigma_c^{\text{src}}}\right)\sigma_c^{\text{ref}} + \mu_c^{\text{ref}}$$

Reinhard normalisation was preferred over Macenko stain decomposition due to numerical instability of the underlying SVD on sparse-cell images with rank-deficient stain matrices.

**2. Region-of-Interest (ROI) Extraction**
- HSV saturation channel thresholded using Otsu's global method to generate a binary cell mask
- Largest connected component identified as the primary cell body
- Bounding square expanded by 10% margin per side; constant white padding applied where crop falls outside image boundary
- If detected cell area < 25% of total image area, thresholding retried at progressively reduced threshold to recover sparsely stained cells
- All images resized to **224 × 224** pixels and normalised using ImageNet channel statistics

**3. Background Ablation**
During initial training, pixels outside the segmented cell region are replaced with a neutral background value for a subset of samples, reducing reliance on slide-level artifacts and spurious correlations.

---

### Model Architecture

| Layer | Output Shape | Parameters |
|---|---|---|
| DenseNet121 Backbone (ImageNet) | (7, 7, 1024) | 7,037,504 |
| SE Block (Channel Attention, r=16) | (7, 7, 1024) | 131,200 |
| Global Average Pooling | (1024,) | 0 |
| Dropout (p=0.5) | (1024,) | 0 |
| Dense (256, ReLU, L2 λ=1e-3) | (256,) | 262,400 |
| Dropout (p=0.4) | (256,) | 0 |
| Dense (1, Sigmoid) | (1,) | 257 |

**DenseNet121 Backbone**: Each layer $l$ receives feature maps from all preceding layers via direct concatenation:

$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1}])$$

This promotes feature reuse across depth and mitigates vanishing gradients — particularly beneficial on limited medical imaging datasets.

**Squeeze-and-Excitation Block**: Channel-wise recalibration applied after the backbone output. Global average pooling produces a channel descriptor, passed through two FC layers (bottleneck ratio r=16: 1024→64→1024) with ReLU and Sigmoid activations respectively. The resulting attention weights multiplicatively gate the feature map, emphasising morphologically discriminative channels (nuclear chromatin texture, nuclear-to-cytoplasmic ratio).

---

### Training Strategy

**Phase 1 — Partial Freeze**
- Layers 0–199 of DenseNet121 frozen; layers 200+ trainable
- All Batch Normalisation layers kept trainable throughout to accommodate domain shift
- Optimiser: Adam, lr = 5×10⁻⁵
- Loss: Binary Cross-Entropy with balanced batch sampling
- Background ablation applied to a subset of training samples

**Phase 2 — Full Fine-Tuning**
- Best checkpoint from Phase 1 reloaded; all layers unfrozen
- Reduced learning rate: 1×10⁻⁵ to prevent destabilisation of learned feature hierarchy
- Background ablation disabled for cleaner gradient signal
- Class imbalance penalty relaxed relative to Phase 1

This staged strategy allows the model to first establish stable task-specific representations under partial freezing before globally refining all parameters.

---

### Interpretability — Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is applied to evaluate model attention alignment. Visualisations confirm that correct predictions correspond to nucleus-focused attention, while failure cases frequently exhibit attention leakage to cytoplasm or background regions. This analysis guides both model diagnostics and future preprocessing design.

---

## Repository Structure

```
├── MDS_BoneMarrow_Classification.ipynb   # Full pipeline: preprocessing, training, evaluation, Grad-CAM
├── outputs/
│   ├── balanced_test_evaluation.png
│   ├── balanced_test_per_class.csv
│   ├── bal_test_probs.npy
│   ├── bal_test_labels.npy
│   └── balanced_test_df.csv
└── README.md
```

---

## Requirements

```
tensorflow >= 2.12
numpy
opencv-python
scikit-learn
matplotlib
scipy
```

---


## Authors

**Soham Khanna** — University School of Automation and Robotics, Guru Gobind Singh Indraprastha University, New Delhi, India


## License

This project is for academic and research purposes. The dataset is subject to its original licensing terms as specified by Shen et al. (2025).
