# DualStream-DFormer: Multi-Modal Semantic Segmentation

**DualStream-DFormer** is a multi-modal segmentation framework designed to fuse high-resolution satellite imagery (Vision) with multi-spectral air quality data (GEMS/Sensors).

By leveraging an **asymmetric dual-stream encoder** and a **hierarchical fusion mechanism**

## 🏆 Results (Epoch 40)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **mIoU** | **99.24%** | Mean Intersection over Union |
| **Weighted Ensenble** | **99.31%** | Mean Intersection over Union with Weighted Ensenble |
| **Val Loss** | 0.0171 | Final Validation Loss |
| **Train Loss**| 0.0099 | Final Training Loss |

## 📚 Dataset

This project utilizes the **"Satellite Image-based Air Quality Analysis Data"** provided by **AI Hub**.
* **Source:** [AI Hub - Land Cover & Air Quality Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71805)
* **Modalities:**
    * **Vision:** High-resolution Satellite Imagery ($512 \times 512$)
    * **Air:** GEMS Satellite Data + Ground Stations ($64 \times 64 \times 12*4$) (GEMS/NO2/SO2/CO)

## 🏗️ System Architecture

The architecture addresses the resolution mismatch between vision and air quality data through a multi-stage fusion strategy.

### 1. Asymmetric Dual-Stream Backbone
* **Vision Stream (SegFormer-B1):** Captures hierarchical semantic features from RGB images.
* **Air Stream (Custom ConvNeXt):** Processes high-dimensional (48-channel) environmental data.

### 2. Hierarchical Feature Fusion
* **Stage 1-2 (Local Fusion):** Utilizes **FiLM (Feature-wise Linear Modulation)** and **Dynamic Gating** to modulate visual features based on environmental embeddings.
* **Stage 3-4 (Global Fusion):** Employs **Cross-Attention Blocks** to model long-range dependencies between visual queries and air quality keys/values.

### 3. Decoder & Refinement
Standard bilinear upsampling is insufficient for high-precision tasks. We implement a robust decoder:
* **ASPP:** Aggregates multi-scale context.
* **Attention Gates:** Filters skip connections to suppress noise.
* **Learnable Upsampling:** Instead of simple interpolation, **ConvNeXt-based Refinement Blocks** are used at each upsampling stage to reconstruct fine boundaries.

## 📂 Directory Structure

Standardized data loader structure required for `DualStreamDataset`:

```plaintext
/content/
├── ts_sn/      # Train Images (Satellite)
├── tl_sn/      # Train Labels (Masks)
├── t_ap/       # Train Air Pollution Data (TIF)
├── t_gems/     # Train GEMS Data (TIF)
├── vs_sn/      # Validation Images
├── vl_sn/      # Validation Labels
├── v_ap/       # Validation Air Pollution Data
└── v_gems/     # Validation GEMS Data
```

## ⚙️ Configuration

Key hyperparameters are defined in the main script:

```python
LEARNING_RATE = 3.0e-4 # Decayed to 1.0e-6 by Epoch 40
BATCH_SIZE = 4
EPOCHS = 40
TARGET_SIZE = 512
NUM_CLASSES = 2 (Background / Target)

# Loss Weights
FOCAL_WEIGHT = 0.5
LOVASZ_WEIGHT = 0.5
AUX_LOSS_WEIGHTS = {'final': 1.0, 'f2': 0.3, 'f3': 0.15}
```

Hardware: Tesla T4 GPU (Google Colab)
