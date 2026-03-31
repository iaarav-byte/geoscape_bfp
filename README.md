# Geoscape BFP - Scalable Satellite Segmentation for Swimming Pool Detection

---

## 🚀 Project Overview

This repository contains a **scalable deep learning pipeline** for detecting and segmenting **swimming pools** from high-resolution satellite imagery.

The system combines:

* Patch-based processing for large geospatial data
* Multiple segmentation models (YOLO, U-Net++)
* Foundation model integration (SAM3)
* Knowledge distillation for efficient deployment

---

## 🎯 High-Level Objectives

The training and experimentation pipeline aims to:

1. **Accurately segment swimming pools**

   * Pixel-level boundary detection (not just bounding boxes)

2. **Handle large-scale satellite imagery**

   * Efficient processing of very large TIFF files via tiling

3. **Improve model generalization**

   * Data filtering and resampling strategies
   * Augmentation and hyperparameter tuning

4. **Leverage foundation models**

   * Use SAM3 for pseudo-labeling and feature extraction

5. **Enable scalable inference**

   * Distill heavy models into lightweight student models

---

## 🧑‍💻 Developer Setup

### 📦 Prerequisites

* Python **>= 3.10**
* Linux/macOS recommended
* GPU (recommended for training)

---

### ⚙️ Installation

```bash
# Clone repository
git clone <repository-url>
cd geoscape_bfp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# (Optional) Dev dependencies
pip install -e ".[dev]"
```

---

### 📚 Key Dependencies

* `torch` → Deep learning framework
* `ultralytics` → YOLO models
* `segmentation-models-pytorch` → U-Net++
* `albumentations` → Data augmentation
* `rasterio` → Geospatial raster handling
* `dvc` → Dataset versioning

---

## 📁 Important Project Modules

* `Data_formating/` → Data cleaning, filtering, annotation fixes
* `PatchGen/` → Converts large TIFFs into 512×512 patches
* `Resampling/` → Dataset balancing and resolution alignment
* `Experiment_yolo/` → YOLO training & testing
* `Experiment_Unet++/` → U-Net++ training & testing
* `SAM3/` → Foundation model integration
* `GeoDistill/` → SAM → student distillation
* `GeoProbing/` → Linear probing on SAM embeddings

---

## 🏋️ Training Guide

### 🔹 Main Training Script (YOLO)

```bash
python src/Experiment_yolo/Training/Train30cm_yolo.py
```

---

### 🔧 Common Training Parameters

* `--image_size` → Patch size (default: 512)
* `--epochs` → Number of training epochs
* `--batch_size` → Batch size
* `--output_dir` → Where logs/checkpoints are saved
* `--hpo_trials` → Hyperparameter tuning trials
* `--no_hpo` → Disable hyperparameter tuning

---

### 📌 Example

```bash
python src/Experiment_yolo/Training/Train30cm_yolo.py \
    --epochs 100 \
    --batch_size 8 \
    --image_size 512 \
    --output_dir runs/exp1
```

---

### ⚙️ Hyperparameter Optimization

```bash
python src/Experiment_yolo/Training/HyperParam_train_yolo.py
```

Uses **Optuna** to:

* Tune learning rate
* Optimize augmentation strategies
* Improve convergence

---

## 🔬 Training Workflow (What Happens Internally)

1. **Data Validation**

   * Checks annotation quality (`data_audit.py`)

2. **Pool Filtering**

   * Extracts relevant samples (`filter_pools.py`, `FindPool.py`)

3. **Patch Generation**

   * Converts large TIFFs → 512×512 patches

4. **Resampling**

   * Balances dataset and aligns resolutions

5. **Augmentation**

   * Applied via Albumentations

6. **Model Training**

   * YOLO / U-Net++ / distilled models

7. **Evaluation**

   * Tested on validation/test splits

---

## 📊 Data Used for Training

### 📁 Dataset Characteristics

* **Source**: Geoscape satellite imagery
* **Resolution**: ~30 cm per pixel
* **Type**: RGB aerial imagery
* **Annotations**:

  * Vector polygons (`.gpkg`)
  * Converted to segmentation masks

---

### 🧾 Data Format

| Component | Description                       |
| --------- | --------------------------------- |
| Input     | Satellite image patches (512×512) |
| Labels    | Binary masks (pool vs background) |
| Classes   | 2 (pool, background)              |

---

### 🔄 Data Processing Steps

1. Raw TIFF files
2. Patch extraction (`PatchGen`)
3. Annotation cleaning (`Data_formating`)
4. Pool filtering
5. Mask generation
6. Training-ready dataset

---

### ⚠️ Data Challenges

* Noisy annotations
* Class imbalance (few pools vs large background)
* Small object size

✅ Addressed via:

* Filtering pipelines
* Resampling
* SAM-based pseudo labels

---

## 🧪 Testing & Evaluation

Available in:

```bash
src/Experiment_yolo/Testing/
src/Experiment_Unet++/Testing/
```

---

### Metrics

* IoU (Intersection over Union)
* Pixel accuracy
* (Planned) Boundary metrics:

  * Hausdorff Distance
  * Chamfer Distance

---

## ⚠️ Known Challenges

* SAM is computationally heavy → requires distillation
* Patch boundaries can create artifacts
* Domain gap between natural and satellite images
* Limited labeled data

---

## 🔮 Future Improvements

* Shape-aware loss functions
* Topology-aware metrics
* Multi-class segmentation
* Real-time inference pipeline

---

## 🤝 Contributing

* Follow modular structure in `src/`
* Keep experiments reproducible
* Document any new pipelines

---

## 📄 License

[Add license here]

---

## 💡 Final Note

This project goes beyond standard model training — it is a **complete geospatial ML system**, combining:

* Data engineering
* Deep learning
* Foundation models
* Scalable deployment

---
