# Duality AI Segmentation - IoU Improvement Pipeline

## 🎯 Project Overview

This project implements a comprehensive IoU (Intersection over Union) improvement pipeline for semantic segmentation of off-road terrain using DINOv2 backbone and ConvNeXt-style segmentation head.

### Current Performance
- **Baseline Mean IoU**: 0.2794 (27.94%)
- **Target Mean IoU**: 0.5-0.6 (50-60%)
- **Status**: ✅ Training in progress with optimized pipeline

---

## 🚀 Quick Start

### Run IoU Improvement Training
```bash
python run_iou_boost.py
```

This will:
1. Compute class weights from training data
2. Create optimized data loaders
3. Build DINOv2 + ConvNeXt model
4. Train for 15 epochs with class-weighted loss
5. Save best model to `iou_boost_output/best_model.pth`

### Evaluate Current Model
```bash
# Quick evaluation (200 samples)
python compute_quick_iou.py

# Full evaluation (all samples)
python compute_current_iou.py
```

---

## 📊 Model Architecture

### Backbone: DINOv2-ViT-Small
- **Parameters**: 22M (frozen)
- **Embedding Dimension**: 384
- **Patch Size**: 14×14
- **Input Resolution**: 266×476 (aligned to patches)

### Segmentation Head: ConvNeXt-Style Decoder
- **Parameters**: 2.4M (trainable)
- **Architecture**:
  - 7×7 stem convolution (384 → 128 channels)
  - Depthwise separable ConvNeXt block
  - 1×1 classifier (128 → 11 classes)

### Total Model
- **Total Parameters**: 24.5M
- **Trainable Parameters**: 2.4M (10%)
- **Efficient Transfer Learning**: Only decoder is trained

---

## 🎓 Training Configuration

### Hyperparameters
```python
Optimizer: AdamW
Learning Rate: 0.001
Weight Decay: 0.01
Scheduler: CosineAnnealingLR
Batch Size: 4
Epochs: 15
Loss: CrossEntropyLoss with class weights
```

### Class Weights (Inverse Frequency)
- **Rare Classes** (Background, Flowers, Logs): 3.67x weight
- **Common Classes** (Trees, Bushes, Grass, etc.): 0.10x weight
- Addresses severe class imbalance in dataset

---

## 📈 Performance Improvements

### Baseline Performance (Before)
| Class | Name | IoU | Status |
|-------|------|-----|--------|
| 0 | Background | 0.0000 | ⚠️ POOR |
| 1 | Trees | 0.4343 | ○ OK |
| 2 | Lush Bushes | 0.4187 | ○ OK |
| 3 | Dry Grass | 0.4797 | ○ OK |
| 4 | Dry Bushes | 0.0134 | ⚠️ POOR |
| 5 | Ground Clutter | 0.0014 | ⚠️ POOR |
| 6 | Flowers | 0.1923 | ⚠️ POOR |
| 7 | Logs | 0.0021 | ⚠️ POOR |
| 8 | Rocks | 0.0194 | ⚠️ POOR |
| 9 | Landscape | 0.5806 | ○ OK |
| 10 | Sky | 0.9316 | ✓ GOOD |

**Mean IoU**: 0.2794 (27.94%)

### Expected Performance (After Training)
- **Mean IoU**: 0.5-0.6 (50-60%) - **2x improvement**
- **Poorly performing classes**: Boosted from <0.02 to 0.30-0.40
- **Well-performing classes**: Maintained or improved

---

## 📁 Project Structure

```
Duality_AI_Segmentation/
├── iou_pipeline/              # Core pipeline modules
│   ├── analyzer.py           # Dataset analysis
│   ├── editor.py             # Dataset augmentation
│   ├── trainer.py            # Training orchestrator
│   ├── evaluator.py          # Evaluation engine
│   ├── models/
│   │   ├── backbone.py       # DINOv2 loading
│   │   └── segmentation_head.py  # ConvNeXt decoder
│   ├── data/
│   │   ├── dataset.py        # SegmentationDataset
│   │   └── transforms.py     # Transform pipelines
│   └── utils/
│       ├── config.py         # Configuration management
│       └── metrics.py        # Metric computation
├── run_iou_boost.py          # Main training script
├── compute_current_iou.py    # Full IoU evaluation
├── compute_quick_iou.py      # Quick IoU estimation
├── iou_boost_output/         # Training outputs
│   ├── best_model.pth        # Best model checkpoint
│   └── training_history.json # Training metrics
└── README.md                 # This file
```

---

## 🔧 Key Features

### 1. Class-Weighted Training
- Inverse frequency weighting for balanced learning
- 3.67x weight for rare classes
- Addresses severe class imbalance

### 2. Optimized Architecture
- DINOv2 pretrained backbone (frozen)
- ConvNeXt-style decoder (trainable)
- Efficient transfer learning

### 3. Advanced Training
- AdamW optimizer with weight decay
- Cosine annealing learning rate schedule
- Best model tracking based on validation IoU

### 4. Comprehensive Evaluation
- Per-class IoU monitoring
- Training history logging
- Quick and full evaluation modes

---

## 📊 Dataset Information

### Training Dataset
- **Samples**: 2857 images
- **Classes**: 11 terrain classes
- **Resolution**: 480×270 (resized to 266×476 for DINOv2)

### Validation Dataset
- **Samples**: 317 images
- **No augmentation**: Preserves original distribution

### Class Mapping
```python
{
    0: Background,
    1: Trees,
    2: Lush Bushes,
    3: Dry Grass,
    4: Dry Bushes,
    5: Ground Clutter,
    6: Flowers,
    7: Logs,
    8: Rocks,
    9: Landscape,
    10: Sky
}
```

---

## 🛠️ Installation

```bash
pip install torch torchvision
pip install albumentations
pip install numpy pillow scipy matplotlib tqdm
```

---

## 📝 Usage

### Train Model
```bash
python run_iou_boost.py
```

### Evaluate Model
```bash
# Quick (200 samples, ~3 min)
python compute_quick_iou.py

# Full (2857 samples, ~30 min)
python compute_current_iou.py
```

### Use Trained Model
```python
import torch
from iou_pipeline.models.backbone import load_dinov2_backbone
from iou_pipeline.models.segmentation_head import SegmentationHeadConvNeXt

# Load model
backbone, _, _ = load_dinov2_backbone('dinov2_vits14', freeze=True)
head = SegmentationHeadConvNeXt(384, 11, 34, 19)
head.load_state_dict(torch.load('iou_boost_output/best_model.pth'))

# Inference
with torch.no_grad():
    features = backbone.forward_features(images)
    logits = head(features["x_norm_patchtokens"])
    predictions = logits.argmax(dim=1)
```

---

## 📊 Results

### Training Outputs
- **Best Model**: `iou_boost_output/best_model.pth`
- **Training History**: `iou_boost_output/training_history.json`

### Expected Improvements
- **2x Mean IoU**: 0.28 → 0.5-0.6
- **Rare Classes**: <0.02 → 0.30-0.40
- **Maintained**: Well-performing classes stay strong

---

## 📚 References

- **DINOv2**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **ConvNeXt**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

---

**Status**: ✅ Training in progress - Check `iou_boost_output/` for results!

**Last Updated**: 2026-05-28
