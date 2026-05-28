# Data Transforms Module

## Overview

The `transforms.py` module provides data transformation utilities for semantic segmentation with DINOv2 backbone. It includes normalization for DINOv2 input requirements and resize transforms for 14×14 patch alignment.

## Features

### 1. **DINOv2 Normalization**
- Uses ImageNet statistics: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- Required for DINOv2 pretrained models

### 2. **Patch-Aligned Resizing**
- Target resolution: **476×266** pixels
- Ensures 14×14 patch alignment for DINOv2-ViT
- 476 ÷ 14 = 34 patches (width)
- 266 ÷ 14 = 19 patches (height)

### 3. **Mask-Preserving Transforms**
- Uses nearest neighbor interpolation for masks
- Preserves integer class labels (no interpolation artifacts)
- Converts masks to int64 tensors for PyTorch

## Usage

### Basic Usage

```python
from iou_pipeline.data import (
    get_training_transforms,
    get_validation_transforms,
    get_mask_transforms,
    SegmentationDataset
)

# Get transform pipelines
train_transform = get_training_transforms()
val_transform = get_validation_transforms()
mask_transform = get_mask_transforms()

# Create datasets
train_dataset = SegmentationDataset(
    data_dir='./data/train',
    transform=train_transform,
    mask_transform=mask_transform
)

val_dataset = SegmentationDataset(
    data_dir='./data/val',
    transform=val_transform,
    mask_transform=mask_transform
)
```

### Using Transform Pipelines

```python
from iou_pipeline.data import get_transform_pipelines

# Get all transforms at once
transforms = get_transform_pipelines()

train_dataset = SegmentationDataset(
    data_dir='./data/train',
    transform=transforms['train'],
    mask_transform=transforms['mask']
)
```

### Custom Target Size

```python
# Use custom resolution (must be divisible by 14 for patch alignment)
custom_transforms = get_training_transforms(target_size=(224, 224))
```

### Disable Normalization

```python
# Get transforms without normalization
transform = get_training_transforms(normalize=False)
```

### Custom Normalization Parameters

```python
# Use custom mean and std
custom_mean = [0.5, 0.5, 0.5]
custom_std = [0.5, 0.5, 0.5]

transform = get_training_transforms(
    mean=custom_mean,
    std=custom_std
)
```

### Augmented Training Transforms

```python
from iou_pipeline.data import get_augmented_training_transforms

# Get transforms with color jittering and random flips
augmented_transform = get_augmented_training_transforms(
    color_jitter=True,
    random_flip=True
)
```

**Note:** For more advanced augmentations (rotation, etc.) that require synchronized image-mask transforms, use the `DatasetEditor` module with albumentations.

### Denormalization for Visualization

```python
from iou_pipeline.data import denormalize_image
from torchvision.transforms import ToPILImage

# Load and normalize image
transform = get_training_transforms()
normalized_tensor = transform(image)

# Denormalize for visualization
denormalized = denormalize_image(normalized_tensor)

# Convert to PIL Image
pil_image = ToPILImage()(denormalized)
pil_image.show()
```

## API Reference

### Constants

- `DINOV2_MEAN`: ImageNet normalization mean `[0.485, 0.456, 0.406]`
- `DINOV2_STD`: ImageNet normalization std `[0.229, 0.224, 0.225]`
- `TARGET_HEIGHT`: Default target height `266` pixels
- `TARGET_WIDTH`: Default target width `476` pixels

### Functions

#### `get_training_transforms(target_size, normalize, mean, std)`
Returns training image transforms.

**Parameters:**
- `target_size` (tuple): Target (height, width). Default: `(266, 476)`
- `normalize` (bool): Apply normalization. Default: `True`
- `mean` (list): Custom mean. Default: `DINOV2_MEAN`
- `std` (list): Custom std. Default: `DINOV2_STD`

**Returns:** `torchvision.transforms.Compose`

#### `get_validation_transforms(target_size, normalize, mean, std)`
Returns validation image transforms (identical to training, no augmentation).

**Parameters:** Same as `get_training_transforms()`

**Returns:** `torchvision.transforms.Compose`

#### `get_mask_transforms(target_size)`
Returns mask transforms with nearest neighbor interpolation.

**Parameters:**
- `target_size` (tuple): Target (height, width). Default: `(266, 476)`

**Returns:** `torchvision.transforms.Compose`

#### `get_augmented_training_transforms(target_size, normalize, mean, std, color_jitter, random_flip)`
Returns augmented training transforms with color jittering and flipping.

**Parameters:**
- `target_size` (tuple): Target (height, width). Default: `(266, 476)`
- `normalize` (bool): Apply normalization. Default: `True`
- `mean` (list): Custom mean. Default: `DINOV2_MEAN`
- `std` (list): Custom std. Default: `DINOV2_STD`
- `color_jitter` (bool): Apply color jittering. Default: `True`
- `random_flip` (bool): Apply random horizontal flip. Default: `True`

**Returns:** `torchvision.transforms.Compose`

#### `get_transform_pipelines(target_size, use_augmentation)`
Returns all transform pipelines in a dictionary.

**Parameters:**
- `target_size` (tuple): Target (height, width). Default: `(266, 476)`
- `use_augmentation` (bool): Use augmented training transforms. Default: `False`

**Returns:** Dictionary with keys `'train'`, `'val'`, `'mask'`

#### `denormalize_image(tensor, mean, std)`
Denormalizes a normalized image tensor for visualization.

**Parameters:**
- `tensor` (torch.Tensor): Normalized image tensor (C, H, W) or (B, C, H, W)
- `mean` (list): Normalization mean. Default: `DINOV2_MEAN`
- `std` (list): Normalization std. Default: `DINOV2_STD`

**Returns:** `torch.Tensor` in [0, 1] range

### Classes

#### `ResizeWithPadding(target_size, fill)`
Resize image while maintaining aspect ratio with padding.

**Parameters:**
- `target_size` (tuple): Target (height, width)
- `fill` (int): Fill value for padding. Default: `0`

#### `ResizeMask(target_size)`
Resize mask using nearest neighbor interpolation.

**Parameters:**
- `target_size` (tuple): Target (height, width)

#### `ToTensorMask()`
Convert PIL Image mask to PyTorch tensor without normalization.

## Implementation Details

### Resize Strategy
- Images are resized using bilinear interpolation
- Masks are resized using nearest neighbor interpolation to preserve class labels
- Target size ensures 14×14 patch alignment for DINOv2-ViT

### Normalization
- Applied after converting to tensor
- Uses ImageNet statistics (DINOv2 requirement)
- Formula: `x_normalized = (x - mean) / std`

### Mask Handling
- Masks are converted to int64 tensors
- No normalization applied to masks
- Class labels are preserved exactly

### Augmentation
- Basic augmentations (flip, color jitter) can be applied to images only
- For synchronized image-mask augmentations (rotation, etc.), use `DatasetEditor` with albumentations

## Testing

The module includes comprehensive unit tests in `test_transforms.py`:

```bash
# Run tests
python -m pytest iou_pipeline/data/test_transforms.py -v

# Or run directly
cd iou_pipeline/data
python test_transforms.py
```

## Requirements

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- Pillow >= 8.0.0
- numpy >= 1.19.0

## Integration with Dataset

The transforms integrate seamlessly with `SegmentationDataset`:

```python
from iou_pipeline.data import (
    SegmentationDataset,
    get_transform_pipelines
)

# Get transforms
transforms = get_transform_pipelines()

# Create dataset
dataset = SegmentationDataset(
    data_dir='./data/train',
    transform=transforms['train'],
    mask_transform=transforms['mask']
)

# Use with DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Iterate
for images, masks in loader:
    # images: (B, 3, 266, 476) - normalized
    # masks: (B, 266, 476) - int64 class labels
    pass
```

## Notes

- The target resolution (476×266) is specifically chosen for DINOv2-ViT patch alignment
- Normalization is required for DINOv2 pretrained models
- Masks must use nearest neighbor interpolation to avoid creating invalid class labels
- For advanced augmentations, use the `DatasetEditor` module which handles synchronized image-mask transforms

## See Also

- `dataset.py`: SegmentationDataset implementation
- `../editor.py`: DatasetEditor for advanced augmentation
- `../trainer.py`: Training orchestrator using these transforms
