# Data Module

This module provides dataset classes for loading and processing semantic segmentation data.

## SegmentationDataset

The `SegmentationDataset` class is a PyTorch Dataset for loading RGB images and corresponding segmentation masks.

### Features

- **Automatic mask value mapping**: Converts raw pixel values to sequential class IDs
- **Graceful error handling**: Handles corrupted files with logging
- **Validation support**: Optional dataset validation during initialization
- **Class distribution analysis**: Built-in method to compute pixel counts per class
- **Transform support**: Compatible with any image transformation pipeline
- **PyTorch DataLoader integration**: Works seamlessly with PyTorch's DataLoader

### Directory Structure

The dataset expects the following directory structure:

```
data_dir/
    Color_Images/
        image1.png
        image2.png
        ...
    Segmentation/
        image1.png
        image2.png
        ...
```

### Usage Example

```python
from iou_pipeline.data import SegmentationDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((266, 476)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((266, 476)),
    transforms.ToTensor(),
])

# Create dataset
dataset = SegmentationDataset(
    data_dir='./data/train',
    transform=transform,
    mask_transform=mask_transform
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for images, masks in dataloader:
    # images: [batch_size, 3, H, W]
    # masks: [batch_size, 1, H, W]
    pass
```

### Validation

To validate the dataset during initialization and remove corrupted files:

```python
dataset = SegmentationDataset(
    data_dir='./data/train',
    validate_on_init=True
)
```

This will:
- Check that both image and mask files exist
- Verify files can be opened without errors
- Validate masks contain only valid class values
- Remove corrupted entries and log warnings

### Class Distribution

To analyze class distribution in the dataset:

```python
distribution = dataset.get_class_distribution()

# distribution is a dict: {class_id: pixel_count}
for class_id, count in distribution.items():
    print(f"Class {class_id}: {count} pixels")
```

### Custom Value Mapping

To use a custom mapping from raw mask values to class IDs:

```python
custom_map = {
    0: 0,      # background
    100: 1,    # class 1
    200: 2,    # class 2
}

dataset = SegmentationDataset(
    data_dir='./data/train',
    value_map=custom_map
)
```

### Sample Information

To get metadata about a specific sample:

```python
info = dataset.get_sample_info(0)
# Returns: {
#     'index': 0,
#     'filename': 'image.png',
#     'image_path': '/path/to/Color_Images/image.png',
#     'mask_path': '/path/to/Segmentation/image.png'
# }
```

## Value Map

The default value map for the Offroad Segmentation dataset:

| Raw Value | Class ID | Class Name      |
|-----------|----------|-----------------|
| 0         | 0        | Background      |
| 100       | 1        | Trees           |
| 200       | 2        | Lush Bushes     |
| 300       | 3        | Dry Grass       |
| 500       | 4        | Dry Bushes      |
| 550       | 5        | Ground Clutter  |
| 600       | 6        | Flowers         |
| 700       | 7        | Logs            |
| 800       | 8        | Rocks           |
| 7100      | 9        | Landscape       |
| 10000     | 10       | Sky             |

## Error Handling

The dataset handles errors gracefully:

- **Missing files**: Logged as warnings during validation
- **Corrupted images**: Logged and skipped during validation
- **Invalid mask values**: Logged as warnings, unmapped pixels set to background (class 0)
- **Loading errors**: Raised as `RuntimeError` with detailed error message

## Testing

Run the unit tests:

```bash
pytest iou_pipeline/data/test_dataset.py -v
```

Run the integration test:

```bash
python test_dataset_integration.py
```

## Requirements

- Python 3.7+
- PyTorch
- PIL/Pillow
- NumPy
