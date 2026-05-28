"""
Unit tests for SegmentationDataset class.

Tests cover initialization, data loading, mask conversion, validation,
and error handling for the SegmentationDataset.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from iou_pipeline.data.dataset import SegmentationDataset, VALUE_MAP, NUM_CLASSES


# Simple transform classes to avoid torchvision dependency issues
class ToTensor:
    """Convert PIL Image to tensor."""
    def __call__(self, img):
        arr = np.array(img)
        if len(arr.shape) == 3:
            # RGB image: H x W x C -> C x H x W
            arr = arr.transpose(2, 0, 1)
        else:
            # Grayscale: H x W -> 1 x H x W
            arr = arr[np.newaxis, :]
        return torch.from_numpy(arr).float() / 255.0


class Resize:
    """Resize PIL Image."""
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def __call__(self, img):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR)


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class TestSegmentationDataset:
    """Test suite for SegmentationDataset class."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory with sample images and masks."""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        image_dir = Path(temp_dir) / 'Color_Images'
        mask_dir = Path(temp_dir) / 'Segmentation'
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        
        # Create sample images and masks
        for i in range(5):
            # Create RGB image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_pil = Image.fromarray(img)
            img_pil.save(image_dir / f'sample_{i}.png')
            
            # Create mask with valid values
            mask = np.zeros((100, 100), dtype=np.uint16)
            # Add some class regions
            mask[0:30, 0:30] = 0      # background
            mask[30:60, 0:30] = 100   # Trees
            mask[60:100, 0:30] = 200  # Lush Bushes
            mask[0:30, 30:60] = 300   # Dry Grass
            mask[30:60, 30:60] = 500  # Dry Bushes
            mask[60:100, 30:60] = 550 # Ground Clutter
            mask[0:50, 60:100] = 600  # Flowers
            mask[50:100, 60:100] = 10000  # Sky
            
            mask_pil = Image.fromarray(mask)
            mask_pil.save(mask_dir / f'sample_{i}.png')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def corrupted_dataset_dir(self):
        """Create a dataset with some corrupted files."""
        temp_dir = tempfile.mkdtemp()
        
        image_dir = Path(temp_dir) / 'Color_Images'
        mask_dir = Path(temp_dir) / 'Segmentation'
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        
        # Create valid sample
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img).save(image_dir / 'valid.png')
        mask = np.zeros((100, 100), dtype=np.uint16)
        Image.fromarray(mask).save(mask_dir / 'valid.png')
        
        # Create image without corresponding mask
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img).save(image_dir / 'no_mask.png')
        
        # Create corrupted image file
        with open(image_dir / 'corrupted.png', 'w') as f:
            f.write('not an image')
        mask = np.zeros((100, 100), dtype=np.uint16)
        Image.fromarray(mask).save(mask_dir / 'corrupted.png')
        
        # Create mask with invalid values
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img).save(image_dir / 'invalid_mask.png')
        invalid_mask = np.full((100, 100), 9999, dtype=np.uint16)  # Invalid value
        Image.fromarray(invalid_mask).save(mask_dir / 'invalid_mask.png')
        
        yield temp_dir
        
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_dataset_dir):
        """Test basic dataset initialization."""
        dataset = SegmentationDataset(temp_dataset_dir)
        
        assert len(dataset) == 5
        assert dataset.num_classes == NUM_CLASSES
        assert dataset.value_map == VALUE_MAP
        assert dataset.image_dir.exists()
        assert dataset.masks_dir.exists()
    
    def test_initialization_with_missing_directory(self):
        """Test initialization fails with missing directories."""
        with pytest.raises(FileNotFoundError):
            SegmentationDataset('/nonexistent/path')
    
    def test_initialization_with_empty_directory(self):
        """Test initialization fails with empty image directory."""
        temp_dir = tempfile.mkdtemp()
        image_dir = Path(temp_dir) / 'Color_Images'
        mask_dir = Path(temp_dir) / 'Segmentation'
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        
        try:
            with pytest.raises(ValueError, match="No image files found"):
                SegmentationDataset(temp_dir)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_getitem(self, temp_dataset_dir):
        """Test loading a sample from the dataset."""
        transform = Compose([
            Resize((50, 50)),
            ToTensor(),
        ])
        mask_transform = Compose([
            Resize((50, 50)),
            ToTensor(),
        ])
        
        dataset = SegmentationDataset(
            temp_dataset_dir,
            transform=transform,
            mask_transform=mask_transform
        )
        
        image, mask = dataset[0]
        
        # Check types
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        
        # Check shapes
        assert image.shape == (3, 50, 50)  # RGB image
        assert mask.shape == (1, 50, 50)   # Single channel mask
        
        # Check value ranges
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert mask.min() >= 0.0
        assert mask.max() <= 255.0
    
    def test_getitem_without_transform(self, temp_dataset_dir):
        """Test loading without transforms returns PIL Images."""
        dataset = SegmentationDataset(temp_dataset_dir)
        
        image, mask = dataset[0]
        
        assert isinstance(image, Image.Image)
        assert isinstance(mask, Image.Image)
    
    def test_mask_conversion(self, temp_dataset_dir):
        """Test mask value mapping from raw values to class IDs."""
        dataset = SegmentationDataset(temp_dataset_dir)
        
        # Load mask directly
        mask_path = dataset.masks_dir / dataset.data_ids[0]
        raw_mask = Image.open(mask_path)
        converted_mask = dataset._convert_mask(raw_mask)
        
        mask_arr = np.array(converted_mask)
        unique_values = np.unique(mask_arr)
        
        # All values should be valid class IDs
        for val in unique_values:
            assert 0 <= val < NUM_CLASSES
    
    def test_validate_dataset(self, corrupted_dataset_dir):
        """Test dataset validation removes corrupted files."""
        dataset = SegmentationDataset(
            corrupted_dataset_dir,
            validate_on_init=True
        )
        
        # Should only have the valid sample
        assert len(dataset) == 1
        assert dataset.data_ids[0] == 'valid.png'
    
    def test_get_class_distribution(self, temp_dataset_dir):
        """Test class distribution computation."""
        dataset = SegmentationDataset(temp_dataset_dir)
        distribution = dataset.get_class_distribution()
        
        # Check return type
        assert isinstance(distribution, dict)
        
        # Check all classes are present
        assert len(distribution) == NUM_CLASSES
        
        # Check all counts are non-negative
        for class_id, count in distribution.items():
            assert count >= 0
        
        # Check total pixels makes sense (5 images * 100 * 100 pixels)
        total_pixels = sum(distribution.values())
        assert total_pixels == 5 * 100 * 100
    
    def test_get_sample_info(self, temp_dataset_dir):
        """Test getting sample metadata."""
        dataset = SegmentationDataset(temp_dataset_dir)
        info = dataset.get_sample_info(0)
        
        assert 'index' in info
        assert 'filename' in info
        assert 'image_path' in info
        assert 'mask_path' in info
        
        assert info['index'] == 0
        assert info['filename'] == dataset.data_ids[0]
        assert Path(info['image_path']).exists()
        assert Path(info['mask_path']).exists()
    
    def test_custom_value_map(self, temp_dataset_dir):
        """Test using a custom value map."""
        custom_map = {0: 0, 100: 1, 200: 2}
        
        dataset = SegmentationDataset(
            temp_dataset_dir,
            value_map=custom_map
        )
        
        assert dataset.value_map == custom_map
        assert dataset.num_classes == len(custom_map)
    
    def test_error_handling_corrupted_file(self, corrupted_dataset_dir):
        """Test error handling when loading corrupted files."""
        # Don't validate on init to test runtime error handling
        dataset = SegmentationDataset(
            corrupted_dataset_dir,
            validate_on_init=False
        )
        
        # Find the corrupted file index
        corrupted_idx = None
        for idx, data_id in enumerate(dataset.data_ids):
            if data_id == 'corrupted.png':
                corrupted_idx = idx
                break
        
        if corrupted_idx is not None:
            with pytest.raises(RuntimeError, match="Failed to load sample"):
                _ = dataset[corrupted_idx]
    
    def test_len(self, temp_dataset_dir):
        """Test __len__ method."""
        dataset = SegmentationDataset(temp_dataset_dir)
        assert len(dataset) == 5
    
    def test_with_dataloader(self, temp_dataset_dir):
        """Test dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        transform = Compose([
            Resize((50, 50)),
            ToTensor(),
        ])
        mask_transform = Compose([
            Resize((50, 50)),
            ToTensor(),
        ])
        
        dataset = SegmentationDataset(
            temp_dataset_dir,
            transform=transform,
            mask_transform=mask_transform
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Test one batch
        images, masks = next(iter(dataloader))
        
        assert images.shape == (2, 3, 50, 50)
        assert masks.shape == (2, 1, 50, 50)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
