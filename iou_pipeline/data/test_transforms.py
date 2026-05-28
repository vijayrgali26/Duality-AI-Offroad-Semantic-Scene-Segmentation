"""
Unit tests for data transformation utilities.

Tests the transform pipelines for training and validation datasets,
including normalization, resizing, and mask transforms.
"""

import unittest
import torch
import numpy as np
from PIL import Image

from transforms import (
    get_training_transforms,
    get_validation_transforms,
    get_mask_transforms,
    get_augmented_training_transforms,
    get_transform_pipelines,
    denormalize_image,
    ResizeMask,
    ToTensorMask,
    DINOV2_MEAN,
    DINOV2_STD,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)


class TestTransforms(unittest.TestCase):
    """Test cases for data transforms."""
    
    def setUp(self):
        """Create sample images and masks for testing."""
        # Create a sample RGB image (480x270)
        self.sample_image = Image.new('RGB', (480, 270), color=(128, 128, 128))
        
        # Create a sample mask with class labels (480x270)
        mask_array = np.zeros((270, 480), dtype=np.uint8)
        mask_array[0:100, :] = 0  # Background
        mask_array[100:200, :] = 1  # Trees
        mask_array[200:270, :] = 2  # Lush Bushes
        self.sample_mask = Image.fromarray(mask_array)
    
    def test_training_transforms_shape(self):
        """Test that training transforms produce correct output shape."""
        transform = get_training_transforms()
        output = transform(self.sample_image)
        
        # Check output is a tensor
        self.assertIsInstance(output, torch.Tensor)
        
        # Check shape is (3, TARGET_HEIGHT, TARGET_WIDTH)
        self.assertEqual(output.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
    
    def test_validation_transforms_shape(self):
        """Test that validation transforms produce correct output shape."""
        transform = get_validation_transforms()
        output = transform(self.sample_image)
        
        # Check output is a tensor
        self.assertIsInstance(output, torch.Tensor)
        
        # Check shape is (3, TARGET_HEIGHT, TARGET_WIDTH)
        self.assertEqual(output.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
    
    def test_mask_transforms_shape(self):
        """Test that mask transforms produce correct output shape."""
        transform = get_mask_transforms()
        output = transform(self.sample_mask)
        
        # Check output is a tensor
        self.assertIsInstance(output, torch.Tensor)
        
        # Check shape is (TARGET_HEIGHT, TARGET_WIDTH)
        self.assertEqual(output.shape, (TARGET_HEIGHT, TARGET_WIDTH))
        
        # Check dtype is int64 (for class labels)
        self.assertEqual(output.dtype, torch.int64)
    
    def test_mask_transforms_preserve_labels(self):
        """Test that mask transforms preserve class labels."""
        transform = get_mask_transforms()
        output = transform(self.sample_mask)
        
        # Check that unique values are preserved (0, 1, 2)
        unique_values = torch.unique(output).tolist()
        self.assertTrue(all(v in [0, 1, 2] for v in unique_values))
    
    def test_normalization_applied(self):
        """Test that normalization is applied correctly."""
        # Create transform with normalization
        transform = get_training_transforms(normalize=True)
        output = transform(self.sample_image)
        
        # Normalized values should be roughly in range [-3, 3] for ImageNet stats
        self.assertTrue(output.min() >= -3.0)
        self.assertTrue(output.max() <= 3.0)
    
    def test_normalization_disabled(self):
        """Test that normalization can be disabled."""
        # Create transform without normalization
        transform = get_training_transforms(normalize=False)
        output = transform(self.sample_image)
        
        # Without normalization, values should be in [0, 1]
        self.assertTrue(output.min() >= 0.0)
        self.assertTrue(output.max() <= 1.0)
    
    def test_custom_normalization_params(self):
        """Test that custom normalization parameters work."""
        custom_mean = [0.5, 0.5, 0.5]
        custom_std = [0.5, 0.5, 0.5]
        
        transform = get_training_transforms(
            normalize=True,
            mean=custom_mean,
            std=custom_std
        )
        output = transform(self.sample_image)
        
        # Check output is a tensor
        self.assertIsInstance(output, torch.Tensor)
    
    def test_denormalize_image(self):
        """Test that denormalization reverses normalization."""
        # Normalize image
        transform = get_training_transforms(normalize=True)
        normalized = transform(self.sample_image)
        
        # Denormalize
        denormalized = denormalize_image(normalized)
        
        # Check shape is preserved
        self.assertEqual(denormalized.shape, normalized.shape)
        
        # Check values are in [0, 1] range
        self.assertTrue(denormalized.min() >= 0.0)
        self.assertTrue(denormalized.max() <= 1.0)
    
    def test_denormalize_batch(self):
        """Test that denormalization works with batched tensors."""
        # Create a batch of normalized images
        transform = get_training_transforms(normalize=True)
        normalized = transform(self.sample_image)
        batch = normalized.unsqueeze(0).repeat(4, 1, 1, 1)  # Batch of 4
        
        # Denormalize batch
        denormalized = denormalize_image(batch)
        
        # Check shape is preserved
        self.assertEqual(denormalized.shape, batch.shape)
        
        # Check values are in [0, 1] range
        self.assertTrue(denormalized.min() >= 0.0)
        self.assertTrue(denormalized.max() <= 1.0)
    
    def test_resize_mask_nearest_neighbor(self):
        """Test that ResizeMask uses nearest neighbor interpolation."""
        resize = ResizeMask((TARGET_HEIGHT, TARGET_WIDTH))
        resized_mask = resize(self.sample_mask)
        
        # Check output is PIL Image
        self.assertIsInstance(resized_mask, Image.Image)
        
        # Check size
        self.assertEqual(resized_mask.size, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Check that only original class labels are present (no interpolation)
        mask_array = np.array(resized_mask)
        unique_values = np.unique(mask_array)
        self.assertTrue(all(v in [0, 1, 2] for v in unique_values))
    
    def test_to_tensor_mask(self):
        """Test that ToTensorMask preserves integer labels."""
        to_tensor = ToTensorMask()
        tensor_mask = to_tensor(self.sample_mask)
        
        # Check output is tensor
        self.assertIsInstance(tensor_mask, torch.Tensor)
        
        # Check dtype is int64
        self.assertEqual(tensor_mask.dtype, torch.int64)
        
        # Check values are preserved
        unique_values = torch.unique(tensor_mask).tolist()
        self.assertTrue(all(v in [0, 1, 2] for v in unique_values))
    
    def test_augmented_training_transforms(self):
        """Test that augmented training transforms work."""
        transform = get_augmented_training_transforms(
            color_jitter=True,
            random_flip=True
        )
        output = transform(self.sample_image)
        
        # Check output is a tensor
        self.assertIsInstance(output, torch.Tensor)
        
        # Check shape
        self.assertEqual(output.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
    
    def test_get_transform_pipelines(self):
        """Test that get_transform_pipelines returns all required transforms."""
        pipelines = get_transform_pipelines()
        
        # Check all keys are present
        self.assertIn('train', pipelines)
        self.assertIn('val', pipelines)
        self.assertIn('mask', pipelines)
        
        # Test each pipeline
        train_output = pipelines['train'](self.sample_image)
        val_output = pipelines['val'](self.sample_image)
        mask_output = pipelines['mask'](self.sample_mask)
        
        # Check shapes
        self.assertEqual(train_output.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
        self.assertEqual(val_output.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
        self.assertEqual(mask_output.shape, (TARGET_HEIGHT, TARGET_WIDTH))
    
    def test_get_transform_pipelines_with_augmentation(self):
        """Test that get_transform_pipelines with augmentation works."""
        pipelines = get_transform_pipelines(use_augmentation=True)
        
        # Check all keys are present
        self.assertIn('train', pipelines)
        self.assertIn('val', pipelines)
        self.assertIn('mask', pipelines)
        
        # Test training pipeline (should have augmentation)
        train_output = pipelines['train'](self.sample_image)
        self.assertEqual(train_output.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
    
    def test_custom_target_size(self):
        """Test that custom target size works."""
        custom_size = (224, 224)
        
        transform = get_training_transforms(target_size=custom_size)
        output = transform(self.sample_image)
        
        # Check shape matches custom size
        self.assertEqual(output.shape, (3, custom_size[0], custom_size[1]))
    
    def test_constants(self):
        """Test that constants are defined correctly."""
        # Check DINOv2 normalization constants
        self.assertEqual(len(DINOV2_MEAN), 3)
        self.assertEqual(len(DINOV2_STD), 3)
        
        # Check target dimensions
        self.assertEqual(TARGET_HEIGHT, 266)
        self.assertEqual(TARGET_WIDTH, 476)
        
        # Verify these are correct for 14×14 patch alignment
        # 476 / 14 = 34 patches, 266 / 14 = 19 patches
        self.assertEqual(TARGET_WIDTH % 14, 0)
        self.assertEqual(TARGET_HEIGHT % 14, 0)


if __name__ == '__main__':
    unittest.main()
