"""
Unit tests for TrainingOrchestrator.create_dataloaders() method.

Tests the data loader creation functionality including:
- Training and validation loader creation
- Correct batch size configuration
- Shuffling behavior
- Dataset integration
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from iou_pipeline.trainer import TrainingOrchestrator, TrainingConfig


class TestCreateDataLoaders(unittest.TestCase):
    """Test cases for create_dataloaders method."""
    
    @classmethod
    def setUpClass(cls):
        """Create temporary datasets for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.train_dir = Path(cls.temp_dir) / 'train'
        cls.val_dir = Path(cls.temp_dir) / 'val'
        
        # Create directory structure
        (cls.train_dir / 'Color_Images').mkdir(parents=True)
        (cls.train_dir / 'Segmentation').mkdir(parents=True)
        (cls.val_dir / 'Color_Images').mkdir(parents=True)
        (cls.val_dir / 'Segmentation').mkdir(parents=True)
        
        # Create synthetic training data (10 samples)
        for i in range(10):
            # Create color image (480x270)
            img = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
            Image.fromarray(img).save(
                cls.train_dir / 'Color_Images' / f'train_{i:03d}.png'
            )
            
            # Create segmentation mask with valid values
            mask = np.random.choice([0, 100, 200, 300, 500], (270, 480))
            Image.fromarray(mask.astype(np.uint8)).save(
                cls.train_dir / 'Segmentation' / f'train_{i:03d}.png'
            )
        
        # Create synthetic validation data (5 samples)
        for i in range(5):
            # Create color image (480x270)
            img = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
            Image.fromarray(img).save(
                cls.val_dir / 'Color_Images' / f'val_{i:03d}.png'
            )
            
            # Create segmentation mask with valid values
            mask = np.random.choice([0, 100, 200, 300, 500], (270, 480))
            Image.fromarray(mask.astype(np.uint8)).save(
                cls.val_dir / 'Segmentation' / f'val_{i:03d}.png'
            )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary datasets."""
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Create TrainingOrchestrator instance for each test."""
        self.config = TrainingConfig(batch_size=4)
        self.orchestrator = TrainingOrchestrator(self.config)
    
    def test_create_dataloaders_returns_tuple(self):
        """Test that create_dataloaders returns a tuple of two DataLoaders."""
        train_loader, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            num_workers=0  # Use 0 workers for testing
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
    
    def test_batch_size_configuration(self):
        """Test that batch size is correctly configured."""
        batch_size = 2
        train_loader, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            batch_size=batch_size,
            num_workers=0
        )
        
        self.assertEqual(train_loader.batch_size, batch_size)
        self.assertEqual(val_loader.batch_size, batch_size)
    
    def test_uses_config_batch_size_when_not_specified(self):
        """Test that config batch size is used when not explicitly provided."""
        train_loader, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            num_workers=0
        )
        
        self.assertEqual(train_loader.batch_size, self.config.batch_size)
        self.assertEqual(val_loader.batch_size, self.config.batch_size)
    
    def test_training_loader_shuffles(self):
        """Test that training loader has shuffle enabled."""
        train_loader, _ = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            num_workers=0
        )
        
        # Check that sampler is a RandomSampler (indicates shuffling)
        # or shuffle is True in the loader
        self.assertTrue(
            hasattr(train_loader.sampler, '__class__') and
            'Random' in train_loader.sampler.__class__.__name__
        )
    
    def test_validation_loader_does_not_shuffle(self):
        """Test that validation loader has shuffle disabled."""
        _, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            num_workers=0
        )
        
        # Check that sampler is a SequentialSampler (indicates no shuffling)
        self.assertTrue(
            hasattr(val_loader.sampler, '__class__') and
            'Sequential' in val_loader.sampler.__class__.__name__
        )
    
    def test_correct_dataset_sizes(self):
        """Test that loaders have correct number of samples."""
        train_loader, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            num_workers=0
        )
        
        # Training dataset should have 10 samples
        self.assertEqual(len(train_loader.dataset), 10)
        
        # Validation dataset should have 5 samples
        self.assertEqual(len(val_loader.dataset), 5)
    
    def test_batch_shapes(self):
        """Test that batches have correct shapes."""
        batch_size = 2
        train_loader, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            batch_size=batch_size,
            num_workers=0
        )
        
        # Get first batch from training loader
        train_images, train_masks = next(iter(train_loader))
        
        # Check image shape: (batch_size, 3, 266, 476)
        self.assertEqual(train_images.shape[0], batch_size)
        self.assertEqual(train_images.shape[1], 3)  # RGB channels
        self.assertEqual(train_images.shape[2], 266)  # Height
        self.assertEqual(train_images.shape[3], 476)  # Width
        
        # Check mask shape: (batch_size, 266, 476)
        self.assertEqual(train_masks.shape[0], batch_size)
        self.assertEqual(train_masks.shape[1], 266)  # Height
        self.assertEqual(train_masks.shape[2], 476)  # Width
        
        # Get first batch from validation loader
        val_images, val_masks = next(iter(val_loader))
        
        # Check validation batch shapes
        self.assertEqual(val_images.shape[0], batch_size)
        self.assertEqual(val_images.shape[1], 3)
        self.assertEqual(val_masks.shape[0], batch_size)
    
    def test_drop_last_behavior(self):
        """Test that training loader drops incomplete batches."""
        batch_size = 3
        train_loader, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            batch_size=batch_size,
            num_workers=0
        )
        
        # Training loader should drop last incomplete batch
        # 10 samples with batch_size=3 -> 3 complete batches (9 samples)
        train_batches = list(train_loader)
        self.assertEqual(len(train_batches), 3)
        
        # Validation loader should keep all samples
        # 5 samples with batch_size=3 -> 2 batches (3 + 2 samples)
        val_batches = list(val_loader)
        self.assertEqual(len(val_batches), 2)
        
        # Last validation batch should have 2 samples
        last_val_batch_images, _ = val_batches[-1]
        self.assertEqual(last_val_batch_images.shape[0], 2)
    
    def test_pin_memory_configuration(self):
        """Test that pin_memory is correctly configured."""
        train_loader, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            pin_memory=True,
            num_workers=0
        )
        
        self.assertTrue(train_loader.pin_memory)
        self.assertTrue(val_loader.pin_memory)
        
        # Test with pin_memory=False
        train_loader, val_loader = self.orchestrator.create_dataloaders(
            train_dataset_path=str(self.train_dir),
            val_dataset_path=str(self.val_dir),
            pin_memory=False,
            num_workers=0
        )
        
        self.assertFalse(train_loader.pin_memory)
        self.assertFalse(val_loader.pin_memory)


if __name__ == '__main__':
    unittest.main()
