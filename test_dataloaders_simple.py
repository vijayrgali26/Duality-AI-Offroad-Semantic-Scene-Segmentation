"""
Simple test for create_dataloaders method without full package import.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Direct imports to avoid package-level imports
sys.path.insert(0, str(Path(__file__).parent))

from iou_pipeline.trainer import TrainingOrchestrator, TrainingConfig


def create_test_dataset(base_dir, num_samples, prefix):
    """Create a synthetic dataset for testing."""
    data_dir = Path(base_dir) / prefix
    (data_dir / 'Color_Images').mkdir(parents=True, exist_ok=True)
    (data_dir / 'Segmentation').mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Create color image (480x270)
        img = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
        Image.fromarray(img).save(
            data_dir / 'Color_Images' / f'{prefix}_{i:03d}.png'
        )
        
        # Create segmentation mask with valid values
        mask = np.random.choice([0, 100, 200, 300, 500], (270, 480))
        Image.fromarray(mask.astype(np.uint8)).save(
            data_dir / 'Segmentation' / f'{prefix}_{i:03d}.png'
        )
    
    return str(data_dir)


def test_create_dataloaders():
    """Test the create_dataloaders method."""
    print("Creating test datasets...")
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create synthetic datasets
        train_dir = create_test_dataset(temp_dir, 10, 'train')
        val_dir = create_test_dataset(temp_dir, 5, 'val')
        
        print(f"Train dataset: {train_dir}")
        print(f"Val dataset: {val_dir}")
        
        # Create orchestrator
        config = TrainingConfig(batch_size=4)
        orchestrator = TrainingOrchestrator(config)
        
        print("\nCreating data loaders...")
        train_loader, val_loader = orchestrator.create_dataloaders(
            train_dataset_path=train_dir,
            val_dataset_path=val_dir,
            num_workers=0  # Use 0 workers for testing
        )
        
        print(f"✓ Data loaders created successfully")
        print(f"  - Training dataset size: {len(train_loader.dataset)}")
        print(f"  - Validation dataset size: {len(val_loader.dataset)}")
        print(f"  - Training batch size: {train_loader.batch_size}")
        print(f"  - Validation batch size: {val_loader.batch_size}")
        
        # Test batch loading
        print("\nTesting batch loading...")
        train_images, train_masks = next(iter(train_loader))
        val_images, val_masks = next(iter(val_loader))
        
        print(f"✓ Batches loaded successfully")
        print(f"  - Training batch shape: images={train_images.shape}, masks={train_masks.shape}")
        print(f"  - Validation batch shape: images={val_images.shape}, masks={val_masks.shape}")
        
        # Verify shapes
        assert train_images.shape == (4, 3, 266, 476), f"Unexpected train image shape: {train_images.shape}"
        assert train_masks.shape == (4, 266, 476), f"Unexpected train mask shape: {train_masks.shape}"
        assert val_images.shape == (4, 3, 266, 476), f"Unexpected val image shape: {val_images.shape}"
        assert val_masks.shape == (4, 266, 476), f"Unexpected val mask shape: {val_masks.shape}"
        
        print("\n✓ All shape assertions passed")
        
        # Test shuffling behavior
        print("\nTesting shuffling behavior...")
        train_sampler_type = train_loader.sampler.__class__.__name__
        val_sampler_type = val_loader.sampler.__class__.__name__
        
        print(f"  - Training sampler: {train_sampler_type}")
        print(f"  - Validation sampler: {val_sampler_type}")
        
        assert 'Random' in train_sampler_type, "Training loader should use RandomSampler"
        assert 'Sequential' in val_sampler_type, "Validation loader should use SequentialSampler"
        
        print("✓ Shuffling behavior correct")
        
        # Test batch count
        print("\nTesting batch counts...")
        train_batches = list(train_loader)
        val_batches = list(val_loader)
        
        print(f"  - Training batches: {len(train_batches)}")
        print(f"  - Validation batches: {len(val_batches)}")
        
        # Training: 10 samples, batch_size=4, drop_last=True -> 2 batches
        assert len(train_batches) == 2, f"Expected 2 training batches, got {len(train_batches)}"
        
        # Validation: 5 samples, batch_size=4, drop_last=False -> 2 batches (4 + 1)
        assert len(val_batches) == 2, f"Expected 2 validation batches, got {len(val_batches)}"
        
        # Check last validation batch has 1 sample
        last_val_images, _ = val_batches[-1]
        assert last_val_images.shape[0] == 1, f"Expected last val batch to have 1 sample, got {last_val_images.shape[0]}"
        
        print("✓ Batch counts correct")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    finally:
        # Clean up
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✓ Cleanup complete")


if __name__ == '__main__':
    test_create_dataloaders()
