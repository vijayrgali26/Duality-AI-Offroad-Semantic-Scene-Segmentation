"""
Example usage of create_dataloaders() method.

This script demonstrates how to create and use data loaders for training
and validation in the IoU Improvement Pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iou_pipeline.trainer import TrainingOrchestrator, TrainingConfig


def example_basic_usage():
    """Basic example of creating data loaders."""
    print("="*60)
    print("Example 1: Basic Data Loader Creation")
    print("="*60)
    
    # Create training configuration
    config = TrainingConfig(
        batch_size=8,
        num_epochs=100,
        learning_rate=0.0001
    )
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    # Create data loaders
    # Note: Replace these paths with actual dataset paths
    train_loader, val_loader = orchestrator.create_dataloaders(
        train_dataset_path='./data/augmented_train',
        val_dataset_path='./data/val',
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n✓ Data loaders created successfully!")
    print(f"  Training dataset size: {len(train_loader.dataset)}")
    print(f"  Validation dataset size: {len(val_loader.dataset)}")
    print(f"  Training batches per epoch: {len(train_loader)}")
    print(f"  Validation batches per epoch: {len(val_loader)}")
    print(f"  Batch size: {train_loader.batch_size}")


def example_custom_batch_size():
    """Example with custom batch size."""
    print("\n" + "="*60)
    print("Example 2: Custom Batch Size")
    print("="*60)
    
    config = TrainingConfig(batch_size=8)  # Default batch size
    orchestrator = TrainingOrchestrator(config)
    
    # Override batch size for this specific data loader creation
    train_loader, val_loader = orchestrator.create_dataloaders(
        train_dataset_path='./data/augmented_train',
        val_dataset_path='./data/val',
        batch_size=16,  # Custom batch size
        num_workers=4
    )
    
    print(f"\n✓ Data loaders created with custom batch size!")
    print(f"  Config batch size: {config.batch_size}")
    print(f"  Actual batch size: {train_loader.batch_size}")


def example_training_loop():
    """Example of using data loaders in a training loop."""
    print("\n" + "="*60)
    print("Example 3: Training Loop Usage")
    print("="*60)
    
    config = TrainingConfig(batch_size=4, num_epochs=2)
    orchestrator = TrainingOrchestrator(config)
    
    train_loader, val_loader = orchestrator.create_dataloaders(
        train_dataset_path='./data/augmented_train',
        val_dataset_path='./data/val',
        num_workers=0  # Use 0 for debugging
    )
    
    print("\nTraining loop structure:")
    print("""
    for epoch in range(config.num_epochs):
        # Training phase
        for batch_idx, (images, masks) in enumerate(train_loader):
            # images shape: (batch_size, 3, 266, 476)
            # masks shape: (batch_size, 266, 476)
            
            # Forward pass
            # outputs = model(images)
            
            # Compute loss
            # loss = criterion(outputs, masks)
            
            # Backward pass
            # loss.backward()
            # optimizer.step()
            
            pass
        
        # Validation phase
        with torch.no_grad():
            for images, masks in val_loader:
                # Forward pass
                # outputs = model(images)
                
                # Compute metrics
                # iou = compute_iou(outputs, masks)
                
                pass
    """)


def example_batch_inspection():
    """Example of inspecting batch contents."""
    print("\n" + "="*60)
    print("Example 4: Batch Inspection")
    print("="*60)
    
    config = TrainingConfig(batch_size=4)
    orchestrator = TrainingOrchestrator(config)
    
    train_loader, val_loader = orchestrator.create_dataloaders(
        train_dataset_path='./data/augmented_train',
        val_dataset_path='./data/val',
        num_workers=0
    )
    
    print("\nInspecting first training batch:")
    print("""
    # Get first batch
    images, masks = next(iter(train_loader))
    
    print(f"Images shape: {images.shape}")  # (4, 3, 266, 476)
    print(f"Images dtype: {images.dtype}")  # torch.float32
    print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
    
    print(f"Masks shape: {masks.shape}")  # (4, 266, 476)
    print(f"Masks dtype: {masks.dtype}")  # torch.int64
    print(f"Unique classes: {torch.unique(masks)}")  # [0, 1, 2, ...]
    """)


def example_dataloader_properties():
    """Example showing data loader properties."""
    print("\n" + "="*60)
    print("Example 5: DataLoader Properties")
    print("="*60)
    
    config = TrainingConfig(batch_size=8)
    orchestrator = TrainingOrchestrator(config)
    
    train_loader, val_loader = orchestrator.create_dataloaders(
        train_dataset_path='./data/augmented_train',
        val_dataset_path='./data/val',
        num_workers=4,
        pin_memory=True
    )
    
    print("\nTraining DataLoader Properties:")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Num workers: {train_loader.num_workers}")
    print(f"  Pin memory: {train_loader.pin_memory}")
    print(f"  Drop last: {train_loader.drop_last}")
    print(f"  Sampler type: {train_loader.sampler.__class__.__name__}")
    
    print("\nValidation DataLoader Properties:")
    print(f"  Batch size: {val_loader.batch_size}")
    print(f"  Num workers: {val_loader.num_workers}")
    print(f"  Pin memory: {val_loader.pin_memory}")
    print(f"  Drop last: {val_loader.drop_last}")
    print(f"  Sampler type: {val_loader.sampler.__class__.__name__}")
    
    print("\nKey Differences:")
    print("  ✓ Training uses RandomSampler (shuffles data)")
    print("  ✓ Validation uses SequentialSampler (no shuffling)")
    print("  ✓ Training drops incomplete batches (drop_last=True)")
    print("  ✓ Validation keeps all samples (drop_last=False)")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("DataLoader Creation Examples")
    print("="*60)
    
    print("\nNote: These examples show the API usage.")
    print("To run them with actual data, replace the dataset paths")
    print("with your actual training and validation directories.")
    
    # Show example code without executing (since we don't have actual data)
    example_basic_usage.__doc__ and print(f"\n{example_basic_usage.__doc__}")
    example_custom_batch_size.__doc__ and print(f"\n{example_custom_batch_size.__doc__}")
    example_training_loop.__doc__ and print(f"\n{example_training_loop.__doc__}")
    example_batch_inspection.__doc__ and print(f"\n{example_batch_inspection.__doc__}")
    example_dataloader_properties.__doc__ and print(f"\n{example_dataloader_properties.__doc__}")
    
    print("\n" + "="*60)
    print("For actual execution, ensure you have:")
    print("  1. Training dataset at specified path")
    print("  2. Validation dataset at specified path")
    print("  3. Proper directory structure (Color_Images/ and Segmentation/)")
    print("="*60)


if __name__ == '__main__':
    main()
