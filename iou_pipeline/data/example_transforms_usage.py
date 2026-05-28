"""
Example usage of data transforms module.

This script demonstrates how to use the transform utilities for
semantic segmentation with DINOv2 backbone.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iou_pipeline.data import (
    get_training_transforms,
    get_validation_transforms,
    get_mask_transforms,
    get_transform_pipelines,
    get_augmented_training_transforms,
    denormalize_image,
    SegmentationDataset,
    DINOV2_MEAN,
    DINOV2_STD,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)


def example_basic_usage():
    """Example 1: Basic usage with individual transforms."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Get individual transforms
    train_transform = get_training_transforms()
    val_transform = get_validation_transforms()
    mask_transform = get_mask_transforms()
    
    print(f"Training transform: {train_transform}")
    print(f"Validation transform: {val_transform}")
    print(f"Mask transform: {mask_transform}")
    print()


def example_transform_pipelines():
    """Example 2: Using get_transform_pipelines."""
    print("=" * 60)
    print("Example 2: Transform Pipelines")
    print("=" * 60)
    
    # Get all transforms at once
    transforms = get_transform_pipelines()
    
    print("Available transforms:")
    for key in transforms:
        print(f"  - {key}: {type(transforms[key])}")
    print()


def example_with_dataset():
    """Example 3: Using transforms with SegmentationDataset."""
    print("=" * 60)
    print("Example 3: Integration with Dataset")
    print("=" * 60)
    
    # Get transforms
    transforms = get_transform_pipelines()
    
    # Example dataset creation (paths would need to exist)
    print("Creating dataset with transforms:")
    print("""
    train_dataset = SegmentationDataset(
        data_dir='./data/train',
        transform=transforms['train'],
        mask_transform=transforms['mask']
    )
    
    val_dataset = SegmentationDataset(
        data_dir='./data/val',
        transform=transforms['val'],
        mask_transform=transforms['mask']
    )
    """)
    print()


def example_custom_parameters():
    """Example 4: Custom parameters."""
    print("=" * 60)
    print("Example 4: Custom Parameters")
    print("=" * 60)
    
    # Custom target size
    custom_size = (224, 224)
    transform_custom_size = get_training_transforms(target_size=custom_size)
    print(f"Custom size transform: target_size={custom_size}")
    
    # Without normalization
    transform_no_norm = get_training_transforms(normalize=False)
    print("Transform without normalization")
    
    # Custom normalization parameters
    custom_mean = [0.5, 0.5, 0.5]
    custom_std = [0.5, 0.5, 0.5]
    transform_custom_norm = get_training_transforms(
        mean=custom_mean,
        std=custom_std
    )
    print(f"Custom normalization: mean={custom_mean}, std={custom_std}")
    print()


def example_augmented_transforms():
    """Example 5: Augmented training transforms."""
    print("=" * 60)
    print("Example 5: Augmented Training Transforms")
    print("=" * 60)
    
    # Get augmented transforms
    augmented_transform = get_augmented_training_transforms(
        color_jitter=True,
        random_flip=True
    )
    
    print("Augmented transform includes:")
    print("  - Random horizontal flip (p=0.5)")
    print("  - Color jitter (brightness=±20%, contrast=±20%)")
    print("  - Resize to (266, 476)")
    print("  - ToTensor")
    print("  - Normalize with DINOv2 stats")
    print()


def example_denormalization():
    """Example 6: Denormalization for visualization."""
    print("=" * 60)
    print("Example 6: Denormalization")
    print("=" * 60)
    
    print("Denormalization usage:")
    print("""
    # Normalize image
    transform = get_training_transforms()
    normalized_tensor = transform(image)
    
    # Denormalize for visualization
    denormalized = denormalize_image(normalized_tensor)
    
    # Convert to PIL Image
    from torchvision.transforms import ToPILImage
    pil_image = ToPILImage()(denormalized)
    pil_image.show()
    """)
    print()


def example_dataloader():
    """Example 7: Using with DataLoader."""
    print("=" * 60)
    print("Example 7: DataLoader Integration")
    print("=" * 60)
    
    print("DataLoader usage:")
    print("""
    from torch.utils.data import DataLoader
    from iou_pipeline.data import SegmentationDataset, get_transform_pipelines
    
    # Get transforms
    transforms = get_transform_pipelines()
    
    # Create dataset
    train_dataset = SegmentationDataset(
        data_dir='./data/train',
        transform=transforms['train'],
        mask_transform=transforms['mask']
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Iterate
    for batch_idx, (images, masks) in enumerate(train_loader):
        # images: (B, 3, 266, 476) - normalized float32
        # masks: (B, 266, 476) - int64 class labels
        
        # Your training code here
        pass
    """)
    print()


def example_constants():
    """Example 8: Using constants."""
    print("=" * 60)
    print("Example 8: Constants")
    print("=" * 60)
    
    print(f"DINOv2 normalization mean: {DINOV2_MEAN}")
    print(f"DINOv2 normalization std: {DINOV2_STD}")
    print(f"Target height: {TARGET_HEIGHT} pixels")
    print(f"Target width: {TARGET_WIDTH} pixels")
    print(f"Patch size: 14×14")
    print(f"Number of patches: {TARGET_WIDTH // 14} × {TARGET_HEIGHT // 14} = {(TARGET_WIDTH // 14) * (TARGET_HEIGHT // 14)}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("DATA TRANSFORMS MODULE - USAGE EXAMPLES")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    example_transform_pipelines()
    example_with_dataset()
    example_custom_parameters()
    example_augmented_transforms()
    example_denormalization()
    example_dataloader()
    example_constants()
    
    print("=" * 60)
    print("For more information, see TRANSFORMS_README.md")
    print("=" * 60)


if __name__ == '__main__':
    main()
