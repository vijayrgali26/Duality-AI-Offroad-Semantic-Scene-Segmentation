"""
Test script for model building utilities (Task 6.1)

Tests the DINOv2 backbone loading and segmentation head creation.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iou_pipeline.trainer import TrainingOrchestrator, TrainingConfig
from iou_pipeline.models.backbone import list_available_backbones, align_image_size_to_patches
from iou_pipeline.models.segmentation_head import SegmentationHeadConvNeXt


def test_backbone_loading():
    """Test DINOv2 backbone loading."""
    print("=" * 80)
    print("TEST 1: DINOv2 Backbone Loading")
    print("=" * 80)
    
    # List available backbones
    list_available_backbones()
    
    print("\n✓ Backbone listing successful")


def test_segmentation_head():
    """Test segmentation head creation."""
    print("\n" + "=" * 80)
    print("TEST 2: Segmentation Head Creation")
    print("=" * 80)
    
    # Create a segmentation head
    head = SegmentationHeadConvNeXt(
        in_channels=384,  # DINOv2-ViT-Small
        out_channels=11,  # 11 terrain classes
        token_h=34,       # 476 / 14
        token_w=19        # 266 / 14
    )
    
    print(f"\nCreated SegmentationHeadConvNeXt")
    print(f"  Input channels: 384")
    print(f"  Output classes: 11")
    print(f"  Patch grid: 34x19")
    print(f"  Parameters: {head.get_num_params():,}")
    
    # Test forward pass
    batch_size = 2
    num_patches = 34 * 19
    embed_dim = 384
    tokens = torch.randn(batch_size, num_patches, embed_dim)
    
    print(f"\nTesting forward pass...")
    print(f"  Input shape: {tokens.shape}")
    
    logits = head(tokens)
    print(f"  Output shape: {logits.shape}")
    
    assert logits.shape == (batch_size, 11, 34, 19), "Output shape mismatch!"
    
    print("\n✓ Segmentation head test successful")


def test_build_model():
    """Test complete model building with TrainingOrchestrator."""
    print("\n" + "=" * 80)
    print("TEST 3: Complete Model Building")
    print("=" * 80)
    
    # Create config
    config = TrainingConfig(
        backbone='dinov2_vits14',
        num_classes=11,
        batch_size=2
    )
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    print(f"\nDevice: {orchestrator.device}")
    
    # Align image size
    h, w = align_image_size_to_patches(480, 270, 14)
    print(f"Aligned image size: {h}x{w}")
    
    # Build model
    print("\nBuilding model...")
    backbone, head = orchestrator.build_model(
        backbone='dinov2_vits14',
        num_classes=11,
        image_height=h,
        image_width=w,
        freeze_backbone=True
    )
    
    # Verify backbone is frozen
    backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    
    print(f"\nVerifying model properties...")
    print(f"  Backbone trainable params: {backbone_trainable:,}")
    print(f"  Head trainable params: {head_trainable:,}")
    
    assert backbone_trainable == 0, "Backbone should be frozen!"
    assert head_trainable > 0, "Head should be trainable!"
    
    print("\n✓ Model building test successful")


def test_forward_pass():
    """Test complete forward pass through backbone and head."""
    print("\n" + "=" * 80)
    print("TEST 4: Complete Forward Pass")
    print("=" * 80)
    
    # Create config
    config = TrainingConfig(backbone='dinov2_vits14', num_classes=11)
    orchestrator = TrainingOrchestrator(config)
    
    # Build model
    h, w = align_image_size_to_patches(480, 270, 14)
    backbone, head = orchestrator.build_model(
        backbone='dinov2_vits14',
        num_classes=11,
        image_height=h,
        image_width=w
    )
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, h, w).to(orchestrator.device)
    
    print(f"\nInput images shape: {images.shape}")
    
    # Forward through backbone
    print("Running forward pass through backbone...")
    with torch.no_grad():
        features = backbone.forward_features(images)
        patch_tokens = features["x_norm_patchtokens"]
    
    print(f"  Patch tokens shape: {patch_tokens.shape}")
    
    # Forward through head
    print("Running forward pass through head...")
    head.eval()
    with torch.no_grad():
        logits = head(patch_tokens)
    
    print(f"  Logits shape: {logits.shape}")
    
    # Verify shapes
    expected_h, expected_w = h // 14, w // 14
    assert logits.shape == (batch_size, 11, expected_h, expected_w), "Output shape mismatch!"
    
    print("\n✓ Forward pass test successful")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MODEL BUILDING UTILITIES TEST SUITE (Task 6.1)")
    print("=" * 80)
    
    try:
        # Test 1: Backbone loading
        test_backbone_loading()
        
        # Test 2: Segmentation head
        test_segmentation_head()
        
        # Test 3: Build model
        test_build_model()
        
        # Test 4: Forward pass
        test_forward_pass()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nTask 6.1 implementation verified successfully!")
        print("- DINOv2 backbone loading works correctly")
        print("- Segmentation head creation works correctly")
        print("- build_model() function works correctly")
        print("- Frozen backbone with trainable head works correctly")
        print("- Complete forward pass works correctly")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
