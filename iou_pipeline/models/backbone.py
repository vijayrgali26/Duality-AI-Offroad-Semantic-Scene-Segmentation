"""
DINOv2 Backbone Loading Utilities

Provides functions to load DINOv2 vision transformer variants for use as
frozen feature extractors in semantic segmentation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


# Supported DINOv2 variants
DINOV2_VARIANTS = {
    'dinov2_vits14': {
        'model_name': 'dinov2_vits14',
        'embed_dim': 384,
        'patch_size': 14,
        'description': 'DINOv2 ViT-Small with 14x14 patches'
    },
    'dinov2_vitb14': {
        'model_name': 'dinov2_vitb14',
        'embed_dim': 768,
        'patch_size': 14,
        'description': 'DINOv2 ViT-Base with 14x14 patches'
    },
    'dinov2_vitb14_reg': {
        'model_name': 'dinov2_vitb14_reg',
        'embed_dim': 768,
        'patch_size': 14,
        'description': 'DINOv2 ViT-Base with registers and 14x14 patches'
    },
    'dinov2_vitl14': {
        'model_name': 'dinov2_vitl14',
        'embed_dim': 1024,
        'patch_size': 14,
        'description': 'DINOv2 ViT-Large with 14x14 patches'
    },
    'dinov2_vitl14_reg': {
        'model_name': 'dinov2_vitl14_reg',
        'embed_dim': 1024,
        'patch_size': 14,
        'description': 'DINOv2 ViT-Large with registers and 14x14 patches'
    },
    'dinov2_vitg14': {
        'model_name': 'dinov2_vitg14',
        'embed_dim': 1536,
        'patch_size': 14,
        'description': 'DINOv2 ViT-Giant with 14x14 patches'
    },
    'dinov2_vitg14_reg': {
        'model_name': 'dinov2_vitg14_reg',
        'embed_dim': 1536,
        'patch_size': 14,
        'description': 'DINOv2 ViT-Giant with registers and 14x14 patches'
    }
}


def get_backbone_info(backbone_name: str) -> Dict[str, any]:
    """
    Get information about a DINOv2 backbone variant.
    
    Args:
        backbone_name: Name of the backbone variant (e.g., 'dinov2_vits14')
        
    Returns:
        Dictionary with model_name, embed_dim, patch_size, and description
        
    Raises:
        ValueError: If backbone_name is not supported
    """
    if backbone_name not in DINOV2_VARIANTS:
        supported = ', '.join(DINOV2_VARIANTS.keys())
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Supported variants: {supported}"
        )
    return DINOV2_VARIANTS[backbone_name]


def load_dinov2_backbone(
    backbone_name: str = 'dinov2_vits14',
    freeze: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, int, int]:
    """
    Load a DINOv2 backbone model from torch.hub.
    
    Args:
        backbone_name: Name of the DINOv2 variant to load
                      Options: 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitb14_reg',
                               'dinov2_vitl14', 'dinov2_vitl14_reg', 'dinov2_vitg14',
                               'dinov2_vitg14_reg'
        freeze: If True, freeze all backbone parameters (recommended for segmentation)
        device: Device to load the model on (defaults to CUDA if available)
        
    Returns:
        Tuple of (backbone_model, embed_dim, patch_size)
        - backbone_model: The loaded DINOv2 model
        - embed_dim: Embedding dimension of the backbone
        - patch_size: Patch size used by the backbone (typically 14)
        
    Raises:
        ValueError: If backbone_name is not supported
        RuntimeError: If model loading fails
        
    Example:
        >>> backbone, embed_dim, patch_size = load_dinov2_backbone('dinov2_vits14')
        >>> print(f"Loaded backbone with {embed_dim} embedding dimensions")
        Loaded backbone with 384 embedding dimensions
    """
    # Get backbone information
    backbone_info = get_backbone_info(backbone_name)
    model_name = backbone_info['model_name']
    embed_dim = backbone_info['embed_dim']
    patch_size = backbone_info['patch_size']
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load model from torch.hub
        print(f"Loading {backbone_info['description']}...")
        backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=model_name,
            skip_validation=True
        )
        
        # Freeze parameters if requested
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()
            print(f"Backbone parameters frozen (trainable: False)")
        else:
            print(f"Backbone parameters trainable (trainable: True)")
        
        # Move to device
        backbone = backbone.to(device)
        print(f"Backbone loaded successfully on {device}")
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        
        return backbone, embed_dim, patch_size
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv2 backbone '{backbone_name}': {str(e)}"
        )


def compute_patch_grid_size(image_height: int, image_width: int, patch_size: int = 14) -> Tuple[int, int]:
    """
    Compute the number of patches in height and width dimensions.
    
    DINOv2 divides images into non-overlapping patches. The input image dimensions
    should be divisible by the patch size for optimal results.
    
    Args:
        image_height: Height of input image in pixels
        image_width: Width of input image in pixels
        patch_size: Size of each patch (default: 14 for DINOv2)
        
    Returns:
        Tuple of (num_patches_height, num_patches_width)
        
    Example:
        >>> h, w = compute_patch_grid_size(476, 266, 14)
        >>> print(f"Patch grid: {h}x{w}")
        Patch grid: 34x19
    """
    num_patches_h = image_height // patch_size
    num_patches_w = image_width // patch_size
    return num_patches_h, num_patches_w


def align_image_size_to_patches(
    image_height: int,
    image_width: int,
    patch_size: int = 14
) -> Tuple[int, int]:
    """
    Align image dimensions to be divisible by patch size.
    
    Rounds down to the nearest multiple of patch_size to ensure the image
    can be evenly divided into patches.
    
    Args:
        image_height: Desired image height in pixels
        image_width: Desired image width in pixels
        patch_size: Size of each patch (default: 14 for DINOv2)
        
    Returns:
        Tuple of (aligned_height, aligned_width)
        
    Example:
        >>> h, w = align_image_size_to_patches(480, 270, 14)
        >>> print(f"Aligned size: {h}x{w}")
        Aligned size: 476x266
    """
    aligned_h = (image_height // patch_size) * patch_size
    aligned_w = (image_width // patch_size) * patch_size
    return aligned_h, aligned_w


def extract_patch_tokens(
    backbone: nn.Module,
    images: torch.Tensor
) -> torch.Tensor:
    """
    Extract patch tokens from DINOv2 backbone.
    
    Args:
        backbone: DINOv2 backbone model
        images: Input images tensor of shape (B, C, H, W)
        
    Returns:
        Patch tokens tensor of shape (B, N, D) where:
        - B is batch size
        - N is number of patches (H/14 * W/14)
        - D is embedding dimension
        
    Example:
        >>> backbone, _, _ = load_dinov2_backbone('dinov2_vits14')
        >>> images = torch.randn(2, 3, 476, 266)
        >>> tokens = extract_patch_tokens(backbone, images)
        >>> print(tokens.shape)
        torch.Size([2, 646, 384])
    """
    with torch.no_grad():
        features = backbone.forward_features(images)
        patch_tokens = features["x_norm_patchtokens"]
    return patch_tokens


def list_available_backbones() -> None:
    """
    Print all available DINOv2 backbone variants with their specifications.
    
    Example:
        >>> list_available_backbones()
        Available DINOv2 Backbones:
        ===========================
        dinov2_vits14
          Description: DINOv2 ViT-Small with 14x14 patches
          Embedding Dim: 384
          Patch Size: 14x14
        ...
    """
    print("Available DINOv2 Backbones:")
    print("=" * 50)
    for name, info in DINOV2_VARIANTS.items():
        print(f"\n{name}")
        print(f"  Description: {info['description']}")
        print(f"  Embedding Dim: {info['embed_dim']}")
        print(f"  Patch Size: {info['patch_size']}x{info['patch_size']}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Demo usage
    print("DINOv2 Backbone Utilities Demo\n")
    
    # List available backbones
    list_available_backbones()
    
    # Load a backbone
    print("\nLoading dinov2_vits14 backbone...")
    backbone, embed_dim, patch_size = load_dinov2_backbone('dinov2_vits14')
    
    # Compute patch grid for common image size
    print("\nComputing patch grid for 480x270 image:")
    aligned_h, aligned_w = align_image_size_to_patches(480, 270, patch_size)
    print(f"  Aligned size: {aligned_h}x{aligned_w}")
    
    num_patches_h, num_patches_w = compute_patch_grid_size(aligned_h, aligned_w, patch_size)
    print(f"  Patch grid: {num_patches_h}x{num_patches_w}")
    print(f"  Total patches: {num_patches_h * num_patches_w}")
