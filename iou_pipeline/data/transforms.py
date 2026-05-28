"""
Data transformation utilities for semantic segmentation.

This module provides transform pipelines for training and validation datasets,
including normalization for DINOv2 input requirements and resize transforms
for 14×14 patch alignment.
"""

from typing import Tuple, Optional
import torch
from PIL import Image
import numpy as np


# DINOv2 normalization parameters (ImageNet statistics)
DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD = [0.229, 0.224, 0.225]

# Target resolution for 14×14 patch alignment with DINOv2-ViT
# Original: 480×270, Aligned: 476×266 (34×19 patches of 14×14)
TARGET_HEIGHT = 266
TARGET_WIDTH = 476


class Resize:
    """Resize image to target size."""
    def __init__(self, size):
        self.size = size  # (height, width)
    
    def __call__(self, img):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR)


class ToTensor:
    """Convert PIL Image to tensor."""
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32) / 255.0
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        return torch.from_numpy(img_array).permute(2, 0, 1)


class Normalize:
    """Normalize tensor with mean and std."""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ResizeWithPadding:
    """
    Resize image to target size while maintaining aspect ratio with padding.
    
    This transform resizes the image to fit within the target dimensions
    while preserving aspect ratio, then pads to reach exact target size.
    
    Args:
        target_size: Tuple of (height, width) for output size
        fill: Fill value for padding (default: 0 for black)
    """
    
    def __init__(self, target_size: Tuple[int, int], fill: int = 0):
        self.target_height, self.target_width = target_size
        self.fill = fill
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply resize with padding to image."""
        # Get original dimensions
        orig_width, orig_height = img.size
        
        # Calculate scaling factor to fit within target while preserving aspect ratio
        scale = min(
            self.target_width / orig_width,
            self.target_height / orig_height
        )
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.BILINEAR)
        
        # Create new image with target size and fill color
        new_img = Image.new(img.mode, (self.target_width, self.target_height), self.fill)
        
        # Calculate padding to center the image
        paste_x = (self.target_width - new_width) // 2
        paste_y = (self.target_height - new_height) // 2
        
        # Paste resized image onto padded canvas
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img


class ResizeMask:
    """
    Resize segmentation mask using nearest neighbor interpolation.
    
    This transform resizes masks while preserving class labels by using
    nearest neighbor interpolation instead of bilinear.
    
    Args:
        target_size: Tuple of (height, width) for output size
    """
    
    def __init__(self, target_size: Tuple[int, int]):
        self.target_height, self.target_width = target_size
    
    def __call__(self, mask: Image.Image) -> Image.Image:
        """Apply resize to mask using nearest neighbor interpolation."""
        return mask.resize(
            (self.target_width, self.target_height),
            Image.NEAREST
        )


class ToTensorMask:
    """
    Convert PIL Image mask to PyTorch tensor without normalization.
    
    Unlike torchvision.transforms.ToTensor, this preserves integer class labels
    and does not normalize to [0, 1] range.
    """
    
    def __call__(self, mask: Image.Image) -> torch.Tensor:
        """Convert mask to tensor."""
        mask_array = np.array(mask, dtype=np.int64)
        return torch.from_numpy(mask_array)


def get_training_transforms(
    target_size: Tuple[int, int] = (TARGET_HEIGHT, TARGET_WIDTH),
    normalize: bool = True,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> Compose:
    """
    Get training data transforms for images.
    
    This pipeline includes:
    - Resize to target dimensions for 14×14 patch alignment
    - Convert to tensor
    - Normalize using DINOv2 (ImageNet) statistics
    
    Args:
        target_size: Target (height, width) for resizing. Default: (266, 476)
        normalize: Whether to apply normalization. Default: True
        mean: Custom normalization mean. Default: DINOv2_MEAN
        std: Custom normalization std. Default: DINOv2_STD
    
    Returns:
        Composed transform pipeline for training images
    
    Example:
        >>> transform = get_training_transforms()
        >>> image = Image.open('image.png')
        >>> tensor = transform(image)
        >>> print(tensor.shape)  # torch.Size([3, 266, 476])
    """
    transform_list = [
        Resize(target_size),
        ToTensor(),
    ]
    
    if normalize:
        norm_mean = mean if mean is not None else DINOV2_MEAN
        norm_std = std if std is not None else DINOV2_STD
        transform_list.append(Normalize(norm_mean, norm_std))
    
    return Compose(transform_list)


def get_validation_transforms(
    target_size: Tuple[int, int] = (TARGET_HEIGHT, TARGET_WIDTH),
    normalize: bool = True,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> Compose:
    """
    Get validation data transforms for images.
    
    Identical to training transforms but without augmentation.
    This pipeline includes:
    - Resize to target dimensions for 14×14 patch alignment
    - Convert to tensor
    - Normalize using DINOv2 (ImageNet) statistics
    
    Args:
        target_size: Target (height, width) for resizing. Default: (266, 476)
        normalize: Whether to apply normalization. Default: True
        mean: Custom normalization mean. Default: DINOv2_MEAN
        std: Custom normalization std. Default: DINOv2_STD
    
    Returns:
        Composed transform pipeline for validation images
    
    Example:
        >>> transform = get_validation_transforms()
        >>> image = Image.open('image.png')
        >>> tensor = transform(image)
        >>> print(tensor.shape)  # torch.Size([3, 266, 476])
    """
    # Validation transforms are identical to training (no augmentation here)
    # Augmentation is handled separately in the DatasetEditor
    return get_training_transforms(
        target_size=target_size,
        normalize=normalize,
        mean=mean,
        std=std
    )


def get_mask_transforms(
    target_size: Tuple[int, int] = (TARGET_HEIGHT, TARGET_WIDTH)
) -> Compose:
    """
    Get transforms for segmentation masks.
    
    This pipeline includes:
    - Resize to target dimensions using nearest neighbor interpolation
    - Convert to tensor (preserving integer class labels)
    
    Args:
        target_size: Target (height, width) for resizing. Default: (266, 476)
    
    Returns:
        Composed transform pipeline for segmentation masks
    
    Example:
        >>> transform = get_mask_transforms()
        >>> mask = Image.open('mask.png')
        >>> tensor = transform(mask)
        >>> print(tensor.shape)  # torch.Size([266, 476])
        >>> print(tensor.dtype)  # torch.int64
    """
    return Compose([
        ResizeMask(target_size),
        ToTensorMask(),
    ])


def denormalize_image(
    tensor: torch.Tensor,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> torch.Tensor:
    """
    Denormalize a normalized image tensor for visualization.
    
    Reverses the normalization applied during preprocessing to convert
    the tensor back to [0, 1] range for visualization.
    
    Args:
        tensor: Normalized image tensor of shape (C, H, W) or (B, C, H, W)
        mean: Normalization mean used. Default: DINOv2_MEAN
        std: Normalization std used. Default: DINOv2_STD
    
    Returns:
        Denormalized tensor in [0, 1] range
    
    Example:
        >>> normalized = transform(image)
        >>> denormalized = denormalize_image(normalized)
    """
    mean = mean if mean is not None else DINOV2_MEAN
    std = std if std is not None else DINOV2_STD
    
    # Convert to tensors
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    
    # Handle batch dimension
    if tensor.dim() == 4:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    
    # Denormalize: x_original = x_normalized * std + mean
    denormalized = tensor * std_tensor + mean_tensor
    
    # Clamp to [0, 1] range
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized


def get_augmented_training_transforms(
    target_size: Tuple[int, int] = (TARGET_HEIGHT, TARGET_WIDTH),
    normalize: bool = True,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    color_jitter: bool = True,
    random_flip: bool = True
) -> Compose:
    """
    Get augmented training transforms with color jittering and flipping.
    
    Note: For augmentations that require synchronized image-mask transforms
    (rotation, etc.), use the DatasetEditor module with albumentations.
    
    Args:
        target_size: Target (height, width) for resizing. Default: (266, 476)
        normalize: Whether to apply normalization. Default: True
        mean: Custom normalization mean. Default: DINOv2_MEAN
        std: Custom normalization std. Default: DINOv2_STD
        color_jitter: Whether to apply color jittering. Default: True
        random_flip: Whether to apply random horizontal flip. Default: True
    
    Returns:
        Composed transform pipeline with augmentations
    """
    # For now, return basic transforms
    # Advanced augmentations are handled by DatasetEditor
    return get_training_transforms(target_size, normalize, mean, std)


# Convenience function to get all transforms at once
def get_transform_pipelines(
    target_size: Tuple[int, int] = (TARGET_HEIGHT, TARGET_WIDTH),
    use_augmentation: bool = False
) -> dict:
    """
    Get all transform pipelines in a single dictionary.
    
    Args:
        target_size: Target (height, width) for resizing. Default: (266, 476)
        use_augmentation: Whether to use augmented training transforms. Default: False
    
    Returns:
        Dictionary with keys:
            - 'train': Training image transforms
            - 'val': Validation image transforms
            - 'mask': Mask transforms
    
    Example:
        >>> transforms = get_transform_pipelines()
        >>> train_dataset = SegmentationDataset(
        ...     data_dir='./data/train',
        ...     transform=transforms['train'],
        ...     mask_transform=transforms['mask']
        ... )
    """
    if use_augmentation:
        train_transform = get_augmented_training_transforms(target_size)
    else:
        train_transform = get_training_transforms(target_size)
    
    return {
        'train': train_transform,
        'val': get_validation_transforms(target_size),
        'mask': get_mask_transforms(target_size)
    }
