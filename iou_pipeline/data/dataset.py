"""
Dataset classes for semantic segmentation.

This module provides PyTorch Dataset classes for loading color images and
segmentation masks with support for data augmentation and quality validation.
"""

import os
import logging
from typing import Optional, Callable, Tuple, Dict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# Mask value mapping from raw pixel values to class IDs
# Based on the Offroad Segmentation Training Dataset format
VALUE_MAP = {
    0: 0,       # background
    100: 1,     # Trees
    200: 2,     # Lush Bushes
    300: 3,     # Dry Grass
    500: 4,     # Dry Bushes
    550: 5,     # Ground Clutter
    600: 6,     # Flowers
    700: 7,     # Logs
    800: 8,     # Rocks
    7100: 9,    # Landscape
    10000: 10   # Sky
}

NUM_CLASSES = len(VALUE_MAP)  # 11 classes


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for semantic segmentation with color images and masks.
    
    This dataset loads RGB images and corresponding segmentation masks from
    a directory structure with 'Color_Images' and 'Segmentation' subdirectories.
    It handles mask value mapping from raw values to class IDs and provides
    graceful error handling for corrupted files.
    
    Directory structure expected:
        data_dir/
            Color_Images/
                image1.png
                image2.png
                ...
            Segmentation/
                image1.png
                image2.png
                ...
    
    Args:
        data_dir: Root directory containing 'Color_Images' and 'Segmentation' subdirs
        transform: Optional transform to apply to images (e.g., torchvision.transforms)
        mask_transform: Optional transform to apply to masks
        value_map: Optional custom mapping from raw mask values to class IDs.
                   Defaults to VALUE_MAP defined in this module.
        validate_on_init: If True, validates all files during initialization and
                         removes corrupted entries. If False, validation happens
                         during __getitem__ with logging. Default: False
    
    Attributes:
        image_dir: Path to color images directory
        masks_dir: Path to segmentation masks directory
        data_ids: List of valid image filenames
        num_classes: Number of segmentation classes
    
    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((266, 476)),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        ...                         std=[0.229, 0.224, 0.225])
        ... ])
        >>> mask_transform = transforms.Compose([
        ...     transforms.Resize((266, 476)),
        ...     transforms.ToTensor(),
        ... ])
        >>> dataset = SegmentationDataset(
        ...     data_dir='./data/train',
        ...     transform=transform,
        ...     mask_transform=mask_transform
        ... )
        >>> image, mask = dataset[0]
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        value_map: Optional[Dict[int, int]] = None,
        validate_on_init: bool = False
    ):
        """Initialize the SegmentationDataset."""
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'Color_Images'
        self.masks_dir = self.data_dir / 'Segmentation'
        self.transform = transform
        self.mask_transform = mask_transform
        self.value_map = value_map if value_map is not None else VALUE_MAP
        self.num_classes = len(self.value_map)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validate directory structure
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )
        if not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {self.masks_dir}"
            )
        
        # Get list of image files
        self.data_ids = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(self.data_ids) == 0:
            raise ValueError(
                f"No image files found in {self.image_dir}"
            )
        
        # Validate files if requested
        if validate_on_init:
            self._validate_dataset()
        
        self.logger.info(
            f"Initialized SegmentationDataset with {len(self.data_ids)} samples "
            f"from {self.data_dir}"
        )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a sample from the dataset.
        
        Args:
            idx: Index of the sample to load
        
        Returns:
            Tuple of (image, mask) where:
                - image: Transformed RGB image tensor
                - mask: Transformed segmentation mask tensor with class IDs
        
        Raises:
            RuntimeError: If the image or mask file is corrupted or cannot be loaded
        """
        data_id = self.data_ids[idx]
        img_path = self.image_dir / data_id
        mask_path = self.masks_dir / data_id
        
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Load and convert mask
            mask = Image.open(mask_path)
            mask = self._convert_mask(mask)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            if self.mask_transform:
                mask = self.mask_transform(mask)
            
            return image, mask
            
        except Exception as e:
            self.logger.error(
                f"Error loading sample {idx} (file: {data_id}): {str(e)}"
            )
            raise RuntimeError(
                f"Failed to load sample {idx} from {img_path} and {mask_path}"
            ) from e
    
    def _convert_mask(self, mask: Image.Image) -> Image.Image:
        """
        Convert raw mask values to class IDs using the value map.
        
        This method maps raw pixel values in the segmentation mask to
        sequential class IDs (0 to num_classes-1).
        
        Args:
            mask: PIL Image containing raw mask values
        
        Returns:
            PIL Image with mapped class IDs
        """
        arr = np.array(mask)
        
        # Handle multi-channel masks
        if len(arr.shape) == 3:
            arr = arr[:, :, 0]
        
        new_arr = np.zeros_like(arr, dtype=np.uint8)
        
        # Map known values
        for raw_value, class_id in self.value_map.items():
            mask_indices = arr == raw_value
            new_arr[mask_indices] = class_id
        
        # Check for unmapped values and map them to background (class 0)
        unique_values = np.unique(arr)
        unmapped_values = [int(v) for v in unique_values if v not in self.value_map]
        
        if unmapped_values:
            self.logger.warning(
                f"Mask contains unmapped values: {unmapped_values}. "
                f"Mapping to class 0 (background)."
            )
            # Unmapped values are already 0 in new_arr, so no action needed
        
        return Image.fromarray(new_arr)
    
    def _validate_dataset(self):
        """
        Validate all files in the dataset and remove corrupted entries.
        
        This method checks that:
        1. Both image and mask files exist for each data_id
        2. Files can be opened without errors
        3. Masks contain only valid values from value_map
        
        Corrupted or invalid files are removed from data_ids and logged.
        """
        valid_ids = []
        corrupted_count = 0
        
        self.logger.info(f"Validating {len(self.data_ids)} samples...")
        
        for data_id in self.data_ids:
            img_path = self.image_dir / data_id
            mask_path = self.masks_dir / data_id
            
            try:
                # Check if both files exist
                if not mask_path.exists():
                    self.logger.warning(
                        f"Missing mask file for {data_id}, skipping"
                    )
                    corrupted_count += 1
                    continue
                
                # Try to open and validate image
                with Image.open(img_path) as img:
                    img.verify()
                
                # Try to open and validate mask
                with Image.open(mask_path) as mask:
                    mask.verify()
                
                # Re-open mask to check values (verify() closes the file)
                mask = Image.open(mask_path)
                mask_arr = np.array(mask)
                unique_values = np.unique(mask_arr)
                
                # Check if all values are valid
                invalid_values = [
                    v for v in unique_values if v not in self.value_map
                ]
                
                if invalid_values:
                    self.logger.warning(
                        f"Mask {data_id} contains invalid values: {invalid_values}, "
                        f"skipping"
                    )
                    corrupted_count += 1
                    continue
                
                # File is valid
                valid_ids.append(data_id)
                
            except Exception as e:
                self.logger.warning(
                    f"Corrupted file {data_id}: {str(e)}, skipping"
                )
                corrupted_count += 1
                continue
        
        # Update data_ids with only valid entries
        self.data_ids = valid_ids
        
        if corrupted_count > 0:
            self.logger.warning(
                f"Removed {corrupted_count} corrupted/invalid samples. "
                f"{len(self.data_ids)} valid samples remaining."
            )
        else:
            self.logger.info("All samples validated successfully.")
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Compute pixel count for each class across the entire dataset.
        
        This method iterates through all masks and counts pixels for each class.
        Useful for analyzing class imbalance and computing class weights.
        
        Returns:
            Dictionary mapping class_id to total pixel count
        
        Example:
            >>> dataset = SegmentationDataset('./data/train')
            >>> distribution = dataset.get_class_distribution()
            >>> print(f"Background pixels: {distribution[0]}")
        """
        class_counts = {class_id: 0 for class_id in range(self.num_classes)}
        
        self.logger.info("Computing class distribution...")
        
        for idx in range(len(self)):
            data_id = self.data_ids[idx]
            mask_path = self.masks_dir / data_id
            
            try:
                mask = Image.open(mask_path)
                mask = self._convert_mask(mask)
                mask_arr = np.array(mask)
                
                # Count pixels for each class
                unique, counts = np.unique(mask_arr, return_counts=True)
                for class_id, count in zip(unique, counts):
                    if class_id < self.num_classes:
                        class_counts[int(class_id)] += int(count)
                
            except Exception as e:
                self.logger.warning(
                    f"Error processing mask {data_id} for distribution: {str(e)}"
                )
                continue
        
        self.logger.info("Class distribution computed successfully.")
        return class_counts
    
    def get_sample_info(self, idx: int) -> Dict[str, any]:
        """
        Get metadata about a specific sample without loading the full data.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary with sample metadata including filename and paths
        """
        data_id = self.data_ids[idx]
        return {
            'index': idx,
            'filename': data_id,
            'image_path': str(self.image_dir / data_id),
            'mask_path': str(self.masks_dir / data_id),
        }
