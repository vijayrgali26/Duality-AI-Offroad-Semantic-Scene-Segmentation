"""
Dataset Analyzer Module

Analyzes training datasets to identify class imbalances, quality issues,
and performance bottlenecks.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, label
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """Represents a quality issue found in the dataset."""
    image_id: str
    issue_type: str  # 'missing_label', 'boundary_error', 'noise', 'invalid_values'
    severity: str    # 'low', 'medium', 'high'
    details: Dict[str, Any]


@dataclass
class AnalysisReport:
    """Comprehensive analysis report of the dataset."""
    class_distribution: Dict[int, int]
    class_balance_ratios: Dict[int, float]
    imbalanced_classes: List[int]  # Classes with ratio < 0.3
    quality_issues: List[QualityIssue]
    per_class_iou: Dict[int, float]
    poorly_performing_classes: List[int]  # Classes with IoU < 0.4
    total_images: int
    total_pixels: int
    timestamp: str


class DatasetAnalyzer:
    """
    Analyzes the training dataset to identify class imbalances, quality issues,
    and performance bottlenecks.
    """
    
    def __init__(self, dataset_path: str, model_path: Optional[str] = None):
        """
        Initialize analyzer with dataset path and optional baseline model.
        
        Args:
            dataset_path: Path to training dataset root
            model_path: Optional path to baseline model for IoU analysis
        """
        self.dataset_path = Path(dataset_path)
        self.model_path = model_path
        
        # Setup paths for images and masks
        self.image_dir = self.dataset_path / 'Color_Images'
        self.masks_dir = self.dataset_path / 'Segmentation'
        
        # Validate paths exist
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory does not exist: {self.masks_dir}")
        
        # Cache for loaded data
        self._image_files: Optional[List[Path]] = None
        self._mask_files: Optional[List[Path]] = None
        
        logger.info(f"Initialized DatasetAnalyzer with dataset: {self.dataset_path}")
    
    def _get_image_mask_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Get list of matching image-mask file pairs.
        
        Returns:
            List of (image_path, mask_path) tuples
        """
        if self._image_files is None:
            # Get all image files (support common formats)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            self._image_files = sorted([
                f for f in self.image_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ])
            logger.info(f"Found {len(self._image_files)} images in {self.image_dir}")
        
        if self._mask_files is None:
            # Get all mask files
            mask_extensions = {'.png', '.jpg', '.jpeg'}
            self._mask_files = sorted([
                f for f in self.masks_dir.iterdir()
                if f.suffix.lower() in mask_extensions
            ])
            logger.info(f"Found {len(self._mask_files)} masks in {self.masks_dir}")
        
        # Match images to masks by filename (without extension)
        pairs = []
        mask_dict = {m.stem: m for m in self._mask_files}
        
        for img_path in self._image_files:
            img_stem = img_path.stem
            if img_stem in mask_dict:
                pairs.append((img_path, mask_dict[img_stem]))
            else:
                logger.warning(f"No matching mask found for image: {img_path.name}")
        
        logger.info(f"Matched {len(pairs)} image-mask pairs")
        return pairs
    
    def _load_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        """
        Load a segmentation mask from file.
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            Mask as numpy array, or None if loading fails
        """
        try:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            
            # Handle multi-channel masks (convert to single channel)
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]
            
            return mask_array
        except Exception as e:
            logger.error(f"Failed to load mask {mask_path}: {e}")
            return None
    
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load an RGB image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (H, W, 3), or None if loading fails
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
        
    def compute_class_distribution(self) -> Dict[int, int]:
        """
        Compute pixel count for each terrain class using NumPy vectorization.
        
        Returns:
            Dictionary mapping class_id to pixel count
        """
        logger.info("Computing class distribution across dataset...")
        
        # Initialize distribution dictionary
        class_distribution = {}
        
        # Get all image-mask pairs
        pairs = self._get_image_mask_pairs()
        
        if not pairs:
            logger.warning("No image-mask pairs found")
            return class_distribution
        
        # Process each mask
        for idx, (img_path, mask_path) in enumerate(pairs):
            mask = self._load_mask(mask_path)
            
            if mask is None:
                continue
            
            # Use NumPy's bincount for efficient pixel counting
            # bincount returns counts for each value from 0 to max value
            unique_values, counts = np.unique(mask, return_counts=True)
            
            # Accumulate counts for each class
            for class_id, count in zip(unique_values, counts):
                class_id = int(class_id)
                if class_id in class_distribution:
                    class_distribution[class_id] += int(count)
                else:
                    class_distribution[class_id] = int(count)
            
            # Log progress every 100 images
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(pairs)} masks")
        
        logger.info(f"Class distribution computed for {len(pairs)} masks")
        logger.info(f"Found {len(class_distribution)} unique classes: {sorted(class_distribution.keys())}")
        
        return class_distribution
        
    def compute_class_balance_ratio(self, distribution: Dict[int, int]) -> Dict[int, float]:
        """
        Calculate balance ratio for each class relative to most frequent.
        Normalized to max class (most frequent class has ratio 1.0).
        
        Args:
            distribution: Class pixel counts
            
        Returns:
            Dictionary mapping class_id to balance ratio (0.0 to 1.0)
        """
        if not distribution:
            logger.warning("Empty distribution provided")
            return {}
        
        # Find the maximum pixel count (most frequent class)
        max_count = max(distribution.values())
        
        if max_count == 0:
            logger.warning("Maximum count is zero")
            return {class_id: 0.0 for class_id in distribution.keys()}
        
        # Compute ratio for each class normalized to max
        balance_ratios = {
            class_id: count / max_count
            for class_id, count in distribution.items()
        }
        
        logger.info(f"Computed balance ratios for {len(balance_ratios)} classes")
        
        # Log classes with low balance ratios
        imbalanced = [class_id for class_id, ratio in balance_ratios.items() if ratio < 0.3]
        if imbalanced:
            logger.warning(f"Found {len(imbalanced)} imbalanced classes (ratio < 0.3): {imbalanced}")
        
        return balance_ratios
    
    def identify_imbalanced_classes(self, balance_ratios: Dict[int, float], 
                                    threshold: float = 0.3) -> List[int]:
        """
        Identify classes with balance ratio below threshold.
        
        Args:
            balance_ratios: Dictionary mapping class_id to balance ratio
            threshold: Threshold for identifying imbalanced classes (default: 0.3)
            
        Returns:
            List of class IDs with ratio below threshold
        """
        imbalanced = [
            class_id for class_id, ratio in balance_ratios.items()
            if ratio < threshold
        ]
        
        logger.info(f"Identified {len(imbalanced)} imbalanced classes with ratio < {threshold}")
        
        return sorted(imbalanced)
    
    def generate_class_distribution_plot(self, distribution: Dict[int, int], 
                                        output_path: Path) -> None:
        """
        Generate matplotlib bar plot for class distribution visualization.
        
        Args:
            distribution: Dictionary mapping class_id to pixel count
            output_path: Path to save the plot image
        """
        if not distribution:
            logger.warning("Empty distribution, skipping plot generation")
            return
        
        # Sort by class ID for consistent visualization
        class_ids = sorted(distribution.keys())
        pixel_counts = [distribution[cid] for cid in class_ids]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        bars = ax.bar(class_ids, pixel_counts, color='steelblue', edgecolor='black')
        
        # Customize plot
        ax.set_xlabel('Class ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pixel Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution Across Dataset', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Format y-axis with scientific notation if needed
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        
        # Add value labels on top of bars
        for bar, count in zip(bars, pixel_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}',
                   ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Ensure integer x-axis ticks
        ax.set_xticks(class_ids)
        
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved class distribution plot to {output_path}")
    
    def generate_balance_ratio_plot(self, balance_ratios: Dict[int, float],
                                   output_path: Path,
                                   threshold: float = 0.3) -> None:
        """
        Generate matplotlib bar plot for class balance ratios with threshold line.
        
        Args:
            balance_ratios: Dictionary mapping class_id to balance ratio
            output_path: Path to save the plot image
            threshold: Threshold line to mark imbalanced classes (default: 0.3)
        """
        if not balance_ratios:
            logger.warning("Empty balance ratios, skipping plot generation")
            return
        
        # Sort by class ID for consistent visualization
        class_ids = sorted(balance_ratios.keys())
        ratios = [balance_ratios[cid] for cid in class_ids]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color bars based on threshold
        colors = ['red' if r < threshold else 'green' for r in ratios]
        bars = ax.bar(class_ids, ratios, color=colors, edgecolor='black', alpha=0.7)
        
        # Add threshold line
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                  label=f'Imbalance Threshold ({threshold})')
        
        # Customize plot
        ax.set_xlabel('Class ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Balance Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Class Balance Ratios (Normalized to Max Class)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right')
        
        # Add value labels on top of bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Ensure integer x-axis ticks
        ax.set_xticks(class_ids)
        
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved balance ratio plot to {output_path}")
    
    def generate_iou_plot(self, per_class_iou: Dict[int, float],
                         output_path: Path,
                         threshold: float = 0.4) -> None:
        """
        Generate matplotlib bar plot for per-class IoU scores with threshold line.
        
        Args:
            per_class_iou: Dictionary mapping class_id to IoU score
            output_path: Path to save the plot image
            threshold: Threshold line to mark poorly performing classes (default: 0.4)
        """
        if not per_class_iou:
            logger.warning("Empty IoU scores, skipping plot generation")
            return
        
        # Sort by class ID for consistent visualization
        class_ids = sorted(per_class_iou.keys())
        iou_scores = [per_class_iou[cid] for cid in class_ids]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color bars based on threshold
        colors = ['red' if iou < threshold else 'green' for iou in iou_scores]
        bars = ax.bar(class_ids, iou_scores, color=colors, edgecolor='black', alpha=0.7)
        
        # Add threshold line
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                  label=f'Poor Performance Threshold ({threshold})')
        
        # Customize plot
        ax.set_xlabel('Class ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class IoU Scores (Baseline Model)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right')
        
        # Add value labels on top of bars
        for bar, iou in zip(bars, iou_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{iou:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Ensure integer x-axis ticks
        ax.set_xticks(class_ids)
        
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved per-class IoU plot to {output_path}")
        
    def detect_quality_issues(self) -> List[QualityIssue]:
        """
        Identify images with annotation problems.
        
        Detects:
        - Invalid mask values (outside valid class indices 0-10)
        - Boundary errors using morphological operations
        - Noise using connected component analysis
        - Missing labels and corrupted files
        
        Returns:
            List of QualityIssue objects with image_id, issue_type, severity
        """
        logger.info("Starting quality issue detection...")
        quality_issues = []
        
        # Get all image-mask pairs
        pairs = self._get_image_mask_pairs()
        
        for img_path, mask_path in pairs:
            image_id = img_path.stem
            
            # Load mask
            mask = self._load_mask(mask_path)
            
            # Check for corrupted/missing files
            if mask is None:
                quality_issues.append(QualityIssue(
                    image_id=image_id,
                    issue_type='corrupted_file',
                    severity='high',
                    details={'reason': 'Failed to load mask file'}
                ))
                continue
            
            # 1. Validate mask values against valid class indices [0-10]
            unique_values = np.unique(mask)
            invalid_values = [int(v) for v in unique_values if v < 0 or v > 10]
            
            if invalid_values:
                num_invalid_pixels = sum((mask == v).sum() for v in invalid_values)
                total_pixels = mask.size
                invalid_ratio = num_invalid_pixels / total_pixels
                
                # Determine severity based on ratio of invalid pixels
                if invalid_ratio > 0.1:
                    severity = 'high'
                elif invalid_ratio > 0.01:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                quality_issues.append(QualityIssue(
                    image_id=image_id,
                    issue_type='invalid_values',
                    severity=severity,
                    details={
                        'invalid_values': invalid_values,
                        'num_invalid_pixels': int(num_invalid_pixels),
                        'invalid_ratio': float(invalid_ratio)
                    }
                ))
            
            # 2. Detect boundary errors using morphological operations
            boundary_issues = self._detect_boundary_errors(mask, image_id)
            if boundary_issues:
                quality_issues.append(boundary_issues)
            
            # 3. Detect noise using connected component analysis
            noise_issues = self._detect_noise(mask, image_id)
            if noise_issues:
                quality_issues.append(noise_issues)
            
            # 4. Detect missing labels (classes that should be present but aren't)
            missing_label_issues = self._detect_missing_labels(mask, image_id)
            if missing_label_issues:
                quality_issues.append(missing_label_issues)
        
        logger.info(f"Quality issue detection complete. Found {len(quality_issues)} issues.")
        return quality_issues
    
    def _detect_boundary_errors(self, mask: np.ndarray, image_id: str) -> Optional[QualityIssue]:
        """
        Detect boundary errors using morphological operations.
        
        Boundary errors are detected by comparing the mask with its morphologically
        processed versions. Significant differences indicate jagged or irregular boundaries.
        
        Args:
            mask: Segmentation mask array
            image_id: Image identifier
            
        Returns:
            QualityIssue if boundary errors detected, None otherwise
        """
        try:
            # Define structuring element for morphological operations (3x3 square)
            struct = ndimage.generate_binary_structure(2, 2)
            
            # Process each class separately to preserve class boundaries
            total_close_diff = 0
            total_open_diff = 0
            
            for class_id in range(11):  # Classes 0-10
                # Create binary mask for this class
                class_mask = (mask == class_id)
                
                if not class_mask.any():
                    continue  # Skip if class not present
                
                # Apply morphological closing (dilation followed by erosion)
                # This fills small holes and smooths boundaries
                dilated = binary_dilation(class_mask, structure=struct)
                closed = binary_erosion(dilated, structure=struct)
                
                # Apply morphological opening (erosion followed by dilation)
                # This removes small noise and smooths boundaries
                eroded = binary_erosion(class_mask, structure=struct)
                opened = binary_dilation(eroded, structure=struct)
                
                # Calculate difference between original and processed masks
                total_close_diff += np.sum(class_mask != closed)
                total_open_diff += np.sum(class_mask != opened)
            
            total_pixels = mask.size
            
            # Calculate boundary roughness ratio
            boundary_roughness = (total_close_diff + total_open_diff) / (2 * total_pixels)
            
            # Threshold for detecting boundary errors
            # Higher values indicate more irregular boundaries
            if boundary_roughness > 0.05:  # More than 5% difference
                severity = 'high'
            elif boundary_roughness > 0.02:  # 2-5% difference
                severity = 'medium'
            elif boundary_roughness > 0.01:  # 1-2% difference
                severity = 'low'
            else:
                return None  # No significant boundary errors
            
            return QualityIssue(
                image_id=image_id,
                issue_type='boundary_error',
                severity=severity,
                details={
                    'boundary_roughness': float(boundary_roughness),
                    'affected_pixels_close': int(total_close_diff),
                    'affected_pixels_open': int(total_open_diff)
                }
            )
        except Exception as e:
            logger.warning(f"Failed to detect boundary errors for {image_id}: {e}")
            return None
    
    def _detect_noise(self, mask: np.ndarray, image_id: str) -> Optional[QualityIssue]:
        """
        Detect noise using connected component analysis.
        
        Noise is detected by finding small isolated regions that are likely
        annotation errors rather than legitimate objects.
        
        Args:
            mask: Segmentation mask array
            image_id: Image identifier
            
        Returns:
            QualityIssue if noise detected, None otherwise
        """
        try:
            # Analyze each class separately
            noisy_components = []
            total_noise_pixels = 0
            
            for class_id in range(11):  # Classes 0-10
                # Create binary mask for this class
                class_mask = (mask == class_id)
                
                if not class_mask.any():
                    continue  # Skip if class not present
                
                # Find connected components using scipy
                labeled_array, num_features = label(class_mask)
                
                # Analyze each component
                for i in range(1, num_features + 1):
                    component_mask = (labeled_array == i)
                    area = np.sum(component_mask)
                    
                    # Consider components smaller than 50 pixels as potential noise
                    if area < 50:
                        noisy_components.append({
                            'class_id': int(class_id),
                            'component_id': i,
                            'area': int(area)
                        })
                        total_noise_pixels += area
            
            if not noisy_components:
                return None  # No noise detected
            
            # Calculate noise ratio
            total_pixels = mask.size
            noise_ratio = total_noise_pixels / total_pixels
            
            # Determine severity based on noise ratio and number of components
            if noise_ratio > 0.05 or len(noisy_components) > 100:
                severity = 'high'
            elif noise_ratio > 0.02 or len(noisy_components) > 50:
                severity = 'medium'
            else:
                severity = 'low'
            
            return QualityIssue(
                image_id=image_id,
                issue_type='noise',
                severity=severity,
                details={
                    'num_noisy_components': len(noisy_components),
                    'total_noise_pixels': int(total_noise_pixels),
                    'noise_ratio': float(noise_ratio),
                    'components': noisy_components[:10]  # Store first 10 for reference
                }
            )
        except Exception as e:
            logger.warning(f"Failed to detect noise for {image_id}: {e}")
            return None
    
    def _detect_missing_labels(self, mask: np.ndarray, image_id: str) -> Optional[QualityIssue]:
        """
        Detect missing labels in the mask.
        
        This checks if the mask has very few classes labeled, which might indicate
        incomplete annotation. For outdoor scenes, we expect multiple terrain classes.
        
        Args:
            mask: Segmentation mask array
            image_id: Image identifier
            
        Returns:
            QualityIssue if missing labels detected, None otherwise
        """
        try:
            # Count unique classes present in the mask
            unique_classes = np.unique(mask)
            num_classes = len(unique_classes)
            
            # For outdoor terrain segmentation, we expect at least 3-4 different classes
            # (e.g., sky, ground, vegetation)
            if num_classes < 2:
                severity = 'high'
            elif num_classes < 3:
                severity = 'medium'
            else:
                return None  # Sufficient class diversity
            
            return QualityIssue(
                image_id=image_id,
                issue_type='missing_label',
                severity=severity,
                details={
                    'num_classes_present': int(num_classes),
                    'classes_present': [int(c) for c in unique_classes],
                    'expected_min_classes': 3
                }
            )
        except Exception as e:
            logger.warning(f"Failed to detect missing labels for {image_id}: {e}")
            return None
        
    def compute_baseline_per_class_iou(self, model_path: str) -> Dict[int, float]:
        """
        Compute per-class IoU using baseline model predictions.
        
        Loads the baseline model checkpoint, runs inference on the training dataset,
        and computes IoU scores for each class to identify poorly performing classes.
        
        Args:
            model_path: Path to baseline model checkpoint (.pth file)
            
        Returns:
            Dictionary mapping class_id to IoU score
        """
        logger.info(f"Computing baseline per-class IoU using model: {model_path}")
        
        # Import PyTorch dependencies
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import Dataset, DataLoader
        except ImportError as e:
            logger.error(f"Failed to import PyTorch: {e}")
            raise ImportError("PyTorch is required for baseline IoU computation") from e
        
        # Validate model path exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Define the segmentation head architecture (matching training script)
        class SegmentationHeadConvNeXt(nn.Module):
            def __init__(self, in_channels, out_channels, tokenW, tokenH):
                super().__init__()
                self.H, self.W = tokenH, tokenW
                
                self.stem = nn.Sequential(
                    nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
                    nn.GELU()
                )
                
                self.block = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
                    nn.GELU(),
                    nn.Conv2d(128, 128, kernel_size=1),
                    nn.GELU(),
                )
                
                self.classifier = nn.Conv2d(128, out_channels, 1)
            
            def forward(self, x):
                B, N, C = x.shape
                x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
                x = self.stem(x)
                x = self.block(x)
                return self.classifier(x)
        
        # Define manual transform functions to avoid torchvision compatibility issues
        def resize_image(img, size):
            """Resize PIL image to target size."""
            return img.resize((size[1], size[0]), Image.BILINEAR)
        
        def resize_mask(mask, size):
            """Resize PIL mask to target size using nearest neighbor."""
            return mask.resize((size[1], size[0]), Image.NEAREST)
        
        def normalize_image(img_array, mean, std):
            """Normalize image array with mean and std."""
            img_array = img_array.astype(np.float32) / 255.0
            for i in range(3):
                img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
            return img_array
        
        # Define dataset class for inference
        class InferenceDataset(Dataset):
            def __init__(self, image_paths, mask_paths, target_size):
                self.image_paths = image_paths
                self.mask_paths = mask_paths
                self.target_size = target_size
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                # Load image
                image = Image.open(self.image_paths[idx]).convert('RGB')
                image = resize_image(image, self.target_size)
                image_array = np.array(image)
                image_array = normalize_image(image_array, self.mean, self.std)
                # Convert to CHW format
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
                
                # Load mask
                mask = Image.open(self.mask_paths[idx])
                mask_array = np.array(mask)
                
                # Handle multi-channel masks
                if len(mask_array.shape) == 3:
                    mask_array = mask_array[:, :, 0]
                
                # Convert mask values to class IDs using value_map
                value_map = {
                    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
                    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
                }
                converted_mask = np.zeros_like(mask_array, dtype=np.uint8)
                for raw_value, class_id in value_map.items():
                    converted_mask[mask_array == raw_value] = class_id
                
                # Resize mask
                mask_pil = Image.fromarray(converted_mask)
                mask_pil = resize_mask(mask_pil, self.target_size)
                mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
                
                return image_tensor, mask_tensor
        
        # Setup image dimensions (matching training script)
        img_width, img_height = 480, 270
        w = int((img_width // 14) * 14)  # 476
        h = int((img_height // 14) * 14)  # 266
        target_size = (h, w)
        
        # Get image-mask pairs
        pairs = self._get_image_mask_pairs()
        if not pairs:
            logger.warning("No image-mask pairs found for IoU computation")
            return {}
        
        image_paths = [str(img_path) for img_path, _ in pairs]
        mask_paths = [str(mask_path) for _, mask_path in pairs]
        
        # Create dataset and dataloader
        dataset = InferenceDataset(image_paths, mask_paths, target_size)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        logger.info(f"Created inference dataset with {len(dataset)} samples")
        
        # Load DINOv2 backbone
        logger.info("Loading DINOv2 backbone...")
        try:
            backbone = torch.hub.load(
                repo_or_dir="facebookresearch/dinov2",
                model="dinov2_vits14",
                skip_validation=True
            )
            backbone.eval()
            backbone.to(device)
            logger.info("Backbone loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DINOv2 backbone: {e}")
            raise RuntimeError(f"Failed to load DINOv2 backbone: {e}") from e
        
        # Get embedding dimension from backbone
        with torch.no_grad():
            sample_img = torch.randn(1, 3, h, w).to(device)
            sample_output = backbone.forward_features(sample_img)["x_norm_patchtokens"]
            n_embedding = sample_output.shape[2]
        
        logger.info(f"Embedding dimension: {n_embedding}")
        
        # Load segmentation head
        num_classes = 11
        segmentation_head = SegmentationHeadConvNeXt(
            in_channels=n_embedding,
            out_channels=num_classes,
            tokenW=w // 14,
            tokenH=h // 14
        )
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=device)
            segmentation_head.load_state_dict(checkpoint)
            segmentation_head.to(device)
            segmentation_head.eval()
            logger.info(f"Loaded segmentation head from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model checkpoint: {e}")
            raise RuntimeError(f"Failed to load model checkpoint: {e}") from e
        
        # Initialize per-class IoU tracking
        class_intersection = {i: 0 for i in range(num_classes)}
        class_union = {i: 0 for i in range(num_classes)}
        
        # Run inference and compute IoU
        logger.info("Running inference on training dataset...")
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(dataloader):
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass through backbone
                features = backbone.forward_features(images)["x_norm_patchtokens"]
                
                # Forward pass through segmentation head
                logits = segmentation_head(features)
                
                # Upsample to original image size
                outputs = F.interpolate(
                    logits,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                )
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=1)
                
                # Compute per-class intersection and union
                for class_id in range(num_classes):
                    pred_mask = (predictions == class_id)
                    target_mask = (masks == class_id)
                    
                    intersection = (pred_mask & target_mask).sum().item()
                    union = (pred_mask | target_mask).sum().item()
                    
                    class_intersection[class_id] += intersection
                    class_union[class_id] += union
                
                # Log progress every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Compute per-class IoU
        per_class_iou = {}
        for class_id in range(num_classes):
            if class_union[class_id] == 0:
                # Class not present in dataset
                per_class_iou[class_id] = 0.0
                logger.warning(f"Class {class_id} not present in dataset (union = 0)")
            else:
                iou = class_intersection[class_id] / class_union[class_id]
                per_class_iou[class_id] = float(iou)
        
        # Log results
        logger.info("Per-class IoU scores:")
        for class_id, iou in sorted(per_class_iou.items()):
            logger.info(f"  Class {class_id}: {iou:.4f}")
        
        # Compute mean IoU
        valid_ious = [iou for iou in per_class_iou.values() if iou > 0]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        logger.info(f"Mean IoU: {mean_iou:.4f}")
        
        # Identify poorly performing classes (IoU < 0.4)
        poorly_performing = [
            class_id for class_id, iou in per_class_iou.items()
            if iou < 0.4
        ]
        logger.info(f"Poorly performing classes (IoU < 0.4): {poorly_performing}")
        
        return per_class_iou
        
    def generate_report(self, output_path: str) -> AnalysisReport:
        """
        Generate comprehensive analysis report.
        
        Aggregates all analysis metrics including:
        - Class distribution and balance ratios
        - Quality issues detected
        - Per-class IoU scores (if model provided)
        - Poorly performing classes
        
        Saves:
        - JSON report with all metrics and timestamp
        - Class distribution bar plot
        - Balance ratio bar plot with threshold line
        - Per-class IoU bar plot (if IoU computed)
        
        Args:
            output_path: Path to save JSON report and plots
            
        Returns:
            AnalysisReport object with all metrics
        """
        import json
        from datetime import datetime
        
        logger.info("Generating comprehensive analysis report...")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Compute class distribution
        logger.info("Step 1/5: Computing class distribution...")
        class_distribution = self.compute_class_distribution()
        
        # 2. Compute balance ratios
        logger.info("Step 2/5: Computing class balance ratios...")
        class_balance_ratios = self.compute_class_balance_ratio(class_distribution)
        
        # 3. Identify imbalanced classes (ratio < 0.3)
        logger.info("Step 3/5: Identifying imbalanced classes...")
        imbalanced_classes = self.identify_imbalanced_classes(class_balance_ratios, threshold=0.3)
        
        # 4. Detect quality issues
        logger.info("Step 4/5: Detecting quality issues...")
        quality_issues = self.detect_quality_issues()
        
        # 5. Compute per-class IoU if model provided
        logger.info("Step 5/5: Computing per-class IoU scores...")
        per_class_iou = {}
        poorly_performing_classes = []
        
        if self.model_path:
            try:
                per_class_iou = self.compute_baseline_per_class_iou(self.model_path)
                # Identify classes with IoU < 0.4
                poorly_performing_classes = [
                    class_id for class_id, iou in per_class_iou.items()
                    if iou < 0.4
                ]
                logger.info(f"Found {len(poorly_performing_classes)} poorly performing classes (IoU < 0.4)")
            except NotImplementedError:
                logger.warning("Per-class IoU computation not yet implemented, skipping...")
            except Exception as e:
                logger.error(f"Failed to compute per-class IoU: {e}")
        else:
            logger.info("No model path provided, skipping IoU computation")
        
        # Calculate total statistics
        total_images = len(self._get_image_mask_pairs())
        total_pixels = sum(class_distribution.values())
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create AnalysisReport object
        report = AnalysisReport(
            class_distribution=class_distribution,
            class_balance_ratios=class_balance_ratios,
            imbalanced_classes=imbalanced_classes,
            quality_issues=quality_issues,
            per_class_iou=per_class_iou,
            poorly_performing_classes=poorly_performing_classes,
            total_images=total_images,
            total_pixels=total_pixels,
            timestamp=timestamp
        )
        
        # Save report as JSON
        logger.info("Saving analysis report as JSON...")
        report_json_path = output_dir / "analysis_report.json"
        
        # Convert report to dictionary for JSON serialization
        report_dict = {
            'class_distribution': class_distribution,
            'class_balance_ratios': class_balance_ratios,
            'imbalanced_classes': imbalanced_classes,
            'quality_issues': [
                {
                    'image_id': issue.image_id,
                    'issue_type': issue.issue_type,
                    'severity': issue.severity,
                    'details': issue.details
                }
                for issue in quality_issues
            ],
            'per_class_iou': per_class_iou,
            'poorly_performing_classes': poorly_performing_classes,
            'total_images': total_images,
            'total_pixels': total_pixels,
            'timestamp': timestamp
        }
        
        with open(report_json_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Saved JSON report to {report_json_path}")
        
        # Generate visualization plots
        logger.info("Generating visualization plots...")
        
        # Plot 1: Class distribution
        distribution_plot_path = output_dir / "class_distribution.png"
        self.generate_class_distribution_plot(class_distribution, distribution_plot_path)
        
        # Plot 2: Balance ratios
        balance_plot_path = output_dir / "class_balance_ratios.png"
        self.generate_balance_ratio_plot(class_balance_ratios, balance_plot_path, threshold=0.3)
        
        # Plot 3: Per-class IoU scores (if available)
        if per_class_iou:
            iou_plot_path = output_dir / "per_class_iou.png"
            self.generate_iou_plot(per_class_iou, iou_plot_path, threshold=0.4)
        
        # Log summary statistics
        logger.info("=" * 60)
        logger.info("ANALYSIS REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Images: {total_images}")
        logger.info(f"Total Pixels: {total_pixels:,}")
        logger.info(f"Unique Classes: {len(class_distribution)}")
        logger.info(f"Imbalanced Classes (ratio < 0.3): {len(imbalanced_classes)}")
        logger.info(f"Quality Issues Found: {len(quality_issues)}")
        if per_class_iou:
            logger.info(f"Poorly Performing Classes (IoU < 0.4): {len(poorly_performing_classes)}")
        logger.info(f"Report saved to: {output_dir}")
        logger.info("=" * 60)
        
        return report
