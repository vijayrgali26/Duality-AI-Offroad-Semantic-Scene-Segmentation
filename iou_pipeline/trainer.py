"""
Training Orchestrator Module

Manages model training with optimized hyperparameters, class weighting,
and advanced techniques.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from iou_pipeline.data.dataset import SegmentationDataset
from iou_pipeline.models.backbone import load_dinov2_backbone, compute_patch_grid_size, align_image_size_to_patches
from iou_pipeline.models.segmentation_head import create_segmentation_head
from iou_pipeline.data.transforms import (
    get_training_transforms,
    get_validation_transforms,
    get_mask_transforms
)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model
    backbone: str = 'dinov2_vits14'  # or 'dinov2_vitb14_reg', 'dinov2_vitl14_reg'
    num_classes: int = 11
    
    # Training
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 0.0001
    optimizer: str = 'sgd'  # or 'adamw'
    momentum: float = 0.9
    weight_decay: float = 0.0001
    
    # Scheduler
    scheduler: str = 'cosine'  # or 'step', 'plateau'
    t_max: int = 100  # for cosine
    eta_min: float = 1e-6
    
    # Loss
    use_class_weights: bool = True
    label_smoothing: float = 0.1
    
    # Advanced
    use_amp: bool = True  # mixed precision
    early_stopping_patience: int = 15
    gradient_clip: float = 1.0
    
    # Hard Example Mining
    use_hard_example_mining: bool = False
    hem_ratio: float = 0.3  # top 30% hardest samples
    
    # Multi-scale Training
    use_multiscale: bool = False
    scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    
    # Deep Supervision
    use_deep_supervision: bool = False
    aux_loss_weight: float = 0.4


@dataclass
class TrainingHistory:
    """Training history with all metrics."""
    train_losses: List[float]
    val_losses: List[float]
    val_ious: List[float]
    val_dices: List[float]
    val_accuracies: List[float]
    learning_rates: List[float]
    best_epoch: int
    best_val_iou: float


class TrainingOrchestrator:
    """
    Manages model training with optimized hyperparameters, class weighting,
    and advanced techniques.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize orchestrator with training configuration.
        
        Args:
            config: TrainingConfig object with all hyperparameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = None
        self.segmentation_head = None
        self.model_built = False
    
    def create_dataloaders(
        self,
        train_dataset_path: str,
        val_dataset_path: str,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoader instances for training and validation.
        
        This method creates data loaders using the SegmentationDataset class
        with appropriate transforms for training and validation. The training
        loader uses the augmented dataset with shuffling enabled, while the
        validation loader uses the original validation set without shuffling.
        
        Args:
            train_dataset_path: Path to augmented training dataset directory
            val_dataset_path: Path to validation dataset directory
            batch_size: Batch size for data loaders. If None, uses config.batch_size
            num_workers: Number of worker processes for data loading. Default: 4
            pin_memory: Whether to pin memory for faster GPU transfer. Default: True
        
        Returns:
            Tuple of (train_loader, val_loader) DataLoader instances
        
        Example:
            >>> orchestrator = TrainingOrchestrator(config)
            >>> train_loader, val_loader = orchestrator.create_dataloaders(
            ...     train_dataset_path='./data/augmented_train',
            ...     val_dataset_path='./data/val'
            ... )
            >>> for images, masks in train_loader:
            ...     # Training loop
            ...     pass
        """
        # Use batch size from config if not specified
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Get transform pipelines
        train_transform = get_training_transforms()
        val_transform = get_validation_transforms()
        mask_transform = get_mask_transforms()
        
        # Create training dataset with augmented data
        train_dataset = SegmentationDataset(
            data_dir=train_dataset_path,
            transform=train_transform,
            mask_transform=mask_transform,
            validate_on_init=False  # Skip validation for speed
        )
        
        # Create validation dataset with original validation set
        val_dataset = SegmentationDataset(
            data_dir=val_dataset_path,
            transform=val_transform,
            mask_transform=mask_transform,
            validate_on_init=False  # Skip validation for speed
        )
        
        # Create training data loader with shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop incomplete batches for consistent batch size
        )
        
        # Create validation data loader without shuffling
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False  # Keep all validation samples
        )
        
        return train_loader, val_loader
        
    def build_model(
        self,
        backbone: str = 'dinov2_vits14',
        num_classes: int = 11,
        image_height: int = 476,
        image_width: int = 266,
        freeze_backbone: bool = True,
        head_type: str = 'convnext',
        hidden_channels: int = 128
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Build segmentation model with specified backbone and head.
        
        This function creates a complete segmentation model by:
        1. Loading a DINOv2 backbone (frozen by default)
        2. Creating a ConvNeXt-style segmentation head
        3. Computing patch grid dimensions based on image size
        
        Args:
            backbone: Backbone architecture name (default: 'dinov2_vits14')
                     Options: 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitb14_reg',
                              'dinov2_vitl14', 'dinov2_vitl14_reg'
            num_classes: Number of segmentation classes (default: 11)
            image_height: Input image height in pixels (default: 476)
            image_width: Input image width in pixels (default: 266)
            freeze_backbone: If True, freeze backbone parameters (default: True)
            head_type: Type of segmentation head (default: 'convnext')
                      Options: 'convnext', 'convnext_deep', 'deep_supervision'
            hidden_channels: Number of intermediate channels in head (default: 128)
            
        Returns:
            Tuple of (backbone_model, segmentation_head)
            - backbone_model: DINOv2 backbone (frozen if freeze_backbone=True)
            - segmentation_head: Trainable segmentation head
            
        Example:
            >>> orchestrator = TrainingOrchestrator(config)
            >>> backbone, head = orchestrator.build_model()
            >>> print(f"Backbone frozen: {not any(p.requires_grad for p in backbone.parameters())}")
            Backbone frozen: True
        """
        # Load DINOv2 backbone
        backbone_model, embed_dim, patch_size = load_dinov2_backbone(
            backbone_name=backbone,
            freeze=freeze_backbone,
            device=self.device
        )
        
        # Compute patch grid dimensions
        num_patches_h, num_patches_w = compute_patch_grid_size(
            image_height, image_width, patch_size
        )
        
        print(f"\nBuilding segmentation head...")
        print(f"  Input embedding dim: {embed_dim}")
        print(f"  Output classes: {num_classes}")
        print(f"  Patch grid: {num_patches_h}x{num_patches_w}")
        print(f"  Head type: {head_type}")
        
        # Create segmentation head
        if self.config.use_deep_supervision and head_type != 'deep_supervision':
            print(f"  Note: Switching to 'deep_supervision' head (use_deep_supervision=True)")
            head_type = 'deep_supervision'
        
        segmentation_head = create_segmentation_head(
            head_type=head_type,
            in_channels=embed_dim,
            out_channels=num_classes,
            token_h=num_patches_h,
            token_w=num_patches_w,
            hidden_channels=hidden_channels
        )
        
        # Move head to device
        segmentation_head = segmentation_head.to(self.device)
        
        # Count parameters
        backbone_params = sum(p.numel() for p in backbone_model.parameters())
        backbone_trainable = sum(p.numel() for p in backbone_model.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in segmentation_head.parameters())
        head_trainable = sum(p.numel() for p in segmentation_head.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Backbone parameters: {backbone_params:,} (trainable: {backbone_trainable:,})")
        print(f"  Head parameters: {head_params:,} (trainable: {head_trainable:,})")
        print(f"  Total parameters: {backbone_params + head_params:,}")
        print(f"  Total trainable: {backbone_trainable + head_trainable:,}")
        
        # Store models
        self.backbone = backbone_model
        self.segmentation_head = segmentation_head
        self.model_built = True
        
        return backbone_model, segmentation_head
        
    def setup_loss_function(self, class_weights: Optional[torch.Tensor] = None,
                           label_smoothing: float = 0.0) -> nn.Module:
        """
        Create loss function with class weighting and label smoothing.
        
        Args:
            class_weights: Per-class weights based on inverse frequency
            label_smoothing: Label smoothing epsilon (0.0 to 0.2)
            
        Returns:
            Loss function module
        """
        raise NotImplementedError("To be implemented in task 4.2")
        
    def setup_optimizer(self, model: nn.Module, lr: float, 
                       optimizer_type: str = 'sgd') -> torch.optim.Optimizer:
        """
        Create optimizer with specified configuration.
        
        Args:
            model: Model to optimize
            lr: Learning rate
            optimizer_type: 'sgd' or 'adamw'
            
        Returns:
            Optimizer instance
        """
        raise NotImplementedError("To be implemented in task 4.3")
        
    def setup_scheduler(self, optimizer: torch.optim.Optimizer,
                       scheduler_type: str = 'cosine') -> Any:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: 'cosine', 'step', or 'plateau'
            
        Returns:
            Scheduler instance
        """
        raise NotImplementedError("To be implemented in task 4.4")
        
    def train_epoch(self, model: nn.Module, dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                   use_amp: bool = True) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            use_amp: Use automatic mixed precision
            
        Returns:
            Dictionary with epoch metrics
        """
        raise NotImplementedError("To be implemented in task 4.5")
        
    def validate_epoch(self, model: nn.Module, dataloader: DataLoader,
                      loss_fn: nn.Module) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            model: Model to validate
            dataloader: Validation data loader
            loss_fn: Loss function
            
        Returns:
            Dictionary with validation metrics (loss, IoU, Dice, accuracy)
        """
        raise NotImplementedError("To be implemented in task 4.6")
        
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float], path: str):
        """
        Save training checkpoint.
        
        Args:
            model: Model state
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics
            path: Save path
        """
        raise NotImplementedError("To be implemented in task 4.7")
        
    def check_early_stopping(self, val_metric: float, patience: int = 15) -> bool:
        """
        Check if early stopping criteria met.
        
        Args:
            val_metric: Current validation metric (IoU)
            patience: Number of epochs without improvement
            
        Returns:
            True if training should stop
        """
        raise NotImplementedError("To be implemented in task 4.8")
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int) -> TrainingHistory:
        """
        Execute complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            
        Returns:
            TrainingHistory with all metrics
        """
        raise NotImplementedError("To be implemented in task 4.9")
