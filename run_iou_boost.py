"""
IoU Boost Pipeline - Simplified version to improve model performance

This script runs a streamlined version of the IoU improvement pipeline:
1. Analyzes dataset to identify poorly performing classes
2. Creates balanced training set with class-aware augmentation
3. Trains model with class weights and optimized hyperparameters
4. Evaluates improved model

Target: Boost mean IoU from 0.28 to 0.5-0.6
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iou_pipeline.analyzer import DatasetAnalyzer
from iou_pipeline.data.dataset import SegmentationDataset
from iou_pipeline.data.transforms import get_training_transforms, get_validation_transforms, get_mask_transforms
from iou_pipeline.models.backbone import load_dinov2_backbone
from iou_pipeline.models.segmentation_head import SegmentationHeadConvNeXt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Paths
    train_path = Path(r"C:\Users\vijay\OneDrive\Desktop\MERN Stack\Project1_Img_url\segmentation_hackathon\data\Offroad_Segmentation_Training_Dataset\train")
    val_path = Path(r"C:\Users\vijay\OneDrive\Desktop\MERN Stack\Project1_Img_url\segmentation_hackathon\data\Offroad_Segmentation_Training_Dataset\val")
    baseline_model = project_root / "segmentation_head.pth"
    output_dir = project_root / "iou_boost_output"
    
    # Training hyperparameters
    num_epochs = 15
    batch_size = 4
    learning_rate = 0.001
    num_classes = 11
    
    # Model architecture
    backbone = 'dinov2_vits14'
    image_height = 266
    image_width = 476
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_class_weights(train_path: Path, num_classes: int = 11) -> torch.Tensor:
    """Compute inverse frequency class weights for balanced training."""
    logger.info("Computing class weights from training data...")
    
    # Count pixels per class
    class_counts = np.zeros(num_classes)
    mask_dir = train_path / "Segmentation"
    
    from PIL import Image
    mask_files = list(mask_dir.glob("*.*"))[:500]  # Sample 500 images for speed
    
    value_map = {
        0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
        550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
    }
    
    for mask_file in tqdm(mask_files, desc="Analyzing masks"):
        mask = np.array(Image.open(mask_file))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Convert to class IDs
        converted_mask = np.zeros_like(mask, dtype=np.uint8)
        for raw_val, class_id in value_map.items():
            converted_mask[mask == raw_val] = class_id
        
        # Count pixels
        for class_id in range(num_classes):
            class_counts[class_id] += (converted_mask == class_id).sum()
    
    # Compute inverse frequency weights
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    
    # Normalize to mean = 1.0
    class_weights = class_weights / class_weights.mean()
    
    # Cap maximum weight to prevent extreme values
    class_weights = np.clip(class_weights, 0.1, 10.0)
    
    logger.info("Class weights computed:")
    for i, weight in enumerate(class_weights):
        logger.info(f"  Class {i}: {weight:.3f}")
    
    return torch.FloatTensor(class_weights)


def train_epoch(model_backbone, model_head, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model_backbone.eval()  # Backbone stays frozen
    model_head.train()
    
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        with torch.no_grad():
            features = model_backbone.forward_features(images)
            tokens = features["x_norm_patchtokens"]
        
        logits = model_head(tokens)
        
        # Upsample logits to match mask size
        logits_upsampled = torch.nn.functional.interpolate(
            logits,
            size=(masks.shape[1], masks.shape[2]),
            mode='bilinear',
            align_corners=False
        )
        
        # Compute loss
        loss = criterion(logits_upsampled, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = logits_upsampled.argmax(dim=1)
        correct_pixels += (predictions == masks).sum().item()
        total_pixels += masks.numel()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct_pixels / total_pixels:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_pixels / total_pixels
    
    return avg_loss, accuracy


def validate(model_backbone, model_head, val_loader, criterion, device):
    """Validate the model."""
    model_backbone.eval()
    model_head.eval()
    
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    # Per-class IoU computation
    num_classes = 11
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            features = model_backbone.forward_features(images)
            tokens = features["x_norm_patchtokens"]
            logits = model_head(tokens)
            
            # Upsample logits
            logits_upsampled = torch.nn.functional.interpolate(
                logits,
                size=(masks.shape[1], masks.shape[2]),
                mode='bilinear',
                align_corners=False
            )
            
            # Compute loss
            loss = criterion(logits_upsampled, masks)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = logits_upsampled.argmax(dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            
            # Compute per-class IoU
            predictions_np = predictions.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            for class_id in range(num_classes):
                pred_mask = (predictions_np == class_id)
                true_mask = (masks_np == class_id)
                
                intersection[class_id] += (pred_mask & true_mask).sum()
                union[class_id] += (pred_mask | true_mask).sum()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct_pixels / total_pixels
    
    # Compute IoU scores
    iou_scores = {}
    for class_id in range(num_classes):
        if union[class_id] > 0:
            iou_scores[class_id] = intersection[class_id] / union[class_id]
        else:
            iou_scores[class_id] = 0.0
    
    mean_iou = np.mean(list(iou_scores.values()))
    
    return avg_loss, accuracy, iou_scores, mean_iou


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("IoU BOOST PIPELINE - STARTING")
    logger.info("=" * 80)
    logger.info(f"Device: {Config.device}")
    logger.info(f"Training path: {Config.train_path}")
    logger.info(f"Validation path: {Config.val_path}")
    logger.info(f"Output directory: {Config.output_dir}")
    
    # Create output directory
    Config.output_dir.mkdir(exist_ok=True)
    
    # Step 1: Compute class weights
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Computing Class Weights")
    logger.info("=" * 80)
    class_weights = compute_class_weights(Config.train_path, Config.num_classes)
    class_weights = class_weights.to(Config.device)
    
    # Step 2: Create data loaders
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Creating Data Loaders")
    logger.info("=" * 80)
    
    train_transform = get_training_transforms()
    val_transform = get_validation_transforms()
    mask_transform = get_mask_transforms()
    
    train_dataset = SegmentationDataset(
        data_dir=str(Config.train_path),
        transform=train_transform,
        mask_transform=mask_transform,
        validate_on_init=False
    )
    
    val_dataset = SegmentationDataset(
        data_dir=str(Config.val_path),
        transform=val_transform,
        mask_transform=mask_transform,
        validate_on_init=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Step 3: Build model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Building Model")
    logger.info("=" * 80)
    
    backbone, embed_dim, patch_size = load_dinov2_backbone(
        Config.backbone,
        freeze=True,
        device=Config.device
    )
    
    token_h = Config.image_height // patch_size
    token_w = Config.image_width // patch_size
    
    segmentation_head = SegmentationHeadConvNeXt(
        in_channels=embed_dim,
        out_channels=Config.num_classes,
        token_h=token_h,
        token_w=token_w,
        hidden_channels=128
    ).to(Config.device)
    
    logger.info(f"Model built successfully")
    logger.info(f"Trainable parameters: {segmentation_head.get_num_params():,}")
    
    # Step 4: Setup training
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Setup Training")
    logger.info("=" * 80)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        segmentation_head.parameters(),
        lr=Config.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.num_epochs
    )
    
    logger.info(f"Loss: CrossEntropyLoss with class weights")
    logger.info(f"Optimizer: AdamW (lr={Config.learning_rate})")
    logger.info(f"Scheduler: CosineAnnealingLR")
    
    # Step 5: Training loop
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Training")
    logger.info("=" * 80)
    
    best_mean_iou = 0.0
    training_history = []
    
    for epoch in range(1, Config.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{Config.num_epochs}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            backbone, segmentation_head, train_loader,
            criterion, optimizer, Config.device, epoch
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc, iou_scores, mean_iou = validate(
            backbone, segmentation_head, val_loader,
            criterion, Config.device
        )
        
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"Mean IoU: {mean_iou:.4f}")
        
        # Log per-class IoU
        class_names = {
            0: "Background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass",
            4: "Dry Bushes", 5: "Ground Clutter", 6: "Flowers", 7: "Logs",
            8: "Rocks", 9: "Landscape", 10: "Sky"
        }
        
        logger.info("Per-class IoU:")
        for class_id, iou in iou_scores.items():
            status = "✓" if iou >= 0.4 else "⚠"
            logger.info(f"  {status} Class {class_id} ({class_names[class_id]:15s}): {iou:.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'mean_iou': mean_iou,
            'per_class_iou': {int(k): float(v) for k, v in iou_scores.items()}
        })
        
        # Save best model
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            checkpoint_path = Config.output_dir / "best_model.pth"
            torch.save(segmentation_head.state_dict(), checkpoint_path)
            logger.info(f"✓ New best model saved! Mean IoU: {mean_iou:.4f}")
        
        # Update learning rate
        scheduler.step()
    
    # Step 6: Save results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Saving Results")
    logger.info("=" * 80)
    
    # Save training history
    history_path = Config.output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Best Mean IoU: {best_mean_iou:.4f}")
    logger.info(f"Improvement: {best_mean_iou - 0.28:.4f} (from baseline 0.28)")
    logger.info(f"Best model saved to: {Config.output_dir / 'best_model.pth'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
