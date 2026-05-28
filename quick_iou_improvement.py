"""
Quick IoU Improvement Script
Implements key improvements to boost IoU from 0.60 to 0.75+

Key improvements:
1. Class-weighted loss (addresses class imbalance)
2. Increased training epochs (50 instead of 10)
3. Better learning rate with cosine annealing
4. AdamW optimizer instead of SGD
5. Label smoothing
6. Early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    data_dir = r"C:\Users\vijay\OneDrive\Desktop\MERN Stack\Project1_Img_url\segmentation_hackathon\data\Offroad_Segmentation_Training_Dataset"
    output_dir = "improved_training_output"
    baseline_model = "segmentation_head.pth"
    
    # Training
    batch_size = 8  # Increased from 4
    num_epochs = 50  # Increased from 10
    learning_rate = 0.0002  # Optimized
    weight_decay = 0.0001
    
    # Model
    backbone = "dinov2_vits14"
    num_classes = 11
    
    # Advanced
    use_class_weights = True
    label_smoothing = 0.1
    early_stopping_patience = 15
    
    # Image size
    img_width = int(((960 / 2) // 14) * 14)  # 476
    img_height = int(((540 / 2) // 14) * 14)  # 266


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        data_root = Path(data_dir)
        split_dir = data_root / split if (data_root / split).is_dir() else data_root
        
        self.image_dir = split_dir / 'Color_Images'
        self.masks_dir = split_dir / 'Segmentation'
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        self.data_ids = sorted(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = self.image_dir / data_id
        mask_path = self.masks_dir / data_id
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((Config.img_width, Config.img_height))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Load mask
        if mask_path.exists():
            mask = Image.open(mask_path)
            mask = convert_mask(mask)
            mask = mask.resize((Config.img_width, Config.img_height), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            mask = torch.zeros((Config.img_height, Config.img_width), dtype=torch.long)
        
        return image, mask


# ============================================================================
# Model
# ============================================================================

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


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=11):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    
    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().item())
    
    return np.nanmean(iou_per_class), iou_per_class


def compute_class_weights(train_loader, num_classes=11):
    """Compute class weights based on inverse frequency."""
    print("Computing class weights from training data...")
    class_counts = torch.zeros(num_classes)
    
    for _, masks in tqdm(train_loader, desc="Analyzing class distribution"):
        for class_id in range(num_classes):
            class_counts[class_id] += (masks == class_id).sum().item()
    
    # Inverse frequency weighting
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Normalize to have mean = 1.0
    class_weights = class_weights / class_weights.mean()
    
    # Handle zero counts
    class_weights[class_counts == 0] = 0.0
    
    print("\nClass weights:")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"  {name:<20}: {weight:.4f}")
    
    return class_weights


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, backbone, train_loader, optimizer, criterion, device, scaler):
    """Train for one epoch."""
    model.train()
    backbone.eval()  # Keep backbone frozen
    
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Extract features
            with torch.no_grad():
                features = backbone.forward_features(images)["x_norm_patchtokens"]
            
            # Segmentation head
            logits = model(features)
            outputs = F.interpolate(logits, size=images.shape[2:], mode="bilinear", align_corners=False)
            
            # Compute loss
            loss = criterion(outputs, masks)
        
        # Backward pass
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        iou, _ = compute_iou(outputs.detach(), masks)
        total_iou += iou
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    
    return total_loss / len(train_loader), total_iou / len(train_loader)


def validate_epoch(model, backbone, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    backbone.eval()
    
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Extract features
            features = backbone.forward_features(images)["x_norm_patchtokens"]
            
            # Segmentation head
            logits = model(features)
            outputs = F.interpolate(logits, size=images.shape[2:], mode="bilinear", align_corners=False)
            
            # Compute loss
            loss = criterion(outputs, masks)
            
            # Metrics
            total_loss += loss.item()
            iou, _ = compute_iou(outputs, masks)
            total_iou += iou
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    
    return total_loss / len(val_loader), total_iou / len(val_loader)


def main():
    print("="*60)
    print("QUICK IoU IMPROVEMENT TRAINING")
    print("="*60)
    print(f"\nBaseline IoU: 0.6055")
    print(f"Target IoU: 0.75+")
    print("\nKey improvements:")
    print("  ✓ Class-weighted loss")
    print("  ✓ Increased epochs (10 → 50)")
    print("  ✓ Better learning rate (0.0001 → 0.0002)")
    print("  ✓ AdamW optimizer")
    print("  ✓ Label smoothing (0.1)")
    print("  ✓ Cosine annealing scheduler")
    print("  ✓ Early stopping (patience=15)")
    print("\n" + "="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MaskDataset(Config.data_dir, split='train')
    val_dataset = MaskDataset(Config.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Load backbone
    print("\nLoading DINOv2 backbone...")
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=Config.backbone, skip_validation=True)
    backbone.eval()
    backbone.to(device)
    
    # Get embedding dimension
    sample_img, _ = train_dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    
    # Create model
    print("\nCreating segmentation head...")
    model = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=Config.num_classes,
        tokenW=Config.img_width // 14,
        tokenH=Config.img_height // 14
    )
    model.to(device)
    
    # Compute class weights
    if Config.use_class_weights:
        class_weights = compute_class_weights(train_loader, Config.num_classes)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)
    
    # Optimizer (AdamW instead of SGD)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    
    # Scheduler (Cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_iou = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_iou': [],
        'val_loss': [], 'val_iou': [],
        'learning_rates': []
    }
    
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_iou = train_epoch(model, backbone, train_loader, optimizer, criterion, device, scaler)
        
        # Validate
        val_loss, val_iou = validate_epoch(model, backbone, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['learning_rates'].append(current_lr)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(Config.output_dir, 'best_model.pth'))
            print(f"✓ New best model saved! IoU: {val_iou:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{Config.early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= Config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(Config.output_dir, 'final_model.pth'))
    
    # Save history
    with open(os.path.join(Config.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs_range, history['train_iou'], label='Train IoU')
    ax2.plot(epochs_range, history['val_iou'], label='Val IoU')
    ax2.axhline(y=0.6055, color='r', linestyle='--', label='Baseline (0.6055)')
    ax2.axhline(y=0.75, color='g', linestyle='--', label='Target (0.75)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Training and Validation IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBaseline IoU: 0.6055")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Improvement: {(best_val_iou - 0.6055):.4f} ({(best_val_iou - 0.6055)*100:.2f}%)")
    
    if best_val_iou >= 0.75:
        print("\n✓ TARGET ACHIEVED! IoU >= 0.75")
    else:
        print(f"\n⚠ Target not reached. Gap: {(0.75 - best_val_iou):.4f}")
    
    print(f"\nOutputs saved to: {Config.output_dir}/")
    print("  - best_model.pth")
    print("  - final_model.pth")
    print("  - training_history.json")
    print("  - training_curves.png")
    print("="*60)


if __name__ == "__main__":
    main()
