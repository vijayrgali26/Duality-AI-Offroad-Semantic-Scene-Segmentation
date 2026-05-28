"""
Quick IoU computation script - samples a subset of data for faster results.
"""

import sys
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iou_pipeline.analyzer import DatasetAnalyzer
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Compute IoU scores on a sample of the dataset for quick results."""
    
    # Define paths
    possible_dataset_paths = [
        Path(r"C:\Users\vijay\OneDrive\Desktop\MERN Stack\Project1_Img_url\segmentation_hackathon\data\Offroad_Segmentation_Training_Dataset\train"),
        project_root / "train",
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if path.exists():
            dataset_path = path
            break
    
    if dataset_path is None:
        logger.error("Dataset not found!")
        return
    
    model_path = project_root / "segmentation_head.pth"
    
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return
    
    logger.info("=" * 80)
    logger.info("QUICK IoU ESTIMATION (Using 200 sample images)")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model: {model_path}")
    logger.info("=" * 80)
    
    try:
        # Create a temporary sampled dataset
        import shutil
        import tempfile
        
        logger.info("\nCreating sample dataset (200 images)...")
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        temp_images = temp_dir / "Color_Images"
        temp_masks = temp_dir / "Segmentation"
        temp_images.mkdir(parents=True)
        temp_masks.mkdir(parents=True)
        
        # Get all image files
        image_dir = dataset_path / "Color_Images"
        mask_dir = dataset_path / "Segmentation"
        
        all_images = list(image_dir.glob("*.*"))
        
        # Sample 200 random images
        sample_size = min(200, len(all_images))
        sampled_images = random.sample(all_images, sample_size)
        
        logger.info(f"Sampling {sample_size} images from {len(all_images)} total images...")
        
        # Copy sampled images and masks
        for img_path in sampled_images:
            # Copy image
            shutil.copy(img_path, temp_images / img_path.name)
            
            # Copy corresponding mask
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                shutil.copy(mask_path, temp_masks / img_path.name)
        
        logger.info(f"Sample dataset created at: {temp_dir}")
        
        # Create analyzer with sampled dataset
        logger.info("\nInitializing DatasetAnalyzer...")
        analyzer = DatasetAnalyzer(
            dataset_path=str(temp_dir),
            model_path=str(model_path)
        )
        
        # Compute per-class IoU
        logger.info("\nComputing per-class IoU scores on sample...")
        logger.info("This should take 2-3 minutes...")
        
        per_class_iou = analyzer.compute_baseline_per_class_iou(
            model_path=str(model_path)
        )
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("ESTIMATED IoU SCORES BY CLASS (Based on 200 sample images)")
        logger.info("=" * 80)
        
        # Class names
        class_names = {
            0: "Background",
            1: "Trees",
            2: "Lush Bushes",
            3: "Dry Grass",
            4: "Dry Bushes",
            5: "Ground Clutter",
            6: "Flowers",
            7: "Logs",
            8: "Rocks",
            9: "Landscape",
            10: "Sky"
        }
        
        # Sort by class ID
        sorted_classes = sorted(per_class_iou.items())
        
        total_iou = 0
        poorly_performing = []
        
        for class_id, iou_score in sorted_classes:
            class_name = class_names.get(class_id, f"Class {class_id}")
            status = "⚠️ POOR" if iou_score < 0.4 else "✓ GOOD" if iou_score >= 0.6 else "○ OK"
            
            logger.info(f"Class {class_id:2d} ({class_name:15s}): {iou_score:.4f} {status}")
            
            total_iou += iou_score
            if iou_score < 0.4:
                poorly_performing.append((class_id, class_name, iou_score))
        
        # Compute mean IoU
        mean_iou = total_iou / len(per_class_iou) if per_class_iou else 0.0
        
        logger.info("=" * 80)
        logger.info(f"ESTIMATED MEAN IoU: {mean_iou:.4f}")
        logger.info("=" * 80)
        
        # Display poorly performing classes
        if poorly_performing:
            logger.info("\n⚠️  POORLY PERFORMING CLASSES (IoU < 0.4):")
            logger.info("-" * 80)
            for class_id, class_name, iou_score in poorly_performing:
                logger.info(f"  • Class {class_id} ({class_name}): {iou_score:.4f}")
            logger.info(f"\nTotal: {len(poorly_performing)} classes need improvement")
        else:
            logger.info("\n✓ All classes performing well (IoU >= 0.4)")
        
        logger.info("\n" + "=" * 80)
        logger.info("NOTE: This is an estimate based on 200 sample images.")
        logger.info("For exact results, run the full evaluation on all 2857 images.")
        logger.info("=" * 80)
        
        return per_class_iou
        
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
