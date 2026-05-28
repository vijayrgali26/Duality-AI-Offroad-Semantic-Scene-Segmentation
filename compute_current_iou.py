"""
Script to compute current IoU scores for the segmentation model.
"""

import sys
from pathlib import Path

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
    """Compute current IoU scores."""
    
    # Define paths
    # Try to find the dataset in common locations
    possible_dataset_paths = [
        Path(r"C:\Users\vijay\OneDrive\Desktop\MERN Stack\Project1_Img_url\segmentation_hackathon\data\Offroad_Segmentation_Training_Dataset\train"),
        project_root / "train",
        project_root / "Offroad_Segmentation_Training_Dataset" / "train",
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if path.exists():
            dataset_path = path
            break
    
    if dataset_path is None:
        logger.error("Dataset not found in any of the expected locations:")
        for path in possible_dataset_paths:
            logger.error(f"  - {path}")
        logger.info("\nPlease provide the correct path to your training dataset.")
        logger.info("The dataset should contain 'Color_Images' and 'Segmentation' subdirectories.")
        return
    
    model_path = project_root / "segmentation_head.pth"  # Baseline model
    
    # Check if paths exist
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        logger.info("Please provide the correct path to your model checkpoint.")
        return
    
    logger.info("=" * 80)
    logger.info("COMPUTING CURRENT IoU SCORES")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model: {model_path}")
    logger.info("=" * 80)
    
    try:
        # Create analyzer
        logger.info("\nInitializing DatasetAnalyzer...")
        analyzer = DatasetAnalyzer(
            dataset_path=str(dataset_path),
            model_path=str(model_path)
        )
        
        # Compute per-class IoU
        logger.info("\nComputing per-class IoU scores...")
        logger.info("This may take several minutes depending on dataset size...")
        
        per_class_iou = analyzer.compute_baseline_per_class_iou(
            model_path=str(model_path)
        )
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("IoU SCORES BY CLASS")
        logger.info("=" * 80)
        
        # Class names for better readability
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
        logger.info(f"MEAN IoU: {mean_iou:.4f}")
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
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        
        return per_class_iou
        
    except FileNotFoundError as e:
        logger.error(f"\nError: {e}")
        logger.info("\nPlease check that the dataset and model paths are correct.")
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
