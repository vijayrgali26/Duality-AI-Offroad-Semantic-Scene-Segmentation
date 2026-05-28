"""
IoU Improvement Pipeline Runner
Simplified script to run the complete pipeline without torchvision dependency issues
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("="*60)
    print("IoU IMPROVEMENT PIPELINE")
    print("="*60)
    print(f"\nCurrent Baseline IoU: 0.6055 (60.55%)")
    print(f"Target IoU: 0.75+ (75%+)")
    print("\nThis pipeline will:")
    print("  1. Analyze dataset quality and class distribution")
    print("  2. Balance and augment the dataset")
    print("  3. Train with optimized hyperparameters")
    print("  4. Evaluate and compare results")
    print("\n" + "="*60)
    
    # Check if required directories exist
    dataset_dir = project_root / "Offroad_Segmentation_Training_Dataset"
    if not dataset_dir.exists():
        print(f"\nERROR: Dataset directory not found: {dataset_dir}")
        print("Please ensure the dataset is in the correct location.")
        return 1
    
    # Create configuration
    config = {
        "pipeline": {
            "experiment_name": "iou_improvement_run_1",
            "output_dir": str(project_root / "experiments"),
            "dry_run": False,
            "resume_from_checkpoint": None
        },
        "dataset": {
            "train_path": str(dataset_dir / "train"),
            "val_path": str(dataset_dir / "val"),
            "test_path": str(project_root / "Offroad_Segmentation_testImages")
        },
        "analysis": {
            "run_analysis": True,
            "baseline_model_path": str(project_root / "segmentation_head.pth"),
            "save_plots": True
        },
        "augmentation": {
            "run_augmentation": True,
            "min_balance_ratio": 0.5,
            "poor_iou_threshold": 0.4,
            "target_dataset_size": 5000,
            "techniques": {
                "horizontal_flip": True,
                "rotation": True,
                "brightness": True,
                "contrast": True
            }
        },
        "training": {
            "backbone": "dinov2_vits14",
            "num_classes": 11,
            "batch_size": 8,
            "num_epochs": 50,
            "learning_rate": 0.0002,
            "optimizer": "adamw",
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "t_max": 50,
            "eta_min": 0.000001,
            "use_class_weights": True,
            "label_smoothing": 0.1,
            "use_amp": True,
            "early_stopping_patience": 15,
            "gradient_clip": 1.0,
            "use_hard_example_mining": False,
            "hem_ratio": 0.3,
            "use_multiscale": False,
            "scales": [0.75, 1.0, 1.25],
            "use_deep_supervision": False,
            "aux_loss_weight": 0.4
        },
        "evaluation": {
            "use_tta": False,
            "num_visualization_samples": 20,
            "compare_with_baseline": True
        }
    }
    
    # Save configuration
    config_path = project_root / "pipeline_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, indent=2, fp=f)
    print(f"\nConfiguration saved to: {config_path}")
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION")
    print("="*60)
    
    try:
        # Import pipeline components
        from iou_pipeline.pipeline import IoUPipeline
        from iou_pipeline.utils.config import PipelineConfig
        
        # Load configuration
        pipeline_config = PipelineConfig.from_dict(config)
        
        # Create and run pipeline
        pipeline = IoUPipeline(pipeline_config)
        results = pipeline.run()
        
        # Print results
        print("\n" + "="*60)
        print("PIPELINE RESULTS")
        print("="*60)
        print(f"Status: {results['status']}")
        
        if results['status'] == 'completed':
            print(f"\nExperiment ID: {results['experiment_id']}")
            print(f"Baseline IoU: 0.6055")
            print(f"Final Val IoU: {results['final_metrics']['val_iou']:.4f}")
            print(f"Improvement: {results['improvement']:.4f} ({results['improvement']*100:.2f}%)")
            print(f"\nOutput directory: {results['output_dir']}")
            
            if results['final_metrics']['val_iou'] >= 0.75:
                print("\n✓ TARGET ACHIEVED! IoU >= 0.75")
            else:
                print(f"\n⚠ Target not reached. Current: {results['final_metrics']['val_iou']:.4f}, Target: 0.75")
        else:
            print(f"\nPipeline failed at phase: {results.get('phase', 'unknown')}")
            print(f"Error: {results.get('error', 'unknown')}")
        
        print("="*60)
        
        return 0 if results['status'] == 'completed' else 1
        
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
