# Implementation Plan: IoU Improvement Pipeline

## Overview

This implementation plan breaks down the IoU Improvement Pipeline into discrete coding tasks. The pipeline enhances semantic segmentation performance through systematic dataset analysis, quality improvement, balanced augmentation, optimized training, and comprehensive evaluation. The implementation uses Python with PyTorch and follows the existing project structure with DINOv2 backbone and ConvNeXt-style segmentation head.

## Tasks

- [x] 1. Setup project infrastructure and core utilities
  - [x] 1.1 Create package structure and base modules
    - Create `iou_pipeline/` package directory with `__init__.py`
    - Create subdirectories: `models/`, `data/`, `utils/`
    - Create module files: `analyzer.py`, `editor.py`, `trainer.py`, `evaluator.py`, `tracker.py`, `pipeline.py`
    - Create `__init__.py` files for all subdirectories
    - _Requirements: 10.1, 10.2_
  
  - [x] 1.2 Implement configuration management system
    - Create `utils/config.py` with `PipelineConfig` and `TrainingConfig` dataclasses
    - Implement `ConfigManager` class with YAML/JSON loading methods
    - Add configuration validation with range checks for hyperparameters
    - Implement configuration merging for inheritance
    - _Requirements: 10.3, 3.1, 3.2, 3.6_
  
  - [x] 1.3 Setup logging and error handling infrastructure
    - Create `utils/logging.py` with `setup_logging()` function
    - Implement custom exception hierarchy in `utils/exceptions.py`
    - Add file and console handlers with appropriate formatters
    - Create error recovery utilities for common failures
    - _Requirements: 10.2, 10.5_
  
  - [x] 1.4 Implement experiment tracking system
    - Create `tracker.py` with `ExperimentTracker` class
    - Implement experiment creation with unique timestamp-based IDs
    - Add methods for logging configs, metrics, and artifacts
    - Implement leaderboard management with JSON persistence
    - Create experiment directory structure automatically
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  
  - [ ]* 1.5 Write unit tests for configuration and tracking
    - Test configuration loading from YAML and JSON
    - Test configuration validation with invalid values
    - Test experiment ID generation and uniqueness
    - Test leaderboard updates and ranking
    - _Requirements: 6.1, 6.2, 6.5_

- [x] 2. Checkpoint - Verify infrastructure setup
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 3. Implement dataset analysis module
  - [x] 3.1 Create core dataset analyzer class
    - Implement `DatasetAnalyzer` class in `analyzer.py`
    - Add `__init__()` method with dataset path and optional model path
    - Create `QualityIssue` and `AnalysisReport` dataclasses
    - Implement dataset loading utilities for images and masks
    - _Requirements: 1.1, 1.5_
  
  - [x] 3.2 Implement class distribution analysis
    - Create `compute_class_distribution()` method using NumPy vectorization
    - Implement `compute_class_balance_ratio()` with normalization to max class
    - Add method to identify imbalanced classes (ratio < 0.3)
    - Generate matplotlib plots for class distribution visualization
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [x] 3.3 Implement quality issue detection
    - Create `detect_quality_issues()` method
    - Add validation for mask values against valid class indices [0-10]
    - Implement boundary error detection using morphological operations
    - Add noise detection using connected component analysis
    - Detect missing labels and corrupted files
    - _Requirements: 1.4, 8.4, 8.5_
  
  - [x] 3.4 Implement baseline IoU analysis
    - Create `compute_baseline_per_class_iou()` method
    - Load baseline model checkpoint and run inference on training set
    - Compute per-class IoU scores using predictions vs ground truth
    - Identify poorly performing classes (IoU < 0.4)
    - _Requirements: 1.6, 5.2_
  
  - [x] 3.5 Implement report generation
    - Create `generate_report()` method
    - Aggregate all analysis metrics into `AnalysisReport` object
    - Save report as JSON with timestamp
    - Generate and save visualization plots (distribution, IoU scores)
    - _Requirements: 1.5_
  
  - [ ]* 3.6 Write unit tests for dataset analyzer
    - Test class distribution computation with synthetic data
    - Test balance ratio calculation and normalization
    - Test quality issue detection for invalid values and boundaries
    - Test report generation and JSON serialization
    - _Requirements: 1.1, 1.2, 1.4_

- [ ] 4. Implement dataset editing and augmentation module
  - [~] 4.1 Create dataset editor class with augmentation pipeline
    - Implement `DatasetEditor` class in `editor.py`
    - Add `__init__()` method accepting dataset path and analysis report
    - Define `AUGMENTATION_TECHNIQUES` configuration dictionary
    - Setup albumentations pipeline for synchronized image-mask transforms
    - _Requirements: 2.1, 2.2_
  
  - [~] 4.2 Implement augmentation target identification
    - Create `identify_augmentation_targets()` method
    - Calculate augmentation factors based on balance ratios and IoU scores
    - Apply factor 2.0 for classes with IoU < 0.4
    - Apply factor 1.5 for classes with balance ratio < 0.5
    - _Requirements: 2.1, 2.3_
  
  - [~] 4.3 Implement augmentation functions
    - Create `apply_augmentation()` method for single transforms
    - Implement horizontal flip with 50% probability
    - Implement rotation within ±15 degrees with 30% probability
    - Implement brightness adjustment (0.8-1.2) with 40% probability
    - Implement contrast adjustment (0.8-1.2) with 40% probability
    - Ensure pixel-perfect mask-image alignment
    - _Requirements: 2.2, 2.5_
  
  - [~] 4.4 Implement class-aware oversampling
    - Create `oversample_class()` method
    - Identify images containing target underrepresented class
    - Generate augmented samples with specified multiplication factor
    - Track new sample IDs and maintain metadata
    - _Requirements: 2.1, 2.3_
  
  - [~] 4.5 Implement quality fixing utilities
    - Create `fix_boundary_errors()` method
    - Apply morphological opening and closing operations
    - Implement mask value correction for invalid pixels
    - Add image exclusion functionality for severe issues
    - Generate quality report listing all corrections
    - _Requirements: 8.1, 8.2, 8.3, 8.5, 8.6_
  
  - [~] 4.6 Implement dataset validation and saving
    - Create `validate_augmented_dataset()` method
    - Verify minimum dataset size (5000 samples)
    - Check mask-image alignment using dimension comparison
    - Validate all mask values are in valid range
    - Implement `save_augmented_dataset()` to write balanced dataset
    - Preserve original validation set without modifications
    - _Requirements: 2.4, 2.5, 2.6_
  
  - [ ]* 4.7 Write unit tests for dataset editor
    - Test augmentation preserves mask-image alignment
    - Test oversampling increases class representation correctly
    - Test boundary smoothing with morphological operations
    - Test dataset validation checks size and alignment
    - _Requirements: 2.2, 2.5, 2.6_

- [~] 5. Checkpoint - Verify dataset processing modules
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement model building and loss functions
  - [~] 6.1 Create model building utilities
    - Create `models/backbone.py` for DINOv2 loading
    - Implement function to load DINOv2 variants (vits14, vitb14, vitl14)
    - Create `models/segmentation_head.py` with ConvNeXt-style decoder
    - Implement `build_model()` in `trainer.py` combining backbone and head
    - Support frozen backbone with trainable segmentation head
    - _Requirements: 4.1, 4.2, 7.3_
  
  - [~] 6.2 Implement class-weighted loss function
    - Create `compute_class_weights()` utility in `utils/metrics.py`
    - Implement inverse frequency weighting formula
    - Normalize weights to have mean = 1.0
    - Create `setup_loss_function()` in `trainer.py`
    - Support CrossEntropyLoss with class weights and label smoothing
    - _Requirements: 3.4, 7.5_
  
  - [~] 6.3 Implement deep supervision support
    - Create `SegmentationHeadWithDeepSupervision` class
    - Add auxiliary classifier from intermediate features
    - Implement forward pass with optional auxiliary output
    - Add auxiliary loss computation with configurable weight
    - _Requirements: 7.6_
  
  - [ ]* 6.4 Write unit tests for model components
    - Test model building with different backbones
    - Test class weight computation with known distributions
    - Test loss function applies weights correctly
    - Test deep supervision auxiliary outputs
    - _Requirements: 3.4, 4.1, 7.6_

- [ ] 7. Implement training orchestrator core functionality
  - [~] 7.1 Create training orchestrator class
    - Implement `TrainingOrchestrator` class in `trainer.py`
    - Add `__init__()` accepting `TrainingConfig`
    - Create `TrainingHistory` dataclass for metrics tracking
    - Setup device management (CPU/GPU detection)
    - _Requirements: 4.1, 4.6_
  
  - [~] 7.2 Implement optimizer and scheduler setup
    - Create `setup_optimizer()` method supporting SGD and AdamW
    - Implement `setup_scheduler()` with cosine annealing support
    - Add support for step and plateau schedulers
    - Configure warm restarts for cosine annealing
    - _Requirements: 3.1, 3.3_
  
  - [~] 7.3 Implement training epoch loop
    - Create `train_epoch()` method with batch iteration
    - Implement forward pass, loss computation, and backpropagation
    - Add gradient clipping with configurable threshold
    - Support automatic mixed precision (AMP) with gradient scaling
    - Compute and return epoch metrics (loss, IoU, Dice, accuracy)
    - _Requirements: 3.5, 4.3, 9.1_
  
  - [~] 7.4 Implement validation epoch loop
    - Create `validate_epoch()` method
    - Run inference without gradient computation
    - Compute validation metrics (loss, IoU, Dice, pixel accuracy)
    - Return metrics dictionary
    - _Requirements: 4.3, 5.3_
  
  - [~] 7.5 Implement checkpoint management
    - Create `save_checkpoint()` method
    - Save model state, optimizer state, epoch, and metrics
    - Implement best checkpoint tracking based on validation IoU
    - Add checkpoint loading functionality for resuming
    - _Requirements: 4.4, 6.6, 10.6_
  
  - [~] 7.6 Implement early stopping mechanism
    - Create `check_early_stopping()` method
    - Track validation metric history
    - Trigger stopping after patience epochs without improvement
    - Log early stopping events
    - _Requirements: 3.7, 4.5_
  
  - [ ]* 7.7 Write unit tests for training orchestrator
    - Test optimizer and scheduler creation
    - Test checkpoint saving and loading
    - Test early stopping triggers after patience
    - Test gradient clipping is applied
    - _Requirements: 3.3, 3.7, 4.4_

- [ ] 8. Implement advanced training techniques
  - [~] 8.1 Implement hard example mining
    - Create `apply_hard_example_mining()` function
    - Select top-k hardest samples based on loss values
    - Integrate into training loop when enabled
    - _Requirements: 7.1_
  
  - [~] 8.2 Implement multi-scale training
    - Create `multiscale_forward()` function
    - Randomly resize inputs to scales [0.75, 1.0, 1.25]
    - Average predictions across scales
    - Integrate into training loop when enabled
    - _Requirements: 7.4_
  
  - [~] 8.3 Implement complete training loop
    - Create `train()` method orchestrating full training
    - Iterate through epochs calling train_epoch and validate_epoch
    - Update learning rate scheduler after each epoch
    - Save checkpoints when validation IoU improves
    - Check early stopping condition
    - Log metrics to experiment tracker
    - Generate and save training curves
    - _Requirements: 4.6, 9.1, 9.2, 9.5, 9.6_
  
  - [ ]* 8.4 Write integration tests for training
    - Test full training loop executes without errors
    - Test training with hard example mining enabled
    - Test training with multi-scale enabled
    - Test checkpoint saving and resuming
    - _Requirements: 7.1, 7.4, 10.6_

- [~] 9. Checkpoint - Verify training implementation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement evaluation engine
  - [~] 10.1 Create evaluation engine class
    - Implement `EvaluationEngine` class in `evaluator.py`
    - Add `__init__()` accepting model path and test dataset path
    - Create `MetricsDict` dataclass for evaluation results
    - Setup model loading from checkpoint
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [~] 10.2 Implement metrics computation
    - Create `compute_metrics()` method in `utils/metrics.py`
    - Implement IoU calculation per class and mean IoU
    - Implement Dice coefficient calculation
    - Implement pixel accuracy calculation
    - Add precision and recall metrics
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [~] 10.3 Implement test-time augmentation
    - Create `apply_test_time_augmentation()` method
    - Define TTA transforms: original, hflip, vflip, rot90, rot270
    - Apply each transform, run inference, apply inverse transform
    - Average predictions across all augmented versions
    - _Requirements: 7.2_
  
  - [~] 10.4 Implement inference and evaluation
    - Create `run_inference()` method
    - Iterate through test dataset with optional TTA
    - Collect predictions and targets
    - Compute comprehensive metrics
    - _Requirements: 5.1, 5.5, 7.2_
  
  - [~] 10.5 Implement comparison and visualization
    - Create `compare_with_baseline()` method
    - Generate comparison report showing improvements and regressions
    - Implement `generate_comparison_visualizations()` method
    - Create side-by-side images: input, ground truth, baseline, new prediction
    - Generate visualizations for 20 sample images
    - _Requirements: 5.4, 5.6_
  
  - [~] 10.6 Implement results saving
    - Create `save_results()` method
    - Save metrics to JSON file
    - Save comparison report
    - Save visualization images
    - _Requirements: 5.5, 5.6_
  
  - [ ]* 10.7 Write unit tests for evaluation engine
    - Test IoU computation with known predictions and targets
    - Test TTA applies 5 transforms and averages correctly
    - Test comparison identifies improvements
    - Test metrics JSON serialization
    - _Requirements: 5.1, 5.2, 5.4, 7.2_

- [ ] 11. Implement dataset classes and data loading
  - [x] 11.1 Create dataset classes
    - Create `data/dataset.py` with `SegmentationDataset` class
    - Implement `__init__()`, `__len__()`, and `__getitem__()` methods
    - Add support for loading color images and segmentation masks
    - Implement mask value mapping from raw values to class IDs
    - Handle corrupted files gracefully with logging
    - _Requirements: 2.4, 8.1_
  
  - [x] 11.2 Implement data transforms
    - Create `data/transforms.py` with transform utilities
    - Implement normalization for DINOv2 input requirements
    - Add resize transforms for 14×14 patch alignment (476×266)
    - Create training and validation transform pipelines
    - _Requirements: 4.1_
  
  - [~] 11.3 Create data loaders
    - Implement `create_dataloaders()` method in `trainer.py`
    - Setup training loader with augmented dataset
    - Setup validation loader with original validation set
    - Configure batch size, num_workers, and shuffling
    - _Requirements: 2.4, 4.1_
  
  - [ ]* 11.4 Write unit tests for data loading
    - Test dataset loads images and masks correctly
    - Test mask value mapping is applied
    - Test corrupted files are skipped
    - Test data loaders produce correct batch shapes
    - _Requirements: 2.4, 8.1_

- [ ] 12. Implement main pipeline orchestration
  - [~] 12.1 Create pipeline orchestration class
    - Implement `IoUPipeline` class in `pipeline.py`
    - Add `__init__()` accepting `PipelineConfig`
    - Setup experiment tracker and logging
    - Create experiment directory structure
    - _Requirements: 10.1, 10.2, 10.5_
  
  - [~] 12.2 Implement analysis phase
    - Create `run_analysis_phase()` method
    - Instantiate `DatasetAnalyzer` with configuration
    - Run complete analysis and generate report
    - Log analysis results to experiment tracker
    - Handle errors gracefully with logging
    - _Requirements: 1.1, 1.5, 10.2_
  
  - [~] 12.3 Implement augmentation phase
    - Create `run_augmentation_phase()` method
    - Instantiate `DatasetEditor` with analysis report
    - Identify augmentation targets and apply transformations
    - Fix quality issues and validate augmented dataset
    - Save augmented dataset and log parameters
    - _Requirements: 2.1, 2.6, 10.2_
  
  - [~] 12.4 Implement training phase
    - Create `run_training_phase()` method
    - Instantiate `TrainingOrchestrator` with configuration
    - Build model, setup loss, optimizer, and scheduler
    - Create data loaders from augmented dataset
    - Execute training loop and save best checkpoint
    - Log training metrics to experiment tracker
    - _Requirements: 4.1, 4.6, 10.2_
  
  - [~] 12.5 Implement evaluation phase
    - Create `run_evaluation_phase()` method
    - Instantiate `EvaluationEngine` with best checkpoint
    - Run inference on test set with optional TTA
    - Compute metrics and generate comparisons
    - Save evaluation results and visualizations
    - Log final metrics to experiment tracker
    - _Requirements: 5.1, 5.6, 10.2_
  
  - [~] 12.6 Implement main pipeline execution
    - Create `run()` method orchestrating all phases
    - Execute phases in sequence: analysis → augmentation → training → evaluation
    - Implement error handling with graceful degradation
    - Update leaderboard with final results
    - Generate summary report
    - Support dry-run mode for configuration validation
    - _Requirements: 10.1, 10.2, 10.4, 10.5_
  
  - [ ]* 12.7 Write integration tests for pipeline
    - Test full pipeline executes all phases
    - Test pipeline handles phase failures gracefully
    - Test dry-run mode validates without execution
    - Test pipeline resumes from checkpoint
    - _Requirements: 10.1, 10.2, 10.4, 10.6_

- [~] 13. Checkpoint - Verify pipeline integration
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Create configuration files and CLI interface
  - [~] 14.1 Create default configuration files
    - Create `configs/` directory
    - Write `configs/default.yaml` with baseline settings
    - Write `configs/optimized.yaml` with improved hyperparameters
    - Write `configs/advanced.yaml` with advanced techniques enabled
    - _Requirements: 10.3_
  
  - [~] 14.2 Implement CLI entry point
    - Create `scripts/run_pipeline.py` as main entry point
    - Add argument parser for config file, experiment name, dry-run, resume
    - Load and validate configuration
    - Instantiate and run pipeline
    - Print summary report with final metrics
    - _Requirements: 10.3, 10.5_
  
  - [~] 14.3 Create standalone utility scripts
    - Create `scripts/analyze_dataset.py` for standalone analysis
    - Create `scripts/augment_dataset.py` for standalone augmentation
    - Create `scripts/evaluate_model.py` for standalone evaluation
    - Add argument parsers and documentation for each script
    - _Requirements: 10.1_
  
  - [ ]* 14.4 Write integration tests for CLI
    - Test CLI runs pipeline with config file
    - Test CLI validates configuration
    - Test CLI handles invalid arguments
    - Test standalone scripts execute correctly
    - _Requirements: 10.3, 10.4_

- [ ] 15. Implement monitoring and visualization utilities
  - [~] 15.1 Create training visualization utilities
    - Create `utils/visualization.py` module
    - Implement function to generate loss curves
    - Implement function to generate IoU curves
    - Implement function to generate per-class metrics plots
    - Add color palette utilities for mask visualization
    - _Requirements: 9.2, 9.3, 9.6_
  
  - [~] 15.2 Implement progress monitoring
    - Add progress bars using tqdm for training epochs
    - Implement remaining time estimation based on epoch duration
    - Add stagnation detection and warning logging
    - Log per-class IoU scores every 5 epochs
    - _Requirements: 9.1, 9.3, 9.4, 9.5_
  
  - [~] 15.3 Create comparison visualization utilities
    - Implement side-by-side comparison image generation
    - Add high-contrast color overlay for predictions
    - Create grid layouts for multiple samples
    - Save comparison images with descriptive filenames
    - _Requirements: 5.6_
  
  - [ ]* 15.4 Write unit tests for visualization
    - Test loss curve generation with sample data
    - Test IoU curve generation
    - Test comparison image creation
    - Test color palette application
    - _Requirements: 9.2, 9.6, 5.6_

- [ ] 16. Add error recovery and robustness features
  - [~] 16.1 Implement automatic error recovery
    - Add OOM error detection and batch size reduction
    - Implement NaN loss detection with checkpoint saving
    - Add gradient explosion handling with clipping
    - Implement automatic retry logic for transient failures
    - _Requirements: 10.2_
  
  - [~] 16.2 Implement input validation
    - Add path existence validation for all file inputs
    - Validate hyperparameter ranges in configuration
    - Check image and mask file formats before loading
    - Validate model architecture matches checkpoint
    - _Requirements: 10.4_
  
  - [~] 16.3 Add comprehensive error logging
    - Log detailed error information with stack traces
    - Add context information (epoch, batch, sample ID) to errors
    - Implement error categorization (dataset, config, training, evaluation)
    - Create error summary in final report
    - _Requirements: 10.2, 10.5_
  
  - [ ]* 16.4 Write tests for error handling
    - Test OOM recovery reduces batch size
    - Test NaN loss detection saves checkpoint
    - Test invalid configuration raises appropriate errors
    - Test corrupted files are handled gracefully
    - _Requirements: 10.2_

- [ ] 17. Create documentation and examples
  - [~] 17.1 Write API documentation
    - Document all public classes and methods with docstrings
    - Create module-level documentation
    - Document configuration options and valid ranges
    - Add type hints to all function signatures
    - _Requirements: 10.1_
  
  - [~] 17.2 Create usage examples
    - Write example for basic pipeline execution
    - Write example for custom configuration
    - Write example for resuming interrupted training
    - Write example for standalone analysis and evaluation
    - _Requirements: 10.1_
  
  - [~] 17.3 Write README documentation
    - Create comprehensive README.md with installation instructions
    - Document dependencies and environment setup
    - Add usage examples and command-line options
    - Include troubleshooting guide for common issues
    - Document expected directory structure and outputs
    - _Requirements: 10.1, 10.5_

- [ ] 18. Final integration and validation
  - [~] 18.1 Run end-to-end pipeline test
    - Execute full pipeline with default configuration
    - Verify all phases complete successfully
    - Check all output files are generated correctly
    - Validate metrics are computed and logged
    - _Requirements: 10.1, 10.5_
  
  - [~] 18.2 Validate against requirements
    - Verify all acceptance criteria are met
    - Test with different configuration combinations
    - Validate IoU improvement over baseline
    - Check experiment tracking and reproducibility
    - _Requirements: 1.1-10.6_
  
  - [~] 18.3 Performance optimization
    - Profile pipeline execution to identify bottlenecks
    - Optimize data loading with appropriate num_workers
    - Verify mixed precision training reduces memory usage
    - Test with different batch sizes for optimal throughput
    - _Requirements: 3.5, 4.1_
  
  - [ ]* 18.4 Run full test suite
    - Execute all unit tests with coverage reporting
    - Run all integration tests
    - Verify test coverage is above 80%
    - Fix any failing tests
    - _Requirements: 10.1_

- [~] 19. Final checkpoint - Complete validation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The implementation uses Python with PyTorch, following the existing project structure
- The pipeline maintains compatibility with existing scripts (`train_segmentation (1).py`, `test_segmentation (1).py`)
- Checkpoints ensure incremental validation throughout development
- No property-based tests are included as this is primarily an infrastructure/orchestration feature
- The design uses DINOv2 backbone (frozen) with ConvNeXt-style segmentation head
- All configuration is managed through YAML/JSON files for reproducibility
- Experiment tracking enables comparison and reproducibility of training runs

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["1.4", "1.5"] },
    { "id": 2, "tasks": ["3.1", "11.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "11.2"] },
    { "id": 4, "tasks": ["3.4", "3.5", "3.6", "11.3", "11.4"] },
    { "id": 5, "tasks": ["4.1", "4.2", "6.1"] },
    { "id": 6, "tasks": ["4.3", "4.4", "6.2"] },
    { "id": 7, "tasks": ["4.5", "4.6", "4.7", "6.3", "6.4"] },
    { "id": 8, "tasks": ["7.1", "7.2", "15.1"] },
    { "id": 9, "tasks": ["7.3", "7.4", "15.2"] },
    { "id": 10, "tasks": ["7.5", "7.6", "7.7", "15.3", "15.4"] },
    { "id": 11, "tasks": ["8.1", "8.2", "10.1"] },
    { "id": 12, "tasks": ["8.3", "8.4", "10.2"] },
    { "id": 13, "tasks": ["10.3", "10.4"] },
    { "id": 14, "tasks": ["10.5", "10.6", "10.7"] },
    { "id": 15, "tasks": ["12.1", "12.2", "14.1"] },
    { "id": 16, "tasks": ["12.3", "12.4", "14.2"] },
    { "id": 17, "tasks": ["12.5", "12.6", "12.7", "14.3", "14.4"] },
    { "id": 18, "tasks": ["16.1", "16.2", "17.1"] },
    { "id": 19, "tasks": ["16.3", "16.4", "17.2"] },
    { "id": 20, "tasks": ["17.3", "18.1"] },
    { "id": 21, "tasks": ["18.2", "18.3", "18.4"] }
  ]
}
```
