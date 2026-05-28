# Experiment Tracking System

## Overview

The Experiment Tracking System provides comprehensive tracking, logging, and management of machine learning experiments for the IoU Improvement Pipeline. It enables reproducibility, comparison, and organization of training runs.

## Features

### Core Functionality

1. **Unique Experiment IDs**: Timestamp-based unique identifiers for each experiment
2. **Configuration Logging**: Save and track all hyperparameters and settings
3. **Metrics Tracking**: Time-series logging of training and validation metrics
4. **Leaderboard Management**: Automatic ranking of experiments by validation IoU
5. **Artifact Management**: Track checkpoints, plots, logs, and visualizations
6. **Experiment Restoration**: Load and reproduce any previous experiment
7. **Candidate Marking**: Automatically mark experiments with IoU > 0.70 as candidates

### Directory Structure

Each experiment creates the following structure:

```
experiments/
├── leaderboard.json                    # Global leaderboard
└── {experiment_id}/
    ├── experiment_info.json            # Experiment metadata
    ├── config.json                     # Training configuration
    ├── metrics.jsonl                   # Time-series metrics (one per line)
    ├── artifacts.json                  # Artifact paths manifest
    ├── checkpoints/                    # Model checkpoints
    ├── plots/                          # Training curves and visualizations
    ├── logs/                           # Training logs
    └── visualizations/                 # Prediction visualizations
```

## Usage

### Basic Usage

```python
from iou_pipeline.tracker import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(experiments_dir='./experiments')

# Create new experiment
exp_id = tracker.create_experiment(name="My Experiment")

# Log configuration
config = {
    'learning_rate': 0.001,
    'batch_size': 8,
    'num_epochs': 100,
    'optimizer': 'sgd'
}
tracker.log_config(exp_id, config)

# Log metrics during training
for epoch in range(num_epochs):
    # ... training code ...
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_iou': val_iou,
        'val_dice': val_dice
    }
    tracker.log_metrics(exp_id, metrics, step=epoch)

# Update leaderboard with final validation IoU
tracker.update_leaderboard(exp_id, val_iou=final_val_iou)

# Save artifacts
artifacts = {
    'best_checkpoint': f'./experiments/{exp_id}/checkpoints/best_model.pth',
    'loss_curve': f'./experiments/{exp_id}/plots/loss_curve.png'
}
tracker.save_artifacts(exp_id, artifacts)

# Mark experiment as completed
tracker.update_status(exp_id, 'completed')
```

### Viewing Leaderboard

```python
# Get top 10 experiments
leaderboard = tracker.get_leaderboard(top_k=10)

for i, exp_info in enumerate(leaderboard, 1):
    print(f"{i}. {exp_info.name}")
    print(f"   Val IoU: {exp_info.metrics['val_iou']:.3f}")
    print(f"   Status: {exp_info.status}")
```

### Loading Previous Experiments

```python
# Load experiment by ID
exp_info = tracker.load_experiment(experiment_id)

# Access configuration
config = exp_info.config

# Access metrics
final_metrics = exp_info.metrics

# Access artifacts path
artifacts_path = exp_info.artifacts_path
```

## API Reference

### ExperimentTracker

#### `__init__(experiments_dir: str = './experiments')`

Initialize the experiment tracker.

**Parameters:**
- `experiments_dir`: Root directory for all experiments (default: './experiments')

#### `create_experiment(name: Optional[str] = None) -> str`

Create a new experiment with unique ID.

**Parameters:**
- `name`: Optional human-readable name for the experiment

**Returns:**
- `experiment_id`: Unique timestamp-based experiment identifier

#### `log_config(experiment_id: str, config: Dict[str, Any])`

Log experiment configuration.

**Parameters:**
- `experiment_id`: Experiment identifier
- `config`: Configuration dictionary with hyperparameters

#### `log_metrics(experiment_id: str, metrics: Dict[str, float], step: Optional[int] = None)`

Log experiment metrics.

**Parameters:**
- `experiment_id`: Experiment identifier
- `metrics`: Dictionary of metric names and values
- `step`: Optional step/epoch number

#### `update_leaderboard(experiment_id: str, val_iou: float)`

Update leaderboard with experiment results.

**Parameters:**
- `experiment_id`: Experiment identifier
- `val_iou`: Validation Mean IoU score

**Note:** Experiments with `val_iou > 0.70` are automatically marked as candidates.

#### `get_leaderboard(top_k: int = 10) -> List[ExperimentInfo]`

Get top experiments from leaderboard.

**Parameters:**
- `top_k`: Number of top experiments to return (default: 10)

**Returns:**
- List of `ExperimentInfo` objects sorted by validation IoU (descending)

#### `save_artifacts(experiment_id: str, artifacts: Dict[str, str])`

Save experiment artifacts.

**Parameters:**
- `experiment_id`: Experiment identifier
- `artifacts`: Dictionary mapping artifact names to file paths

#### `load_experiment(experiment_id: str) -> ExperimentInfo`

Load experiment information and configuration.

**Parameters:**
- `experiment_id`: Experiment identifier

**Returns:**
- `ExperimentInfo` object with all experiment data

#### `update_status(experiment_id: str, status: str)`

Update experiment status.

**Parameters:**
- `experiment_id`: Experiment identifier
- `status`: New status ('running', 'completed', 'failed')

### ExperimentInfo

Data class containing experiment information:

```python
@dataclass
class ExperimentInfo:
    experiment_id: str              # Unique experiment identifier
    name: Optional[str]             # Human-readable name
    config: Dict[str, Any]          # Configuration dictionary
    metrics: Dict[str, float]       # Latest metrics
    timestamp: str                  # ISO format timestamp
    status: str                     # 'running', 'completed', 'failed'
    artifacts_path: str             # Path to experiment directory
```

## File Formats

### experiment_info.json

Contains experiment metadata and latest metrics:

```json
{
  "experiment_id": "20260518_101830",
  "name": "Optimized Configuration",
  "config": { ... },
  "metrics": {
    "val_iou": 0.85,
    "val_dice": 0.90,
    "is_candidate": true
  },
  "timestamp": "2026-05-18T10:18:30.232087",
  "status": "completed",
  "artifacts_path": "experiments/20260518_101830"
}
```

### config.json

Contains training configuration:

```json
{
  "learning_rate": 0.001,
  "batch_size": 8,
  "num_epochs": 100,
  "optimizer": "sgd",
  "scheduler": "cosine",
  "use_class_weights": true,
  "use_amp": true
}
```

### metrics.jsonl

Time-series metrics (one JSON object per line):

```jsonl
{"timestamp": "2026-05-18T10:18:30.250572", "step": 1, "metrics": {"train_loss": 0.52, "val_loss": 0.58, "val_iou": 0.65}}
{"timestamp": "2026-05-18T10:18:30.262135", "step": 2, "metrics": {"train_loss": 0.44, "val_loss": 0.51, "val_iou": 0.70}}
```

### artifacts.json

Artifact paths manifest:

```json
{
  "best_checkpoint": "./experiments/20260518_101830/checkpoints/best_model.pth",
  "final_checkpoint": "./experiments/20260518_101830/checkpoints/final_model.pth",
  "loss_curve": "./experiments/20260518_101830/plots/loss_curve.png",
  "iou_curve": "./experiments/20260518_101830/plots/iou_curve.png"
}
```

### leaderboard.json

Global leaderboard with top experiments:

```json
[
  {
    "experiment_id": "20260518_101830",
    "name": "Optimized Configuration",
    "metrics": {"val_iou": 0.85, "is_candidate": true},
    ...
  },
  ...
]
```

## Requirements Mapping

This implementation satisfies the following requirements from the specification:

- **Requirement 6.1**: Log all training configuration parameters for each experiment
- **Requirement 6.2**: Assign unique experiment identifier to each training run
- **Requirement 6.3**: Save dataset statistics, augmentation parameters, and training logs
- **Requirement 6.4**: Mark experiments with Mean IoU > 0.70 as candidates
- **Requirement 6.5**: Maintain leaderboard ranking experiments by validation Mean IoU
- **Requirement 6.6**: Enable restoration of any previous experiment by loading configuration and checkpoint

## Examples

See the following example scripts:

- `test_tracker.py`: Unit tests demonstrating all functionality
- `demo_tracker.py`: Comprehensive demonstration with multiple experiments

Run the demo:

```bash
python demo_tracker.py
```

## Integration with Pipeline

The experiment tracker integrates with the IoU Improvement Pipeline:

```python
from iou_pipeline.pipeline import IoUPipeline
from iou_pipeline.tracker import ExperimentTracker

# Pipeline automatically uses tracker
pipeline = IoUPipeline(config)
pipeline.run()  # Tracker logs all phases automatically
```

## Best Practices

1. **Unique Names**: Use descriptive names for experiments to easily identify them
2. **Regular Logging**: Log metrics after each epoch for detailed tracking
3. **Artifact Organization**: Use consistent naming for artifacts across experiments
4. **Status Updates**: Always update status to 'completed' or 'failed' when done
5. **Leaderboard Review**: Regularly review leaderboard to identify best configurations
6. **Candidate Experiments**: Focus on experiments marked as candidates (IoU > 0.70)

## Troubleshooting

### Experiment Not Found

```python
# Check if experiment exists
exp_dir = Path(experiments_dir) / experiment_id
if not exp_dir.exists():
    print(f"Experiment {experiment_id} not found")
```

### Corrupted Experiment Data

If experiment data is corrupted, you can manually inspect the JSON files:

```bash
cat experiments/{experiment_id}/experiment_info.json
cat experiments/{experiment_id}/config.json
cat experiments/{experiment_id}/metrics.jsonl
```

### Leaderboard Not Updating

Ensure you call `update_leaderboard()` after training completes:

```python
tracker.update_leaderboard(exp_id, val_iou=final_val_iou)
```

## Future Enhancements

Potential future improvements:

- Web-based dashboard for experiment visualization
- Automatic hyperparameter optimization integration
- Experiment comparison tools
- Export to MLflow or Weights & Biases
- Distributed experiment tracking across multiple machines
