# IoU Pipeline Utilities

This directory contains utility modules for the IoU Improvement Pipeline, providing logging, exception handling, and error recovery capabilities.

## Modules

### `logging.py`
Centralized logging configuration with file and console handlers.

**Key Functions:**
- `setup_logging()`: Initialize logging with file and console handlers
- `get_logger()`: Get a logger instance for a specific module
- `log_section()`: Log section headers for better readability
- `log_metrics()`: Log metrics in a formatted way
- `log_config()`: Log configuration parameters

**Example:**
```python
from iou_pipeline.utils import setup_logging, log_metrics

logger = setup_logging(
    log_dir='./experiments/exp_001/logs',
    experiment_id='exp_001'
)

logger.info('Starting training phase')

metrics = {'mean_iou': 0.75, 'pixel_accuracy': 0.89}
log_metrics(logger, metrics, prefix='Validation')
```

### `exceptions.py`
Custom exception hierarchy for precise error handling.

**Exception Categories:**
- **Configuration Errors**: `ConfigurationError`, `InvalidHyperparameterError`, `MissingConfigurationError`
- **Dataset Errors**: `DatasetError`, `DatasetNotFoundError`, `CorruptedDataError`, `InvalidMaskError`
- **Model Errors**: `ModelError`, `ModelLoadError`, `CheckpointNotFoundError`
- **Training Errors**: `TrainingError`, `OutOfMemoryError`, `NaNLossError`, `GradientExplosionError`
- **Evaluation Errors**: `EvaluationError`, `InferenceError`, `MetricsComputationError`
- **I/O Errors**: `IOError`, `FileReadError`, `FileWriteError`
- **Augmentation Errors**: `AugmentationError`, `AlignmentError`, `TransformError`

**Utility Functions:**
- `handle_error()`: Convert generic exceptions to pipeline-specific exceptions
- `is_recoverable()`: Check if an error is recoverable
- `get_recovery_strategy()`: Get recommended recovery strategy

**Example:**
```python
from iou_pipeline.utils import OutOfMemoryError, handle_error

try:
    # Training code
    model.train()
except RuntimeError as e:
    pipeline_error = handle_error(e, logger, {'batch_size': 32})
    if isinstance(pipeline_error, OutOfMemoryError):
        # Apply recovery strategy
        reduce_batch_size()
```

### `recovery.py`
Error recovery utilities for automatic retry and resource adjustment.

**Key Classes:**
- `BatchSizeReducer`: Automatically reduce batch size on OOM errors
- `LearningRateReducer`: Automatically reduce learning rate on gradient issues
- `CheckpointRecovery`: Checkpoint-based recovery from training failures
- `CorruptedDataHandler`: Track and skip corrupted data samples

**Decorators:**
- `@retry_on_failure`: Automatic retry with exponential backoff

**Utility Functions:**
- `safe_file_read()`: Read files with automatic retry
- `safe_file_write()`: Write files with automatic retry

**Example:**
```python
from iou_pipeline.utils import BatchSizeReducer, retry_on_failure

# Automatic batch size reduction
reducer = BatchSizeReducer(initial_batch_size=32, min_batch_size=2)

try:
    train_model(batch_size=reducer.current_batch_size)
except OutOfMemoryError:
    new_size = reducer.reduce(factor=0.5)
    train_model(batch_size=new_size)

# Automatic retry
@retry_on_failure(max_retries=3, delay=1.0)
def load_dataset(path):
    return Dataset(path)
```

## Design Principles

1. **Comprehensive Error Handling**: Custom exception hierarchy enables precise error identification and recovery
2. **Automatic Recovery**: Built-in utilities for common failure scenarios (OOM, gradient explosion, corrupted data)
3. **Detailed Logging**: File and console handlers with appropriate formatters for debugging and monitoring
4. **Graceful Degradation**: Recovery strategies allow pipeline to continue with adjusted parameters
5. **Context Preservation**: Exceptions carry context information for better debugging

## Requirements Mapping

This module implements:
- **Requirement 10.2**: Error handling and graceful termination
- **Requirement 10.5**: Summary report generation with error details

## Testing

Run the test suite:
```bash
python test_logging_infrastructure.py
```

Run the demonstration:
```bash
python demo_logging_usage.py
```

## Integration

All pipeline components should use these utilities:

```python
from iou_pipeline.utils import (
    setup_logging,
    get_logger,
    PipelineError,
    handle_error,
    BatchSizeReducer
)

# Setup logging at pipeline start
logger = setup_logging(log_dir='./logs', experiment_id='exp_001')

# Use module-specific loggers
logger = get_logger('dataset_analyzer')
logger.info('Starting analysis')

# Handle errors with context
try:
    process_data()
except Exception as e:
    pipeline_error = handle_error(e, logger, {'phase': 'analysis'})
    if is_recoverable(pipeline_error):
        apply_recovery_strategy(pipeline_error)
    else:
        raise
```
