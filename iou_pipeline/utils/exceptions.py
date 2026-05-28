"""
Custom exception hierarchy for the IoU Improvement Pipeline.

This module defines a comprehensive exception hierarchy for different types of
errors that can occur during pipeline execution, enabling precise error handling
and recovery strategies.
"""


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize pipeline error.
        
        Args:
            message: Error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            details_str = ', '.join(f'{k}={v}' for k, v in self.details.items())
            return f'{self.message} ({details_str})'
        return self.message


# Configuration Errors
class ConfigurationError(PipelineError):
    """Raised when configuration is invalid or missing required parameters."""
    pass


class InvalidHyperparameterError(ConfigurationError):
    """Raised when a hyperparameter value is outside valid range."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration parameters are missing."""
    pass


# Dataset Errors
class DatasetError(PipelineError):
    """Base exception for dataset-related errors."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when dataset path does not exist."""
    pass


class CorruptedDataError(DatasetError):
    """Raised when dataset contains corrupted or invalid files."""
    pass


class InvalidMaskError(DatasetError):
    """Raised when segmentation mask contains invalid values."""
    pass


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails."""
    pass


class InsufficientDataError(DatasetError):
    """Raised when dataset does not meet minimum size requirements."""
    pass


# Model Errors
class ModelError(PipelineError):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model checkpoint cannot be loaded."""
    pass


class ModelBuildError(ModelError):
    """Raised when model architecture cannot be built."""
    pass


class CheckpointNotFoundError(ModelError):
    """Raised when checkpoint file does not exist."""
    pass


class IncompatibleCheckpointError(ModelError):
    """Raised when checkpoint is incompatible with model architecture."""
    pass


# Training Errors
class TrainingError(PipelineError):
    """Base exception for training-related errors."""
    pass


class OutOfMemoryError(TrainingError):
    """Raised when GPU/CPU runs out of memory during training."""
    pass


class NaNLossError(TrainingError):
    """Raised when loss becomes NaN during training."""
    pass


class GradientExplosionError(TrainingError):
    """Raised when gradients explode during training."""
    pass


class ConvergenceError(TrainingError):
    """Raised when model fails to converge."""
    pass


class EarlyStoppingError(TrainingError):
    """Raised when early stopping is triggered (informational, not critical)."""
    pass


# Evaluation Errors
class EvaluationError(PipelineError):
    """Base exception for evaluation-related errors."""
    pass


class InferenceError(EvaluationError):
    """Raised when inference fails on test data."""
    pass


class MetricsComputationError(EvaluationError):
    """Raised when metrics computation fails."""
    pass


# I/O Errors
class IOError(PipelineError):
    """Base exception for input/output errors."""
    pass


class FileReadError(IOError):
    """Raised when file cannot be read."""
    pass


class FileWriteError(IOError):
    """Raised when file cannot be written."""
    pass


class DirectoryCreationError(IOError):
    """Raised when directory cannot be created."""
    pass


# Experiment Tracking Errors
class ExperimentError(PipelineError):
    """Base exception for experiment tracking errors."""
    pass


class ExperimentNotFoundError(ExperimentError):
    """Raised when experiment ID does not exist."""
    pass


class LeaderboardError(ExperimentError):
    """Raised when leaderboard operations fail."""
    pass


# Augmentation Errors
class AugmentationError(PipelineError):
    """Base exception for augmentation-related errors."""
    pass


class AlignmentError(AugmentationError):
    """Raised when augmented mask and image are misaligned."""
    pass


class TransformError(AugmentationError):
    """Raised when augmentation transform fails."""
    pass


# Utility Functions
def handle_error(error: Exception, logger, context: dict = None) -> PipelineError:
    """
    Convert generic exceptions to pipeline-specific exceptions with context.
    
    Args:
        error: Original exception
        logger: Logger instance for error logging
        context: Optional context dictionary with additional information
        
    Returns:
        PipelineError instance with appropriate type
        
    Example:
        >>> try:
        ...     model.load_state_dict(checkpoint)
        ... except Exception as e:
        ...     raise handle_error(e, logger, {'checkpoint_path': path})
    """
    context = context or {}
    
    # Map common exceptions to pipeline exceptions
    if isinstance(error, FileNotFoundError):
        if 'checkpoint' in str(error).lower():
            return CheckpointNotFoundError(str(error), context)
        elif 'dataset' in str(error).lower():
            return DatasetNotFoundError(str(error), context)
        else:
            return FileReadError(str(error), context)
    
    elif isinstance(error, PermissionError):
        return FileWriteError(str(error), context)
    
    elif isinstance(error, ValueError):
        if 'shape' in str(error).lower() or 'dimension' in str(error).lower():
            return AlignmentError(str(error), context)
        else:
            return InvalidHyperparameterError(str(error), context)
    
    elif isinstance(error, RuntimeError):
        error_msg = str(error).lower()
        if 'out of memory' in error_msg or 'oom' in error_msg:
            return OutOfMemoryError(str(error), context)
        elif 'cuda' in error_msg:
            return TrainingError(str(error), context)
        else:
            return ModelError(str(error), context)
    
    elif isinstance(error, KeyError):
        return MissingConfigurationError(str(error), context)
    
    # If already a PipelineError, return as-is
    elif isinstance(error, PipelineError):
        return error
    
    # Default: wrap in generic PipelineError
    return PipelineError(str(error), context)


def is_recoverable(error: Exception) -> bool:
    """
    Check if an error is recoverable with automatic retry or adjustment.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is recoverable, False otherwise
        
    Example:
        >>> if is_recoverable(error):
        ...     retry_with_reduced_batch_size()
    """
    recoverable_types = (
        OutOfMemoryError,
        GradientExplosionError,
        CorruptedDataError,
        FileReadError,
    )
    return isinstance(error, recoverable_types)


def get_recovery_strategy(error: Exception) -> str:
    """
    Get recommended recovery strategy for an error.
    
    Args:
        error: Exception to analyze
        
    Returns:
        String describing recovery strategy
        
    Example:
        >>> strategy = get_recovery_strategy(OutOfMemoryError('CUDA OOM'))
        >>> print(strategy)  # 'reduce_batch_size'
    """
    if isinstance(error, OutOfMemoryError):
        return 'reduce_batch_size'
    elif isinstance(error, GradientExplosionError):
        return 'reduce_learning_rate'
    elif isinstance(error, NaNLossError):
        return 'restore_checkpoint'
    elif isinstance(error, CorruptedDataError):
        return 'skip_sample'
    elif isinstance(error, FileReadError):
        return 'retry_read'
    else:
        return 'abort'
