"""
Utility modules for the IoU Improvement Pipeline.

This package provides logging, exception handling, error recovery utilities,
and configuration management for robust pipeline execution.
"""

from .logging import (
    setup_logging,
    get_logger,
    log_section,
    log_metrics,
    log_config
)

from .config import (
    TrainingConfig,
    PipelineConfig,
    ConfigManager
)

from .exceptions import (
    # Base exceptions
    PipelineError,
    
    # Configuration errors
    ConfigurationError,
    InvalidHyperparameterError,
    MissingConfigurationError,
    
    # Dataset errors
    DatasetError,
    DatasetNotFoundError,
    CorruptedDataError,
    InvalidMaskError,
    DatasetValidationError,
    InsufficientDataError,
    
    # Model errors
    ModelError,
    ModelLoadError,
    ModelBuildError,
    CheckpointNotFoundError,
    IncompatibleCheckpointError,
    
    # Training errors
    TrainingError,
    OutOfMemoryError,
    NaNLossError,
    GradientExplosionError,
    ConvergenceError,
    EarlyStoppingError,
    
    # Evaluation errors
    EvaluationError,
    InferenceError,
    MetricsComputationError,
    
    # I/O errors
    IOError,
    FileReadError,
    FileWriteError,
    DirectoryCreationError,
    
    # Experiment tracking errors
    ExperimentError,
    ExperimentNotFoundError,
    LeaderboardError,
    
    # Augmentation errors
    AugmentationError,
    AlignmentError,
    TransformError,
    
    # Utility functions
    handle_error,
    is_recoverable,
    get_recovery_strategy
)

from .recovery import (
    retry_on_failure,
    BatchSizeReducer,
    LearningRateReducer,
    CheckpointRecovery,
    CorruptedDataHandler,
    safe_file_read,
    safe_file_write
)

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    'log_section',
    'log_metrics',
    'log_config',
    
    # Configuration management
    'TrainingConfig',
    'PipelineConfig',
    'ConfigManager',
    
    # Base exceptions
    'PipelineError',
    
    # Configuration errors
    'ConfigurationError',
    'InvalidHyperparameterError',
    'MissingConfigurationError',
    
    # Dataset errors
    'DatasetError',
    'DatasetNotFoundError',
    'CorruptedDataError',
    'InvalidMaskError',
    'DatasetValidationError',
    'InsufficientDataError',
    
    # Model errors
    'ModelError',
    'ModelLoadError',
    'ModelBuildError',
    'CheckpointNotFoundError',
    'IncompatibleCheckpointError',
    
    # Training errors
    'TrainingError',
    'OutOfMemoryError',
    'NaNLossError',
    'GradientExplosionError',
    'ConvergenceError',
    'EarlyStoppingError',
    
    # Evaluation errors
    'EvaluationError',
    'InferenceError',
    'MetricsComputationError',
    
    # I/O errors
    'IOError',
    'FileReadError',
    'FileWriteError',
    'DirectoryCreationError',
    
    # Experiment tracking errors
    'ExperimentError',
    'ExperimentNotFoundError',
    'LeaderboardError',
    
    # Augmentation errors
    'AugmentationError',
    'AlignmentError',
    'TransformError',
    
    # Exception utilities
    'handle_error',
    'is_recoverable',
    'get_recovery_strategy',
    
    # Recovery utilities
    'retry_on_failure',
    'BatchSizeReducer',
    'LearningRateReducer',
    'CheckpointRecovery',
    'CorruptedDataHandler',
    'safe_file_read',
    'safe_file_write',
]
