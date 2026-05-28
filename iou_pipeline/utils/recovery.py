"""
Error recovery utilities for the IoU Improvement Pipeline.

This module provides utilities for automatic error recovery, including retry logic,
resource adjustment, and graceful degradation strategies.
"""

import time
import functools
from typing import Callable, Any, Optional, Tuple
from pathlib import Path

from .exceptions import (
    OutOfMemoryError,
    GradientExplosionError,
    NaNLossError,
    FileReadError,
    CorruptedDataError,
    is_recoverable,
    get_recovery_strategy
)


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        >>> @retry_on_failure(max_retries=3, delay=1.0)
        ... def load_file(path):
        ...     return open(path).read()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator


class BatchSizeReducer:
    """
    Utility for automatically reducing batch size on OOM errors.
    
    Tracks batch size reductions and provides methods to adjust batch size
    when out-of-memory errors occur during training.
    """
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 1):
        """
        Initialize batch size reducer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
        """
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_count = 0
    
    def reduce(self, factor: float = 0.5) -> int:
        """
        Reduce batch size by a factor.
        
        Args:
            factor: Reduction factor (default: 0.5 for halving)
            
        Returns:
            New batch size
            
        Raises:
            OutOfMemoryError: If batch size cannot be reduced further
        """
        new_batch_size = max(int(self.current_batch_size * factor), self.min_batch_size)
        
        if new_batch_size == self.current_batch_size:
            raise OutOfMemoryError(
                f'Cannot reduce batch size below {self.min_batch_size}',
                {'current_batch_size': self.current_batch_size}
            )
        
        self.current_batch_size = new_batch_size
        self.reduction_count += 1
        return self.current_batch_size
    
    def reset(self):
        """Reset batch size to initial value."""
        self.current_batch_size = self.initial_batch_size
        self.reduction_count = 0
    
    def can_reduce(self) -> bool:
        """Check if batch size can be reduced further."""
        return self.current_batch_size > self.min_batch_size


class LearningRateReducer:
    """
    Utility for automatically reducing learning rate on gradient issues.
    
    Tracks learning rate reductions and provides methods to adjust learning rate
    when gradient explosion or NaN loss occurs.
    """
    
    def __init__(self, initial_lr: float, min_lr: float = 1e-7):
        """
        Initialize learning rate reducer.
        
        Args:
            initial_lr: Starting learning rate
            min_lr: Minimum allowed learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.reduction_count = 0
    
    def reduce(self, factor: float = 0.1) -> float:
        """
        Reduce learning rate by a factor.
        
        Args:
            factor: Reduction factor (default: 0.1 for 10x reduction)
            
        Returns:
            New learning rate
            
        Raises:
            GradientExplosionError: If learning rate cannot be reduced further
        """
        new_lr = max(self.current_lr * factor, self.min_lr)
        
        if new_lr == self.current_lr:
            raise GradientExplosionError(
                f'Cannot reduce learning rate below {self.min_lr}',
                {'current_lr': self.current_lr}
            )
        
        self.current_lr = new_lr
        self.reduction_count += 1
        return self.current_lr
    
    def reset(self):
        """Reset learning rate to initial value."""
        self.current_lr = self.initial_lr
        self.reduction_count = 0
    
    def can_reduce(self) -> bool:
        """Check if learning rate can be reduced further."""
        return self.current_lr > self.min_lr


class CheckpointRecovery:
    """
    Utility for checkpoint-based recovery from training failures.
    
    Manages checkpoint saving and restoration for recovery from NaN loss,
    gradient explosion, or other training failures.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint recovery manager.
        
        Args:
            checkpoint_dir: Directory to store recovery checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_good_checkpoint = None
    
    def save_recovery_checkpoint(
        self,
        model_state: dict,
        optimizer_state: dict,
        epoch: int,
        metrics: dict
    ) -> str:
        """
        Save a recovery checkpoint.
        
        Args:
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            epoch: Current epoch number
            metrics: Current metrics dictionary
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f'recovery_epoch_{epoch}.pth'
        
        import torch
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'epoch': epoch,
            'metrics': metrics
        }, checkpoint_path)
        
        self.last_good_checkpoint = str(checkpoint_path)
        return str(checkpoint_path)
    
    def restore_last_checkpoint(self) -> Optional[dict]:
        """
        Restore the last good checkpoint.
        
        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        if self.last_good_checkpoint is None:
            return None
        
        checkpoint_path = Path(self.last_good_checkpoint)
        if not checkpoint_path.exists():
            return None
        
        import torch
        return torch.load(checkpoint_path)
    
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """
        Remove old recovery checkpoints, keeping only the most recent.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob('recovery_epoch_*.pth'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[keep_last:]:
            checkpoint.unlink()


class CorruptedDataHandler:
    """
    Utility for handling corrupted or invalid data samples.
    
    Tracks corrupted samples and provides methods to skip them during
    data loading and training.
    """
    
    def __init__(self):
        """Initialize corrupted data handler."""
        self.corrupted_samples = set()
        self.skip_count = 0
    
    def mark_corrupted(self, sample_id: str):
        """
        Mark a sample as corrupted.
        
        Args:
            sample_id: Identifier for the corrupted sample
        """
        self.corrupted_samples.add(sample_id)
    
    def is_corrupted(self, sample_id: str) -> bool:
        """
        Check if a sample is marked as corrupted.
        
        Args:
            sample_id: Sample identifier to check
            
        Returns:
            True if sample is corrupted, False otherwise
        """
        return sample_id in self.corrupted_samples
    
    def skip_sample(self, sample_id: str):
        """
        Skip a corrupted sample and increment skip counter.
        
        Args:
            sample_id: Sample identifier to skip
        """
        self.mark_corrupted(sample_id)
        self.skip_count += 1
    
    def get_corrupted_list(self) -> list:
        """
        Get list of all corrupted sample IDs.
        
        Returns:
            List of corrupted sample identifiers
        """
        return list(self.corrupted_samples)
    
    def save_corrupted_list(self, output_path: str):
        """
        Save list of corrupted samples to file.
        
        Args:
            output_path: Path to save corrupted samples list
        """
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'corrupted_samples': list(self.corrupted_samples),
                'total_count': len(self.corrupted_samples),
                'skip_count': self.skip_count
            }, f, indent=2)


def safe_file_read(file_path: str, max_retries: int = 3) -> Any:
    """
    Safely read a file with automatic retry on failure.
    
    Args:
        file_path: Path to file to read
        max_retries: Maximum number of retry attempts
        
    Returns:
        File contents
        
    Raises:
        FileReadError: If file cannot be read after all retries
    """
    @retry_on_failure(max_retries=max_retries, delay=0.5, exceptions=(IOError, OSError))
    def _read():
        with open(file_path, 'r') as f:
            return f.read()
    
    try:
        return _read()
    except Exception as e:
        raise FileReadError(f'Failed to read file: {file_path}', {'error': str(e)})


def safe_file_write(file_path: str, content: Any, max_retries: int = 3):
    """
    Safely write to a file with automatic retry on failure.
    
    Args:
        file_path: Path to file to write
        content: Content to write
        max_retries: Maximum number of retry attempts
        
    Raises:
        FileWriteError: If file cannot be written after all retries
    """
    from .exceptions import FileWriteError
    
    @retry_on_failure(max_retries=max_retries, delay=0.5, exceptions=(IOError, OSError))
    def _write():
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
    
    try:
        _write()
    except Exception as e:
        raise FileWriteError(f'Failed to write file: {file_path}', {'error': str(e)})
