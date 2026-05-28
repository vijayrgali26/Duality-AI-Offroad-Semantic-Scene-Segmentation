"""
Logging utilities for the IoU Improvement Pipeline.

This module provides centralized logging configuration with file and console handlers,
appropriate formatters, and log level management.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    experiment_id: Optional[str] = None,
    console_level: Optional[int] = None
) -> logging.Logger:
    """
    Setup logging infrastructure with file and console handlers.
    
    Creates a logger with both file and console output. File logs are saved to
    the specified directory with timestamps. Console logs can have a different
    level than file logs for better user experience.
    
    Args:
        log_dir: Directory to save log files. If None, uses './logs'
        log_level: Logging level for file handler (default: INFO)
        experiment_id: Optional experiment ID to include in log filename
        console_level: Logging level for console handler. If None, uses log_level
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging(log_dir='./experiments/exp_001/logs', 
        ...                        log_level=logging.DEBUG,
        ...                        experiment_id='exp_001')
        >>> logger.info('Pipeline started')
    """
    # Create logger
    logger = logging.getLogger('iou_pipeline')
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create log directory if specified
    if log_dir is None:
        log_dir = './logs'
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_id:
        log_filename = f'{experiment_id}_{timestamp}.log'
    else:
        log_filename = f'pipeline_{timestamp}.log'
    
    file_handler = logging.FileHandler(log_path / log_filename, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level if console_level is not None else log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f'Logging initialized. Log file: {log_path / log_filename}')
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name. If None, returns the root pipeline logger
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger('dataset_analyzer')
        >>> logger.info('Starting analysis')
    """
    if name:
        return logging.getLogger(f'iou_pipeline.{name}')
    return logging.getLogger('iou_pipeline')


def log_section(logger: logging.Logger, title: str, level: int = logging.INFO):
    """
    Log a section header for better readability.
    
    Args:
        logger: Logger instance
        title: Section title
        level: Logging level (default: INFO)
        
    Example:
        >>> logger = get_logger()
        >>> log_section(logger, 'Dataset Analysis Phase')
    """
    separator = '=' * 80
    logger.log(level, separator)
    logger.log(level, f' {title}')
    logger.log(level, separator)


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ''):
    """
    Log metrics in a formatted way.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for metric names
        
    Example:
        >>> logger = get_logger()
        >>> metrics = {'mean_iou': 0.75, 'pixel_accuracy': 0.89}
        >>> log_metrics(logger, metrics, prefix='Validation')
    """
    if prefix:
        logger.info(f'{prefix} Metrics:')
    else:
        logger.info('Metrics:')
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f'  {key}: {value:.4f}')
        else:
            logger.info(f'  {key}: {value}')


def log_config(logger: logging.Logger, config: dict, title: str = 'Configuration'):
    """
    Log configuration parameters in a formatted way.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        title: Title for the configuration section
        
    Example:
        >>> logger = get_logger()
        >>> config = {'batch_size': 8, 'learning_rate': 0.0001}
        >>> log_config(logger, config, title='Training Configuration')
    """
    log_section(logger, title)
    for key, value in config.items():
        logger.info(f'  {key}: {value}')
