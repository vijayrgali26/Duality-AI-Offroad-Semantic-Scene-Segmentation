"""
IoU Improvement Pipeline

A comprehensive system for enhancing semantic segmentation performance through
dataset analysis, quality improvement, balanced augmentation, and optimized training.
"""

__version__ = "0.1.0"

from .analyzer import DatasetAnalyzer
from .editor import DatasetEditor
from .trainer import TrainingOrchestrator
from .evaluator import EvaluationEngine
from .tracker import ExperimentTracker
from .pipeline import IoUPipeline

__all__ = [
    "DatasetAnalyzer",
    "DatasetEditor",
    "TrainingOrchestrator",
    "EvaluationEngine",
    "ExperimentTracker",
    "IoUPipeline",
]
