"""
Experiment Tracker Module

Tracks experiments, logs configurations and metrics, maintains leaderboard.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ExperimentInfo:
    """Information about a single experiment."""
    experiment_id: str
    name: Optional[str]
    config: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str
    status: str  # 'running', 'completed', 'failed'
    artifacts_path: str


class ExperimentTracker:
    """
    Tracks experiments, logs configurations and metrics, maintains leaderboard.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
    """
    
    def __init__(self, experiments_dir: str = './experiments'):
        """
        Initialize tracker with experiments directory.
        
        Args:
            experiments_dir: Root directory for all experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard_path = self.experiments_dir / 'leaderboard.json'
        self.experiments_cache: Dict[str, ExperimentInfo] = {}
        
        # Load existing leaderboard if it exists
        if self.leaderboard_path.exists():
            self._load_leaderboard()
        
    def create_experiment(self, name: Optional[str] = None) -> str:
        """
        Create new experiment with unique ID.
        
        Args:
            name: Optional human-readable name
            
        Returns:
            Experiment ID (timestamp-based)
        """
        # Generate unique timestamp-based ID
        timestamp = datetime.now()
        experiment_id = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Ensure uniqueness by adding microseconds if needed
        base_id = experiment_id
        counter = 0
        while (self.experiments_dir / experiment_id).exists():
            counter += 1
            experiment_id = f"{base_id}_{counter}"
        
        # Create experiment directory structure
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for artifacts
        (exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (exp_dir / 'plots').mkdir(exist_ok=True)
        (exp_dir / 'logs').mkdir(exist_ok=True)
        (exp_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Initialize experiment info
        exp_info = ExperimentInfo(
            experiment_id=experiment_id,
            name=name,
            config={},
            metrics={},
            timestamp=timestamp.isoformat(),
            status='running',
            artifacts_path=str(exp_dir)
        )
        
        # Save experiment info
        self._save_experiment_info(exp_info)
        self.experiments_cache[experiment_id] = exp_info
        
        return experiment_id
        
    def log_config(self, experiment_id: str, config: Dict[str, Any]):
        """
        Log experiment configuration.
        
        Args:
            experiment_id: Experiment identifier
            config: Configuration dictionary
        """
        exp_info = self._get_experiment_info(experiment_id)
        exp_info.config = config
        
        # Save config to separate file for easy access
        exp_dir = self.experiments_dir / experiment_id
        config_path = exp_dir / 'config.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update experiment info
        self._save_experiment_info(exp_info)
        
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], 
                   step: Optional[int] = None):
        """
        Log experiment metrics.
        
        Args:
            experiment_id: Experiment identifier
            metrics: Metrics dictionary
            step: Optional step/epoch number
        """
        exp_dir = self.experiments_dir / experiment_id
        metrics_path = exp_dir / 'metrics.jsonl'
        
        # Append metrics to JSONL file (one JSON object per line)
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'metrics': metrics
        }
        
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')
        
        # Update latest metrics in experiment info
        exp_info = self._get_experiment_info(experiment_id)
        exp_info.metrics.update(metrics)
        self._save_experiment_info(exp_info)
        
    def update_leaderboard(self, experiment_id: str, val_iou: float):
        """
        Update leaderboard with experiment results.
        
        Args:
            experiment_id: Experiment identifier
            val_iou: Validation Mean IoU score
        """
        exp_info = self._get_experiment_info(experiment_id)
        exp_info.metrics['val_iou'] = val_iou
        
        # Mark as candidate if IoU > 0.70
        if val_iou > 0.70:
            exp_info.metrics['is_candidate'] = True
        
        self._save_experiment_info(exp_info)
        self._save_leaderboard()
        
    def get_leaderboard(self, top_k: int = 10) -> List[ExperimentInfo]:
        """
        Get top experiments from leaderboard.
        
        Args:
            top_k: Number of top experiments to return
            
        Returns:
            List of ExperimentInfo sorted by validation IoU
        """
        # Load all experiments
        all_experiments = []
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / 'experiment_info.json').exists():
                exp_info = self._load_experiment_info(exp_dir.name)
                if 'val_iou' in exp_info.metrics:
                    all_experiments.append(exp_info)
        
        # Sort by validation IoU (descending)
        all_experiments.sort(key=lambda x: x.metrics.get('val_iou', 0.0), reverse=True)
        
        return all_experiments[:top_k]
        
    def save_artifacts(self, experiment_id: str, artifacts: Dict[str, str]):
        """
        Save experiment artifacts (checkpoints, plots, etc.).
        
        Args:
            experiment_id: Experiment identifier
            artifacts: Dictionary mapping artifact names to file paths
        """
        exp_dir = self.experiments_dir / experiment_id
        artifacts_path = exp_dir / 'artifacts.json'
        
        # Load existing artifacts if any
        existing_artifacts = {}
        if artifacts_path.exists():
            with open(artifacts_path, 'r') as f:
                existing_artifacts = json.load(f)
        
        # Update with new artifacts
        existing_artifacts.update(artifacts)
        
        # Save artifacts manifest
        with open(artifacts_path, 'w') as f:
            json.dump(existing_artifacts, f, indent=2)
        
    def load_experiment(self, experiment_id: str) -> ExperimentInfo:
        """
        Load experiment information and configuration.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            ExperimentInfo object
        """
        return self._load_experiment_info(experiment_id)
    
    def update_status(self, experiment_id: str, status: str):
        """
        Update experiment status.
        
        Args:
            experiment_id: Experiment identifier
            status: New status ('running', 'completed', 'failed')
        """
        exp_info = self._get_experiment_info(experiment_id)
        exp_info.status = status
        self._save_experiment_info(exp_info)
        
        if status in ['completed', 'failed']:
            self._save_leaderboard()
    
    def _get_experiment_info(self, experiment_id: str) -> ExperimentInfo:
        """Get experiment info from cache or load from disk."""
        if experiment_id in self.experiments_cache:
            return self.experiments_cache[experiment_id]
        return self._load_experiment_info(experiment_id)
    
    def _load_experiment_info(self, experiment_id: str) -> ExperimentInfo:
        """Load experiment info from disk."""
        exp_dir = self.experiments_dir / experiment_id
        info_path = exp_dir / 'experiment_info.json'
        
        if not info_path.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with open(info_path, 'r') as f:
            data = json.load(f)
        
        exp_info = ExperimentInfo(**data)
        self.experiments_cache[experiment_id] = exp_info
        return exp_info
    
    def _save_experiment_info(self, exp_info: ExperimentInfo):
        """Save experiment info to disk."""
        exp_dir = self.experiments_dir / exp_info.experiment_id
        info_path = exp_dir / 'experiment_info.json'
        
        with open(info_path, 'w') as f:
            json.dump(asdict(exp_info), f, indent=2)
        
        self.experiments_cache[exp_info.experiment_id] = exp_info
    
    def _load_leaderboard(self):
        """Load leaderboard from disk."""
        if self.leaderboard_path.exists():
            with open(self.leaderboard_path, 'r') as f:
                data = json.load(f)
                for exp_data in data:
                    exp_info = ExperimentInfo(**exp_data)
                    self.experiments_cache[exp_info.experiment_id] = exp_info
    
    def _save_leaderboard(self):
        """Save leaderboard to disk."""
        leaderboard = self.get_leaderboard(top_k=100)  # Save top 100
        
        with open(self.leaderboard_path, 'w') as f:
            json.dump([asdict(exp) for exp in leaderboard], f, indent=2)
