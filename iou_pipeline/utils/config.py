"""Configuration management for IoU Improvement Pipeline.

This module provides dataclasses and utilities for managing pipeline configuration,
including training hyperparameters, dataset settings, and pipeline control options.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import ConfigurationError from exceptions module to maintain consistency
try:
    from .exceptions import ConfigurationError
except ImportError:
    # Fallback if exceptions module is not available
    class ConfigurationError(Exception):
        """Exception raised for configuration validation errors."""
        pass


@dataclass
class TrainingConfig:
    """Training configuration with hyperparameters and advanced options."""
    
    # Model
    backbone: str = 'dinov2_vits14'  # or 'dinov2_vitb14_reg', 'dinov2_vitl14_reg'
    num_classes: int = 11
    
    # Training
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 0.0001
    optimizer: str = 'sgd'  # or 'adamw'
    momentum: float = 0.9
    weight_decay: float = 0.0001
    
    # Scheduler
    scheduler: str = 'cosine'  # or 'step', 'plateau'
    t_max: int = 100  # for cosine
    eta_min: float = 1e-6
    
    # Loss
    use_class_weights: bool = True
    label_smoothing: float = 0.1
    
    # Advanced
    use_amp: bool = True  # mixed precision
    early_stopping_patience: int = 15
    gradient_clip: float = 1.0
    
    # Hard Example Mining
    use_hard_example_mining: bool = False
    hem_ratio: float = 0.3  # top 30% hardest samples
    
    # Multi-scale Training
    use_multiscale: bool = False
    scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    
    # Deep Supervision
    use_deep_supervision: bool = False
    aux_loss_weight: float = 0.4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    # Paths
    dataset_path: str = ""
    output_dir: str = "./experiments"
    experiment_name: Optional[str] = None
    
    # Analysis settings
    run_analysis: bool = True
    baseline_model_path: Optional[str] = None
    
    # Dataset editing settings
    run_augmentation: bool = True
    min_balance_ratio: float = 0.5
    poor_iou_threshold: float = 0.4
    target_dataset_size: int = 5000
    
    # Training settings
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Evaluation settings
    use_tta: bool = False
    num_visualization_samples: int = 20
    
    # Pipeline control
    dry_run: bool = False
    resume_from_checkpoint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        config_dict = asdict(self)
        # Ensure training_config is properly serialized
        if isinstance(self.training_config, TrainingConfig):
            config_dict['training_config'] = self.training_config.to_dict()
        return config_dict
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            PipelineConfig object
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        yaml_path_obj = Path(yaml_path)
        if not yaml_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested training_config
        if 'training_config' in config_dict and isinstance(config_dict['training_config'], dict):
            config_dict['training_config'] = TrainingConfig.from_dict(config_dict['training_config'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            PipelineConfig object
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON parsing fails
        """
        json_path_obj = Path(json_path)
        if not json_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        # Handle nested training_config
        if 'training_config' in config_dict and isinstance(config_dict['training_config'], dict):
            config_dict['training_config'] = TrainingConfig.from_dict(config_dict['training_config'])
        
        return cls(**config_dict)
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def save_json(self, json_path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration
        """
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ConfigManager:
    """Manage configuration loading, validation, and merging."""
    
    @staticmethod
    def load_config(config_path: str) -> PipelineConfig:
        """Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file (.yaml, .yml, or .json)
            
        Returns:
            Validated PipelineConfig object
            
        Raises:
            ConfigurationError: If file format is unsupported or validation fails
            FileNotFoundError: If configuration file doesn't exist
        """
        config_path_lower = config_path.lower()
        
        if config_path_lower.endswith('.yaml') or config_path_lower.endswith('.yml'):
            config = PipelineConfig.from_yaml(config_path)
        elif config_path_lower.endswith('.json'):
            config = PipelineConfig.from_json(config_path)
        else:
            raise ConfigurationError(
                f"Unsupported config format: {config_path}. "
                "Supported formats: .yaml, .yml, .json"
            )
        
        # Validate the loaded configuration
        errors = ConfigManager.validate_config(config)
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ConfigurationError(error_msg)
        
        return config
    
    @staticmethod
    def validate_config(config: PipelineConfig) -> List[str]:
        """Validate configuration and return list of issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate paths
        if config.dataset_path and not Path(config.dataset_path).exists():
            errors.append(f"Dataset path does not exist: {config.dataset_path}")
        
        if not config.dataset_path:
            errors.append("Dataset path is required")
        
        # Validate training hyperparameters
        tc = config.training_config
        
        # Learning rate range: 0.00001 to 0.001 (Requirement 3.1)
        if not (0.00001 <= tc.learning_rate <= 0.001):
            errors.append(
                f"Learning rate out of range [0.00001, 0.001]: {tc.learning_rate}"
            )
        
        # Batch size must be 4, 8, 16, or 32 (Requirement 3.2)
        if tc.batch_size not in [4, 8, 16, 32]:
            errors.append(
                f"Batch size must be 4, 8, 16, or 32: {tc.batch_size}"
            )
        
        # Num epochs range: 50 to 200 (Requirement 3.6)
        if not (50 <= tc.num_epochs <= 200):
            errors.append(
                f"Num epochs out of range [50, 200]: {tc.num_epochs}"
            )
        
        # Validate optimizer choice
        if tc.optimizer not in ['sgd', 'adamw']:
            errors.append(
                f"Optimizer must be 'sgd' or 'adamw': {tc.optimizer}"
            )
        
        # Validate scheduler choice
        if tc.scheduler not in ['cosine', 'step', 'plateau']:
            errors.append(
                f"Scheduler must be 'cosine', 'step', or 'plateau': {tc.scheduler}"
            )
        
        # Validate momentum (for SGD)
        if tc.optimizer == 'sgd' and not (0.0 <= tc.momentum <= 1.0):
            errors.append(
                f"Momentum out of range [0.0, 1.0]: {tc.momentum}"
            )
        
        # Validate weight decay
        if not (0.0 <= tc.weight_decay <= 0.01):
            errors.append(
                f"Weight decay out of range [0.0, 0.01]: {tc.weight_decay}"
            )
        
        # Validate label smoothing
        if not (0.0 <= tc.label_smoothing <= 0.2):
            errors.append(
                f"Label smoothing out of range [0.0, 0.2]: {tc.label_smoothing}"
            )
        
        # Validate early stopping patience
        if tc.early_stopping_patience < 1:
            errors.append(
                f"Early stopping patience must be >= 1: {tc.early_stopping_patience}"
            )
        
        # Validate gradient clip
        if tc.gradient_clip <= 0.0:
            errors.append(
                f"Gradient clip must be > 0.0: {tc.gradient_clip}"
            )
        
        # Validate hard example mining ratio
        if tc.use_hard_example_mining and not (0.0 < tc.hem_ratio <= 1.0):
            errors.append(
                f"Hard example mining ratio out of range (0.0, 1.0]: {tc.hem_ratio}"
            )
        
        # Validate multi-scale training scales
        if tc.use_multiscale:
            if not tc.scales:
                errors.append("Multi-scale training enabled but no scales provided")
            elif any(scale <= 0.0 for scale in tc.scales):
                errors.append("All scales must be positive")
        
        # Validate deep supervision aux loss weight
        if tc.use_deep_supervision and not (0.0 <= tc.aux_loss_weight <= 1.0):
            errors.append(
                f"Auxiliary loss weight out of range [0.0, 1.0]: {tc.aux_loss_weight}"
            )
        
        # Validate augmentation settings
        if not (0.0 <= config.min_balance_ratio <= 1.0):
            errors.append(
                f"Min balance ratio out of range [0.0, 1.0]: {config.min_balance_ratio}"
            )
        
        if not (0.0 <= config.poor_iou_threshold <= 1.0):
            errors.append(
                f"Poor IoU threshold out of range [0.0, 1.0]: {config.poor_iou_threshold}"
            )
        
        if config.target_dataset_size < 1:
            errors.append(
                f"Target dataset size must be >= 1: {config.target_dataset_size}"
            )
        
        # Validate evaluation settings
        if config.num_visualization_samples < 0:
            errors.append(
                f"Num visualization samples must be >= 0: {config.num_visualization_samples}"
            )
        
        # Validate checkpoint path if resuming
        if config.resume_from_checkpoint and not Path(config.resume_from_checkpoint).exists():
            errors.append(
                f"Resume checkpoint does not exist: {config.resume_from_checkpoint}"
            )
        
        return errors
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge override configuration into base configuration recursively.
        
        This enables configuration inheritance where a child config can override
        specific fields from a parent config while keeping other fields unchanged.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
            
        Example:
            >>> base = {'training_config': {'batch_size': 8, 'lr': 0.0001}}
            >>> override = {'training_config': {'batch_size': 16}}
            >>> merged = ConfigManager.merge_configs(base, override)
            >>> merged['training_config']['batch_size']
            16
            >>> merged['training_config']['lr']
            0.0001
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Recursively merge nested dictionaries
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                # Override value
                merged[key] = value
        
        return merged
    
    @staticmethod
    def create_config_from_inheritance(base_path: str, override_path: str) -> PipelineConfig:
        """Create configuration by merging base and override configs.
        
        Args:
            base_path: Path to base configuration file
            override_path: Path to override configuration file
            
        Returns:
            Merged PipelineConfig object
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Load base config as dict
        if base_path.lower().endswith(('.yaml', '.yml')):
            with open(base_path, 'r') as f:
                base_dict = yaml.safe_load(f)
        else:
            with open(base_path, 'r') as f:
                base_dict = json.load(f)
        
        # Load override config as dict
        if override_path.lower().endswith(('.yaml', '.yml')):
            with open(override_path, 'r') as f:
                override_dict = yaml.safe_load(f)
        else:
            with open(override_path, 'r') as f:
                override_dict = json.load(f)
        
        # Merge configurations
        merged_dict = ConfigManager.merge_configs(base_dict, override_dict)
        
        # Convert to PipelineConfig
        if 'training_config' in merged_dict and isinstance(merged_dict['training_config'], dict):
            merged_dict['training_config'] = TrainingConfig.from_dict(merged_dict['training_config'])
        
        config = PipelineConfig(**merged_dict)
        
        # Validate merged config
        errors = ConfigManager.validate_config(config)
        if errors:
            error_msg = "Merged configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ConfigurationError(error_msg)
        
        return config
