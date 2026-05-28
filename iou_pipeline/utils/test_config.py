"""Unit tests for configuration management system.

Tests cover:
- TrainingConfig and PipelineConfig dataclass creation
- YAML and JSON loading/saving
- Configuration validation with range checks
- Configuration merging for inheritance
"""

import json
import yaml
import pytest
import tempfile
from pathlib import Path
from iou_pipeline.utils.config import (
    TrainingConfig,
    PipelineConfig,
    ConfigManager,
    ConfigurationError
)


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_default_initialization(self):
        """Test TrainingConfig with default values."""
        config = TrainingConfig()
        
        assert config.backbone == 'dinov2_vits14'
        assert config.num_classes == 11
        assert config.batch_size == 8
        assert config.num_epochs == 100
        assert config.learning_rate == 0.0001
        assert config.optimizer == 'sgd'
        assert config.use_amp is True
        assert config.early_stopping_patience == 15
    
    def test_custom_initialization(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            batch_size=16,
            learning_rate=0.0005,
            num_epochs=150,
            optimizer='adamw'
        )
        
        assert config.batch_size == 16
        assert config.learning_rate == 0.0005
        assert config.num_epochs == 150
        assert config.optimizer == 'adamw'
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig(batch_size=16, learning_rate=0.0005)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['batch_size'] == 16
        assert config_dict['learning_rate'] == 0.0005
        assert 'backbone' in config_dict
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'batch_size': 32,
            'learning_rate': 0.0002,
            'num_epochs': 80,
            'optimizer': 'adamw'
        }
        config = TrainingConfig.from_dict(config_dict)
        
        assert config.batch_size == 32
        assert config.learning_rate == 0.0002
        assert config.num_epochs == 80
        assert config.optimizer == 'adamw'


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""
    
    def test_default_initialization(self):
        """Test PipelineConfig with default values."""
        config = PipelineConfig()
        
        assert config.dataset_path == ""
        assert config.output_dir == "./experiments"
        assert config.run_analysis is True
        assert config.run_augmentation is True
        assert config.min_balance_ratio == 0.5
        assert config.poor_iou_threshold == 0.4
        assert config.target_dataset_size == 5000
        assert isinstance(config.training_config, TrainingConfig)
    
    def test_custom_initialization(self):
        """Test PipelineConfig with custom values."""
        training_config = TrainingConfig(batch_size=16)
        config = PipelineConfig(
            dataset_path="/data/dataset",
            output_dir="/output",
            min_balance_ratio=0.3,
            training_config=training_config
        )
        
        assert config.dataset_path == "/data/dataset"
        assert config.output_dir == "/output"
        assert config.min_balance_ratio == 0.3
        assert config.training_config.batch_size == 16
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PipelineConfig(dataset_path="/data/test")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['dataset_path'] == "/data/test"
        assert 'training_config' in config_dict
        assert isinstance(config_dict['training_config'], dict)


class TestConfigYAMLOperations:
    """Test YAML loading and saving operations."""
    
    def test_save_and_load_yaml(self):
        """Test saving and loading PipelineConfig to/from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            
            # Create and save config
            original_config = PipelineConfig(
                dataset_path="/data/test",
                output_dir="/output/test",
                min_balance_ratio=0.6
            )
            original_config.training_config.batch_size = 16
            original_config.training_config.learning_rate = 0.0005
            
            original_config.save_yaml(str(yaml_path))
            
            # Load config
            loaded_config = PipelineConfig.from_yaml(str(yaml_path))
            
            assert loaded_config.dataset_path == original_config.dataset_path
            assert loaded_config.output_dir == original_config.output_dir
            assert loaded_config.min_balance_ratio == original_config.min_balance_ratio
            assert loaded_config.training_config.batch_size == 16
            assert loaded_config.training_config.learning_rate == 0.0005
    
    def test_load_yaml_file_not_found(self):
        """Test loading from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/config.yaml")
    
    def test_load_yaml_with_nested_training_config(self):
        """Test loading YAML with nested training_config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            
            # Create YAML with nested structure
            config_data = {
                'dataset_path': '/data/test',
                'output_dir': '/output',
                'training_config': {
                    'batch_size': 32,
                    'learning_rate': 0.0003,
                    'num_epochs': 120
                }
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load config
            config = PipelineConfig.from_yaml(str(yaml_path))
            
            assert config.training_config.batch_size == 32
            assert config.training_config.learning_rate == 0.0003
            assert config.training_config.num_epochs == 120


class TestConfigJSONOperations:
    """Test JSON loading and saving operations."""
    
    def test_save_and_load_json(self):
        """Test saving and loading PipelineConfig to/from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            
            # Create and save config
            original_config = PipelineConfig(
                dataset_path="/data/test",
                output_dir="/output/test",
                poor_iou_threshold=0.35
            )
            original_config.training_config.batch_size = 4
            original_config.training_config.optimizer = 'adamw'
            
            original_config.save_json(str(json_path))
            
            # Load config
            loaded_config = PipelineConfig.from_json(str(json_path))
            
            assert loaded_config.dataset_path == original_config.dataset_path
            assert loaded_config.output_dir == original_config.output_dir
            assert loaded_config.poor_iou_threshold == 0.35
            assert loaded_config.training_config.batch_size == 4
            assert loaded_config.training_config.optimizer == 'adamw'
    
    def test_load_json_file_not_found(self):
        """Test loading from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_json("/nonexistent/config.json")
    
    def test_load_json_with_nested_training_config(self):
        """Test loading JSON with nested training_config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            
            # Create JSON with nested structure
            config_data = {
                'dataset_path': '/data/test',
                'output_dir': '/output',
                'training_config': {
                    'batch_size': 16,
                    'learning_rate': 0.00025,
                    'scheduler': 'step'
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(config_data, f)
            
            # Load config
            config = PipelineConfig.from_json(str(json_path))
            
            assert config.training_config.batch_size == 16
            assert config.training_config.learning_rate == 0.00025
            assert config.training_config.scheduler == 'step'


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_load_config_yaml(self):
        """Test loading config from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            
            # Create a valid config file
            config_data = {
                'dataset_path': str(tmpdir),  # Use tmpdir as valid path
                'output_dir': '/output',
                'training_config': {
                    'batch_size': 8,
                    'learning_rate': 0.0001,
                    'num_epochs': 100
                }
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load using ConfigManager
            config = ConfigManager.load_config(str(yaml_path))
            
            assert isinstance(config, PipelineConfig)
            assert config.training_config.batch_size == 8
    
    def test_load_config_json(self):
        """Test loading config from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            
            # Create a valid config file
            config_data = {
                'dataset_path': str(tmpdir),  # Use tmpdir as valid path
                'output_dir': '/output',
                'training_config': {
                    'batch_size': 16,
                    'learning_rate': 0.0002,
                    'num_epochs': 80
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(config_data, f)
            
            # Load using ConfigManager
            config = ConfigManager.load_config(str(json_path))
            
            assert isinstance(config, PipelineConfig)
            assert config.training_config.batch_size == 16
    
    def test_load_config_unsupported_format(self):
        """Test loading config with unsupported format."""
        with pytest.raises(ConfigurationError, match="Unsupported config format"):
            ConfigManager.load_config("config.txt")
    
    def test_load_config_validation_failure(self):
        """Test loading config that fails validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            
            # Create invalid config (learning rate out of range)
            config_data = {
                'dataset_path': str(tmpdir),
                'training_config': {
                    'learning_rate': 0.01,  # Out of range
                    'num_epochs': 100
                }
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Should raise ConfigurationError
            with pytest.raises(ConfigurationError, match="validation failed"):
                ConfigManager.load_config(str(yaml_path))


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                dataset_path=str(tmpdir),
                output_dir="/output"
            )
            
            errors = ConfigManager.validate_config(config)
            assert len(errors) == 0
    
    def test_validate_missing_dataset_path(self):
        """Test validation with missing dataset path."""
        config = PipelineConfig(dataset_path="")
        
        errors = ConfigManager.validate_config(config)
        assert any("Dataset path is required" in err for err in errors)
    
    def test_validate_nonexistent_dataset_path(self):
        """Test validation with non-existent dataset path."""
        config = PipelineConfig(dataset_path="/nonexistent/path")
        
        errors = ConfigManager.validate_config(config)
        assert any("Dataset path does not exist" in err for err in errors)
    
    def test_validate_learning_rate_out_of_range(self):
        """Test validation with learning rate out of range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.learning_rate = 0.01  # Too high
            
            errors = ConfigManager.validate_config(config)
            assert any("Learning rate out of range" in err for err in errors)
            
            config.training_config.learning_rate = 0.000001  # Too low
            errors = ConfigManager.validate_config(config)
            assert any("Learning rate out of range" in err for err in errors)
    
    def test_validate_batch_size_invalid(self):
        """Test validation with invalid batch size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.batch_size = 12  # Not in [4, 8, 16, 32]
            
            errors = ConfigManager.validate_config(config)
            assert any("Batch size must be 4, 8, 16, or 32" in err for err in errors)
    
    def test_validate_num_epochs_out_of_range(self):
        """Test validation with num_epochs out of range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.num_epochs = 30  # Too low
            
            errors = ConfigManager.validate_config(config)
            assert any("Num epochs out of range" in err for err in errors)
            
            config.training_config.num_epochs = 250  # Too high
            errors = ConfigManager.validate_config(config)
            assert any("Num epochs out of range" in err for err in errors)
    
    def test_validate_invalid_optimizer(self):
        """Test validation with invalid optimizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.optimizer = 'adam'  # Not 'sgd' or 'adamw'
            
            errors = ConfigManager.validate_config(config)
            assert any("Optimizer must be 'sgd' or 'adamw'" in err for err in errors)
    
    def test_validate_invalid_scheduler(self):
        """Test validation with invalid scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.scheduler = 'exponential'
            
            errors = ConfigManager.validate_config(config)
            assert any("Scheduler must be 'cosine', 'step', or 'plateau'" in err for err in errors)
    
    def test_validate_label_smoothing_out_of_range(self):
        """Test validation with label smoothing out of range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.label_smoothing = 0.5  # Too high
            
            errors = ConfigManager.validate_config(config)
            assert any("Label smoothing out of range" in err for err in errors)
    
    def test_validate_min_balance_ratio_out_of_range(self):
        """Test validation with min_balance_ratio out of range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                dataset_path=str(tmpdir),
                min_balance_ratio=1.5
            )
            
            errors = ConfigManager.validate_config(config)
            assert any("Min balance ratio out of range" in err for err in errors)
    
    def test_validate_poor_iou_threshold_out_of_range(self):
        """Test validation with poor_iou_threshold out of range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                dataset_path=str(tmpdir),
                poor_iou_threshold=1.2
            )
            
            errors = ConfigManager.validate_config(config)
            assert any("Poor IoU threshold out of range" in err for err in errors)


class TestConfigMerging:
    """Test configuration merging for inheritance."""
    
    def test_merge_simple_configs(self):
        """Test merging simple configurations."""
        base = {
            'dataset_path': '/data/base',
            'output_dir': '/output/base',
            'run_analysis': True
        }
        
        override = {
            'output_dir': '/output/override',
            'run_augmentation': False
        }
        
        merged = ConfigManager.merge_configs(base, override)
        
        assert merged['dataset_path'] == '/data/base'  # From base
        assert merged['output_dir'] == '/output/override'  # Overridden
        assert merged['run_analysis'] is True  # From base
        assert merged['run_augmentation'] is False  # From override
    
    def test_merge_nested_configs(self):
        """Test merging nested configurations."""
        base = {
            'dataset_path': '/data/base',
            'training_config': {
                'batch_size': 8,
                'learning_rate': 0.0001,
                'num_epochs': 100
            }
        }
        
        override = {
            'training_config': {
                'batch_size': 16,
                'optimizer': 'adamw'
            }
        }
        
        merged = ConfigManager.merge_configs(base, override)
        
        assert merged['dataset_path'] == '/data/base'
        assert merged['training_config']['batch_size'] == 16  # Overridden
        assert merged['training_config']['learning_rate'] == 0.0001  # From base
        assert merged['training_config']['num_epochs'] == 100  # From base
        assert merged['training_config']['optimizer'] == 'adamw'  # From override
    
    def test_merge_deep_nested_configs(self):
        """Test merging deeply nested configurations."""
        base = {
            'level1': {
                'level2': {
                    'value1': 'base1',
                    'value2': 'base2'
                },
                'other': 'base_other'
            }
        }
        
        override = {
            'level1': {
                'level2': {
                    'value1': 'override1'
                }
            }
        }
        
        merged = ConfigManager.merge_configs(base, override)
        
        assert merged['level1']['level2']['value1'] == 'override1'
        assert merged['level1']['level2']['value2'] == 'base2'
        assert merged['level1']['other'] == 'base_other'
    
    def test_create_config_from_inheritance(self):
        """Test creating config from base and override files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.yaml"
            override_path = Path(tmpdir) / "override.yaml"
            
            # Create base config
            base_data = {
                'dataset_path': str(tmpdir),
                'output_dir': '/output/base',
                'training_config': {
                    'batch_size': 8,
                    'learning_rate': 0.0001,
                    'num_epochs': 100
                }
            }
            
            with open(base_path, 'w') as f:
                yaml.dump(base_data, f)
            
            # Create override config
            override_data = {
                'output_dir': '/output/override',
                'training_config': {
                    'batch_size': 16,
                    'optimizer': 'adamw'
                }
            }
            
            with open(override_path, 'w') as f:
                yaml.dump(override_data, f)
            
            # Merge configs
            config = ConfigManager.create_config_from_inheritance(
                str(base_path),
                str(override_path)
            )
            
            assert config.output_dir == '/output/override'
            assert config.training_config.batch_size == 16
            assert config.training_config.learning_rate == 0.0001
            assert config.training_config.num_epochs == 100
            assert config.training_config.optimizer == 'adamw'


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_scales_list(self):
        """Test validation with empty scales list for multiscale training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.use_multiscale = True
            config.training_config.scales = []
            
            errors = ConfigManager.validate_config(config)
            assert any("no scales provided" in err for err in errors)
    
    def test_negative_scale_values(self):
        """Test validation with negative scale values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.use_multiscale = True
            config.training_config.scales = [0.5, -1.0, 1.5]
            
            errors = ConfigManager.validate_config(config)
            assert any("All scales must be positive" in err for err in errors)
    
    def test_boundary_learning_rate_values(self):
        """Test validation with boundary learning rate values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test minimum valid value
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.learning_rate = 0.00001
            errors = ConfigManager.validate_config(config)
            assert len(errors) == 0
            
            # Test maximum valid value
            config.training_config.learning_rate = 0.001
            errors = ConfigManager.validate_config(config)
            assert len(errors) == 0
    
    def test_boundary_num_epochs_values(self):
        """Test validation with boundary num_epochs values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test minimum valid value
            config = PipelineConfig(dataset_path=str(tmpdir))
            config.training_config.num_epochs = 50
            errors = ConfigManager.validate_config(config)
            assert len(errors) == 0
            
            # Test maximum valid value
            config.training_config.num_epochs = 200
            errors = ConfigManager.validate_config(config)
            assert len(errors) == 0
