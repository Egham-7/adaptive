"""Configuration system for routing agent parameters and thresholds."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import ValidationError

from .models import RoutingConfig, TaskType


class ConfigManager:
    """Manages routing agent configuration with file persistence."""
    
    DEFAULT_CONFIG_PATH = "routing_config.json"
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else Path(self.DEFAULT_CONFIG_PATH)
        self._config: Optional[RoutingConfig] = None
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> RoutingConfig:
        """Load configuration from file or create default.
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            RoutingConfig instance
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Convert string keys back to enums for task_benchmark_weights
                if 'task_benchmark_weights' in config_data:
                    converted_weights = {}
                    for task_str, weights in config_data['task_benchmark_weights'].items():
                        try:
                            task_enum = TaskType(task_str)
                            converted_weights[task_enum] = weights
                        except ValueError:
                            # Skip invalid task types
                            continue
                    config_data['task_benchmark_weights'] = converted_weights
                
                # Convert string keys for min_performance_thresholds
                if 'min_performance_thresholds' in config_data:
                    converted_thresholds = {}
                    for task_str, threshold in config_data['min_performance_thresholds'].items():
                        try:
                            task_enum = TaskType(task_str)
                            converted_thresholds[task_enum] = threshold
                        except ValueError:
                            continue
                    config_data['min_performance_thresholds'] = converted_thresholds
                
                # Convert string keys for complexity_length_thresholds if present
                if 'complexity_length_thresholds' in config_data:
                    from .models import ComplexityLevel
                    converted_complexity = {}
                    for complexity_str, threshold in config_data['complexity_length_thresholds'].items():
                        try:
                            complexity_enum = ComplexityLevel(complexity_str)
                            converted_complexity[complexity_enum] = threshold
                        except ValueError:
                            continue
                    config_data['complexity_length_thresholds'] = converted_complexity
                
                self._config = RoutingConfig(**config_data)
                print(f"Loaded routing configuration from {self.config_path}")
                
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Error loading config from {self.config_path}: {e}")
                print("Using default configuration")
                self._config = RoutingConfig()
        else:
            print(f"No config file found at {self.config_path}, using defaults")
            self._config = RoutingConfig()
        
        return self._config
    
    def save_config(self, config: Optional[RoutingConfig] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save (uses current if None)
        """
        if config:
            self._config = config
        
        if not self._config:
            raise ValueError("No configuration to save")
        
        # Convert to dict and handle enum serialization
        config_dict = self._config.model_dump()
        
        # Convert enum keys to strings for JSON serialization
        if 'task_benchmark_weights' in config_dict:
            converted_weights = {}
            for task_enum, weights in config_dict['task_benchmark_weights'].items():
                task_str = task_enum.value if hasattr(task_enum, 'value') else str(task_enum)
                converted_weights[task_str] = weights
            config_dict['task_benchmark_weights'] = converted_weights
        
        if 'min_performance_thresholds' in config_dict:
            converted_thresholds = {}
            for task_enum, threshold in config_dict['min_performance_thresholds'].items():
                task_str = task_enum.value if hasattr(task_enum, 'value') else str(task_enum)
                converted_thresholds[task_str] = threshold
            config_dict['min_performance_thresholds'] = converted_thresholds
        
        if 'complexity_length_thresholds' in config_dict:
            converted_complexity = {}
            for complexity_enum, threshold in config_dict['complexity_length_thresholds'].items():
                complexity_str = complexity_enum.value if hasattr(complexity_enum, 'value') else str(complexity_enum)
                converted_complexity[complexity_str] = threshold
            config_dict['complexity_length_thresholds'] = converted_complexity
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file with pretty formatting
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Saved routing configuration to {self.config_path}")
    
    def get_config(self) -> RoutingConfig:
        """Get current configuration.
        
        Returns:
            Current RoutingConfig instance
        """
        if not self._config:
            return self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> RoutingConfig:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            Updated RoutingConfig instance
        """
        current_config = self.get_config()
        
        # Convert dict to new config
        config_dict = current_config.model_dump()
        config_dict.update(updates)
        
        try:
            self._config = RoutingConfig(**config_dict)
            return self._config
        except ValidationError as e:
            print(f"Configuration update failed: {e}")
            return current_config
    
    def reset_to_defaults(self) -> RoutingConfig:
        """Reset configuration to defaults.
        
        Returns:
            Default RoutingConfig instance
        """
        self._config = RoutingConfig()
        return self._config


class PresetConfigs:
    """Predefined configuration presets for different use cases."""
    
    @staticmethod
    def get_speed_optimized() -> RoutingConfig:
        """Configuration optimized for fastest routing decisions."""
        return RoutingConfig(
            performance_weight=0.3,
            efficiency_weight=0.5,  # Higher weight on efficiency
            context_weight=0.1,
            speed_weight=0.1,
            prefer_smaller_models_for_simple=True,
            max_parameters_for_simple=15.0,  # Prefer smaller models
            min_performance_thresholds={
                TaskType.MATH: 65.0,  # Slightly lower thresholds
                TaskType.CODING: 70.0,
                TaskType.REASONING: 75.0,
                TaskType.GENERAL_QA: 60.0,
            }
        )
    
    @staticmethod
    def get_quality_optimized() -> RoutingConfig:
        """Configuration optimized for highest quality results."""
        return RoutingConfig(
            performance_weight=0.6,  # Higher weight on performance
            efficiency_weight=0.2,
            context_weight=0.15,
            speed_weight=0.05,
            prefer_smaller_models_for_simple=False,  # Use powerful models even for simple tasks
            max_parameters_for_simple=100.0,
            min_performance_thresholds={
                TaskType.MATH: 80.0,  # Higher thresholds
                TaskType.CODING: 85.0,
                TaskType.REASONING: 85.0,
                TaskType.GENERAL_QA: 75.0,
            }
        )
    
    @staticmethod
    def get_balanced() -> RoutingConfig:
        """Balanced configuration for general use."""
        return RoutingConfig()  # Use defaults
    
    @staticmethod
    def get_cost_optimized() -> RoutingConfig:
        """Configuration optimized for cost efficiency."""
        return RoutingConfig(
            performance_weight=0.25,
            efficiency_weight=0.45,  # High efficiency weight
            context_weight=0.15,
            speed_weight=0.15,
            prefer_smaller_models_for_simple=True,
            max_parameters_for_simple=10.0,  # Prefer very small models
            fallback_to_efficient=True,
            min_performance_thresholds={
                TaskType.MATH: 60.0,  # Lower thresholds to allow smaller models
                TaskType.CODING: 65.0,
                TaskType.REASONING: 70.0,
                TaskType.GENERAL_QA: 55.0,
            }
        )
    
    @staticmethod
    def get_research_optimized() -> RoutingConfig:
        """Configuration optimized for research and academic tasks."""
        return RoutingConfig(
            performance_weight=0.5,
            efficiency_weight=0.2,
            context_weight=0.25,  # Higher context weight for long documents
            speed_weight=0.05,
            prefer_smaller_models_for_simple=False,
            task_benchmark_weights={
                TaskType.MATH: {"math": 0.6, "gsm8k": 0.2, "mmlu": 0.2},  # Strong math focus
                TaskType.CODING: {"humaneval": 0.7, "mmlu": 0.3},
                TaskType.REASONING: {"math": 0.3, "mmlu": 0.7},  # Strong reasoning focus
                TaskType.GENERAL_QA: {"mmlu": 0.9, "gsm8k": 0.1},
            },
            min_performance_thresholds={
                TaskType.MATH: 85.0,  # High standards for research
                TaskType.CODING: 80.0,
                TaskType.REASONING: 85.0,
                TaskType.GENERAL_QA: 80.0,
            },
            context_buffer_ratio=1.5  # Larger safety margin for context
        )


def create_config_from_preset(preset_name: str) -> RoutingConfig:
    """Create configuration from preset name.
    
    Args:
        preset_name: Name of preset (speed, quality, balanced, cost, research)
        
    Returns:
        RoutingConfig instance for the preset
    """
    presets = {
        "speed": PresetConfigs.get_speed_optimized,
        "quality": PresetConfigs.get_quality_optimized,
        "balanced": PresetConfigs.get_balanced,
        "cost": PresetConfigs.get_cost_optimized,
        "research": PresetConfigs.get_research_optimized
    }
    
    preset_func = presets.get(preset_name.lower())
    if not preset_func:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    return preset_func()


def load_config_from_env() -> RoutingConfig:
    """Load configuration from environment variables.
    
    Environment variables should be prefixed with ROUTING_ and use uppercase.
    Example: ROUTING_PERFORMANCE_WEIGHT=0.5
    
    Returns:
        RoutingConfig instance with environment overrides
    """
    config = RoutingConfig()
    config_dict = config.model_dump()
    
    # Check for environment overrides
    env_overrides = {}
    for key in config_dict.keys():
        env_key = f"ROUTING_{key.upper()}"
        env_value = os.getenv(env_key)
        
        if env_value is not None:
            try:
                # Try to parse as appropriate type
                if isinstance(config_dict[key], bool):
                    env_overrides[key] = env_value.lower() in ('true', '1', 'yes')
                elif isinstance(config_dict[key], (int, float)):
                    env_overrides[key] = type(config_dict[key])(env_value)
                else:
                    env_overrides[key] = env_value
                    
                print(f"Using environment override: {env_key}={env_value}")
            except (ValueError, TypeError):
                print(f"Invalid environment value for {env_key}: {env_value}")
    
    if env_overrides:
        config_dict.update(env_overrides)
        try:
            config = RoutingConfig(**config_dict)
        except ValidationError as e:
            print(f"Environment configuration validation failed: {e}")
    
    return config


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None

def get_global_config_manager() -> ConfigManager:
    """Get global configuration manager instance.
    
    Returns:
        Global ConfigManager instance
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def set_global_config(config: RoutingConfig) -> None:
    """Set global routing configuration.
    
    Args:
        config: Configuration to set globally
    """
    manager = get_global_config_manager()
    manager._config = config


def get_global_config() -> RoutingConfig:
    """Get global routing configuration.
    
    Returns:
        Global RoutingConfig instance
    """
    manager = get_global_config_manager()
    return manager.get_config()