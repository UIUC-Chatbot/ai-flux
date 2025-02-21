#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ResourceConfig(BaseModel):
    gpu_layers: int = Field(..., ge=1)
    gpu_memory: str = Field(..., pattern=r"^\d+GB$")
    batch_size: int = Field(..., ge=1)
    max_concurrent: int = Field(..., ge=1)


class ParameterConfig(BaseModel):
    temperature: float = Field(..., ge=0.0, le=1.0)
    top_p: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., ge=1)
    stop_sequences: List[str]


class SystemConfig(BaseModel):
    prompt: str


class ValidationConfig(BaseModel):
    temperature_range: List[float] = Field(..., min_items=2, max_items=2)
    max_tokens_limit: int = Field(..., ge=1)
    batch_size_range: List[int] = Field(..., min_items=2, max_items=2)
    concurrent_range: List[int] = Field(..., min_items=2, max_items=2)

    @field_validator("temperature_range")
    @classmethod
    def validate_temperature_range(cls, v):
        if not (0.0 <= v[0] <= v[1] <= 1.0):
            raise ValueError("Temperature range must be between 0.0 and 1.0")
        return v

    @field_validator("batch_size_range", "concurrent_range")
    @classmethod
    def validate_range(cls, v):
        if not (v[0] <= v[1]):
            raise ValueError("Range start must be less than or equal to end")
        return v


class RequirementsConfig(BaseModel):
    min_gpu_memory: str = Field(..., pattern=r"^\d+GB$")
    recommended_gpu: str
    cuda_version: str = Field(..., pattern=r"^>=\d+\.\d+$")
    cpu_threads: int = Field(..., ge=1)
    gpu_memory_utilization: float = Field(..., ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    name: str = Field(..., pattern=r"^[a-zA-Z0-9.-]+:\d+b$")
    resources: ResourceConfig
    parameters: ParameterConfig
    system: SystemConfig
    validation: ValidationConfig
    requirements: RequirementsConfig


def parse_gpu_memory(memory_str: str) -> int:
    """Convert GPU memory string to GB value."""
    match = re.match(r"^(\d+)GB$", memory_str)
    if not match:
        raise ValueError(f"Invalid GPU memory format: {memory_str}")
    return int(match.group(1))


def validate_config_file(config_path: Path) -> Optional[ModelConfig]:
    """Validate a model configuration file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Validate against schema
        model_config = ModelConfig(**config_data)
        
        # Additional validation
        gpu_mem = parse_gpu_memory(model_config.resources.gpu_memory)
        min_gpu_mem = parse_gpu_memory(model_config.requirements.min_gpu_memory)
        
        if gpu_mem < min_gpu_mem:
            raise ValueError(
                f"GPU memory ({gpu_mem}GB) less than minimum required ({min_gpu_mem}GB)"
            )
        
        if (model_config.resources.batch_size < model_config.validation.batch_size_range[0] or
            model_config.resources.batch_size > model_config.validation.batch_size_range[1]):
            raise ValueError("Batch size outside valid range")
        
        if (model_config.resources.max_concurrent < model_config.validation.concurrent_range[0] or
            model_config.resources.max_concurrent > model_config.validation.concurrent_range[1]):
            raise ValueError("Max concurrent requests outside valid range")
        
        return model_config
    
    except Exception as e:
        print(f"Error validating {config_path}: {str(e)}")
        return None


def validate_all_configs(config_dir: Path) -> Dict[str, ModelConfig]:
    """Validate all model configurations in directory."""
    valid_configs = {}
    
    for model_dir in config_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        for config_file in model_dir.glob("*.yaml"):
            config = validate_config_file(config_file)
            if config is not None:
                model_key = f"{model_dir.name}/{config_file.stem}"
                valid_configs[model_key] = config
    
    return valid_configs


if __name__ == "__main__":
    # Example usage
    config_dir = Path("config/models")
    valid_configs = validate_all_configs(config_dir)
    
    print(f"\nValidated {len(valid_configs)} configurations:")
    for model_key, config in valid_configs.items():
        print(f"✓ {model_key}: {config.name}") 