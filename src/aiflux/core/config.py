#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator

class ResourceConfig(BaseModel):
    """Model resource configuration."""
    gpu_layers: int = Field(..., ge=1)
    gpu_memory: str = Field(..., pattern=r"^\d+GB$")
    batch_size: int = Field(..., ge=1)
    max_concurrent: int = Field(..., ge=1)

class ParameterConfig(BaseModel):
    """Model parameter configuration."""
    temperature: float = Field(..., ge=0.0, le=1.0)
    top_p: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., ge=1)
    stop_sequences: List[str]

class SystemConfig(BaseModel):
    """Model system configuration."""
    prompt: str

class ValidationConfig(BaseModel):
    """Model validation configuration."""
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
    """Model requirements configuration."""
    min_gpu_memory: str = Field(..., pattern=r"^\d+GB$")
    recommended_gpu: str
    cuda_version: str = Field(..., pattern=r"^>=\d+\.\d+$")
    cpu_threads: int = Field(..., ge=1)
    gpu_memory_utilization: float = Field(..., ge=0.0, le=1.0)

class ModelConfig(BaseModel):
    """Complete model configuration."""
    name: str = Field(..., pattern=r"^[a-zA-Z0-9.-]+:\d+b$")
    resources: ResourceConfig
    parameters: ParameterConfig
    system: SystemConfig
    validation: ValidationConfig
    requirements: RequirementsConfig

class SlurmConfig(BaseModel):
    """SLURM configuration with env var support."""
    account: str = Field(default_factory=lambda: os.getenv('SLURM_ACCOUNT'))
    partition: str = Field(
        default_factory=lambda: os.getenv('SLURM_PARTITION', 'gpuA100x4')
    )
    nodes: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_NODES', '1'))
    )
    gpus_per_node: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_GPUS_PER_NODE', '1'))
    )
    time: str = Field(
        default_factory=lambda: os.getenv('SLURM_TIME', '00:30:00')
    )
    memory: str = Field(
        default_factory=lambda: os.getenv('SLURM_MEM', '32G')
    )
    cpus_per_task: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_CPUS_PER_TASK', '4'))
    )

def parse_gpu_memory(memory_str: str) -> int:
    """Convert GPU memory string to GB value."""
    match = re.match(r"^(\d+)GB$", memory_str)
    if not match:
        raise ValueError(f"Invalid GPU memory format: {memory_str}")
    return int(match.group(1))

class Config:
    """Central configuration management."""
    
    def __init__(self):
        self.package_dir = Path(__file__).parent.parent
        self.templates_dir = self.package_dir / 'templates'
    
    def load_model_config(
        self,
        model_type: str,
        model_size: str,
        custom_config_path: Optional[str] = None
    ) -> ModelConfig:
        """Load and validate model configuration.
        
        Args:
            model_type: Type of model (e.g., 'qwen', 'llama')
            model_size: Size of model (e.g., '7b', '70b')
            custom_config_path: Optional path to custom config
            
        Returns:
            Validated ModelConfig
            
        Raises:
            ValueError: If configuration is invalid
        """
        if custom_config_path:
            config_path = Path(custom_config_path)
        else:
            config_path = self.templates_dir / model_type / f"{model_size}.yaml"
        
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
                    f"GPU memory ({gpu_mem}GB) less than minimum required "
                    f"({min_gpu_mem}GB)"
                )
            
            if (model_config.resources.batch_size < 
                model_config.validation.batch_size_range[0] or
                model_config.resources.batch_size > 
                model_config.validation.batch_size_range[1]):
                raise ValueError("Batch size outside valid range")
            
            if (model_config.resources.max_concurrent < 
                model_config.validation.concurrent_range[0] or
                model_config.resources.max_concurrent > 
                model_config.validation.concurrent_range[1]):
                raise ValueError("Max concurrent requests outside valid range")
            
            return model_config
            
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {str(e)}")
    
    def get_slurm_config(
        self,
        overrides: Optional[Dict[str, Any]] = None
    ) -> SlurmConfig:
        """Get SLURM configuration with overrides.
        
        Args:
            overrides: Optional configuration overrides
            
        Returns:
            SlurmConfig instance
        """
        config = SlurmConfig()
        if overrides:
            for key, value in overrides.items():
                setattr(config, key, value)
        return config
    
    def to_env_dict(self) -> Dict[str, str]:
        """Convert configurations to environment variables.
        
        Returns:
            Dictionary of environment variables
        """
        slurm_config = self.get_slurm_config()
        
        env_dict = {
            'SLURM_ACCOUNT': slurm_config.account,
            'SLURM_PARTITION': slurm_config.partition,
            'SLURM_NODES': str(slurm_config.nodes),
            'SLURM_GPUS_PER_NODE': str(slurm_config.gpus_per_node),
            'SLURM_TIME': slurm_config.time,
            'SLURM_MEM': slurm_config.memory,
            'SLURM_CPUS_PER_TASK': str(slurm_config.cpus_per_task)
        }
        
        return {k: v for k, v in env_dict.items() if v is not None} 