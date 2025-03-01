#!/usr/bin/env python3
"""Example for processing JSON batch files with AI-Flux."""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aiflux.core.config import ModelConfig, Config
from src.aiflux.processors.batch import BatchProcessor
from src.aiflux.io import JSONBatchHandler, JSONOutputHandler
from src.aiflux.runners.slurm import SlurmRunner

def process_json_batch(input_path: str, output_path: str):
    """Process JSON batch files.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save results
    """
    # Load model config
    config = Config()
    model_config = config.get_model_config("llama3")
    
    # Initialize batch processor with OpenAI-compatible client
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=JSONBatchHandler(),
        output_handler=JSONOutputHandler(),
        batch_size=8
    )
    
    # Setup SLURM runner
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "01:00:00"  # 1 hour time limit
    
    # Run on SLURM
    runner = SlurmRunner(processor, slurm_config)
    runner.run(input_path=input_path, output_path=output_path)

def create_example_json_data():
    """Create example JSON data for processing."""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create example data
    prompts = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain quantum computing in simple terms."
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with expertise in healthcare."
                },
                {
                    "role": "user",
                    "content": "How is machine learning being used in healthcare today?"
                }
            ],
            "temperature": 0.5,
            "max_tokens": 800
        }
    ]
    
    # Write to file
    with open(data_dir / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Created example data at {data_dir / 'prompts.json'}")
    return str(data_dir / "prompts.json")

if __name__ == "__main__":
    # Create example data
    input_file = create_example_json_data()
    
    # Process JSON batch
    output_file = "results/batch_results.json"
    process_json_batch(input_file, output_file) 