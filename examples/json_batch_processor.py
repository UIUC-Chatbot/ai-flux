#!/usr/bin/env python3
"""Example script for JSON batch processing with AI-Flux."""

import os
from pathlib import Path

from aiflux import BatchProcessor, SlurmRunner
from aiflux.core.config import Config
from aiflux.io import JSONBatchHandler

def process_json_batch():
    """Example of processing a JSON batch file."""
    # Load model configuration
    config = Config()
    model_config = config.load_model_config('qwen2.5', '7b')
    
    # Initialize processor with JSON handler
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=JSONBatchHandler(),
        batch_size=8
    )
    
    # Run on SLURM
    runner = SlurmRunner(
        config=config.get_slurm_config({
            'account': os.getenv('SLURM_ACCOUNT'),
            'time': '01:00:00'
        })
    )
    
    # Process inputs
    runner.run(
        processor,
        input_source='data/prompts.json',
        output_path='results/batch_results.json'
    )

def create_example_json_data():
    """Create example JSON data for batch processing."""
    # Create example data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create example JSON input
    json_input = """[
        {
            "prompt": "Explain quantum computing in simple terms.",
            "temperature": 0.7,
            "max_tokens": 1024
        },
        {
            "prompt": "What are the main applications of machine learning in healthcare?",
            "temperature": 0.8,
            "max_tokens": 2048
        }
    ]"""
    
    with open(data_dir / 'prompts.json', 'w') as f:
        f.write(json_input)

if __name__ == '__main__':
    create_example_json_data()
    print("Processing JSON batch...")
    process_json_batch() 