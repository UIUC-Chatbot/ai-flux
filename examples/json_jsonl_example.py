#!/usr/bin/env python3
"""Example for processing JSON files with the JSONL-first approach in AI-Flux."""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aiflux.core.config import Config
from src.aiflux.processors.batch import BatchProcessor
from src.aiflux.converters import json_to_jsonl
from src.aiflux.slurm.runner import SlurmRunner
from examples.utils import get_timestamped_filename, ensure_results_dir

def process_json_with_jsonl(input_path: str, output_path: str):
    """Process JSON files using the JSONL-first approach.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save results
    """
    # Load model config
    config = Config()
    model_config = config.load_model_config("llama3.2", "3b")
    
    # Convert JSON to JSONL first
    jsonl_path = str(Path(input_path).with_suffix('.jsonl'))
    conversion_result = json_to_jsonl(
        input_path=input_path,
        output_path=jsonl_path,
        model="llama3.2:3b"
    )
    
    print(f"Converted JSON to JSONL: {jsonl_path}")
    print(f"Total entries: {conversion_result['total_entries']}")
    print(f"Successful conversions: {conversion_result['successful_conversions']}")
    
    # Initialize batch processor (no input handler needed)
    processor = BatchProcessor(
        model_config=model_config,
        batch_size=8
    )
    
    # Setup SLURM runner
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "01:00:00"  # 1 hour time limit
    
    # Run on SLURM with JSONL file directly
    runner = SlurmRunner(config=slurm_config, workspace=os.getcwd())
    runner.run(
        input_path=jsonl_path,  # processor is optional now
        output_path=output_path
    )
    
    print(f"Results saved to: {output_path}")

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
    json_path = data_dir / "prompts.json"
    with open(json_path, "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Created example data at {json_path}")
    return str(json_path)

if __name__ == "__main__":
    # Ensure results directory exists
    ensure_results_dir()
    
    # Create example data
    input_file = create_example_json_data()
    
    # Process JSON with JSONL-first approach
    output_file = get_timestamped_filename("results/jsonl_batch_results.json")
    print("Processing JSON with JSONL-first approach...")
    process_json_with_jsonl(input_file, output_file) 