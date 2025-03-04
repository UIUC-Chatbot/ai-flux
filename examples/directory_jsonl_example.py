#!/usr/bin/env python3
"""Example for processing a directory of files with the JSONL-first approach in AI-Flux."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aiflux.core.config import Config
from src.aiflux.processors.batch import BatchProcessor
from src.aiflux.converters import directory_to_jsonl
from src.aiflux.slurm.runner import SlurmRunner
from examples.utils import get_timestamped_filename, ensure_results_dir

def process_directory_with_jsonl():
    """Process a directory of text files using the JSONL-first approach."""
    # Load model config
    config = Config()
    model_config = config.load_model_config("llama3.2", "3b")
    
    # Define input and output paths
    input_dir = Path('data/text_files')
    jsonl_path = get_timestamped_filename('data/directory_content.jsonl')
    output_path = get_timestamped_filename('results/directory_results.json')
    
    # Convert directory contents to JSONL first
    conversion_result = directory_to_jsonl(
        input_path=input_dir,
        output_path=jsonl_path,
        file_pattern="*.txt",  # Only process .txt files
        recursive=True,  # Include subdirectories
        prompt_template=(
            "Please analyze the following text file:\n\n"
            "Filename: {filename}\n"
            "Content:\n{content}\n\n"
            "Provide a summary and extract key information."
        ),
        system_prompt="You are a document analysis assistant specialized in extracting information from text files.",
        model="llama3.2:3b"
    )
    
    print(f"Converted directory contents to JSONL: {jsonl_path}")
    print(f"Total files: {conversion_result['total_files']}")
    print(f"Successful conversions: {conversion_result['successful_conversions']}")
    if conversion_result.get('skipped_files'):
        print(f"Skipped files: {len(conversion_result['skipped_files'])}")
    
    # Initialize batch processor with JSONL input
    processor = BatchProcessor(
        model_config=model_config,
        batch_size=4
    )
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "01:30:00"  # 1.5 hour time limit
    
    # Run on SLURM with the JSONL file directly
    runner = SlurmRunner(config=slurm_config, workspace=os.getcwd())
    runner.run(
        input_path=jsonl_path,
        output_path=output_path
    )
    
    print(f"Results saved to: {output_path}")

def create_example_files():
    """Create example text files for directory processing."""
    # Create directory structure
    data_dir = Path('data/text_files')
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectory
    reports_dir = data_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Create example files
    files = [
        (data_dir / 'sample1.txt', "This is a sample document with important information. The project deadline is March 15, 2025."),
        (data_dir / 'sample2.txt', "Meeting notes: Discussed the new JSONL-first architecture and its benefits for scalability."),
        (reports_dir / 'quarterly_report.txt', "Q1 2025 Report: Revenue increased by 15% compared to previous quarter. New initiatives are showing promising results.")
    ]
    
    # Write content to files
    for file_path, content in files:
        with open(file_path, 'w') as f:
            f.write(content)
    
    print(f"Created {len(files)} example files in {data_dir}")

if __name__ == "__main__":
    # Ensure results directory exists
    ensure_results_dir()
    
    # Create example files
    create_example_files()
    
    # Process directory with JSONL-first approach
    print("Processing directory with JSONL-first approach...")
    process_directory_with_jsonl() 