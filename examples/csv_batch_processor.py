#!/usr/bin/env python3
"""Example script for CSV batch processing with AI-Flux."""

import os
from pathlib import Path

from aiflux import BatchProcessor, SlurmRunner
from aiflux.core.config import Config
from aiflux.io import CSVSinglePromptHandler

def process_csv_data():
    """Example of processing a CSV file."""
    # Load model configuration
    config = Config()
    model_config = config.load_model_config('llama3.2', '3b')
    
    # Initialize processor with CSV handler
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=CSVSinglePromptHandler(),
        batch_size=4
    )
    
    # Run on SLURM
    runner = SlurmRunner(
        config=config.get_slurm_config({
            'account': os.getenv('SLURM_ACCOUNT'),
            'time': '02:00:00'
        })
    )
    
    # Process inputs with template
    runner.run(
        processor,
        input_source='data/papers.csv',
        output_path='results/paper_summaries.json',
        prompt_template=(
            "Please summarize the following research paper:\n\n"
            "Title: {title}\n"
            "Abstract: {abstract}\n\n"
            "Provide a concise summary focusing on:\n"
            "1. Main research question\n"
            "2. Key methodology\n"
            "3. Main findings\n"
            "4. Significance of results"
        )
    )

def create_example_csv_data():
    """Create example CSV data for batch processing."""
    # Create example data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create example CSV input
    csv_input = """title,abstract
Quantum Supremacy Using a Programmable Superconducting Processor,"The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space."
Large Language Models in Machine Learning,"Recent advances in transformer architectures and pre-training techniques have led to significant improvements in natural language processing tasks. This paper surveys the current state of large language models and their applications."
"""
    
    with open(data_dir / 'papers.csv', 'w') as f:
        f.write(csv_input)

if __name__ == '__main__':
    create_example_csv_data()
    print("Processing CSV data...")
    process_csv_data() 