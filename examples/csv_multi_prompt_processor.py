#!/usr/bin/env python3
"""Example script for CSV multi-prompt processing with AI-Flux."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aiflux import BatchProcessor, SlurmRunner
from src.aiflux.core.config import Config
from src.aiflux.io import CSVMultiPromptHandler, JSONOutputHandler

def process_csv_multi_prompts():
    """Example of processing a CSV file with multiple prompts."""
    # Load model configuration
    config = Config()
    model_config = config.get_model_config("qwen2.5")
    
    # Define a system prompt for all prompts in the CSV
    system_prompt = "You are an educational assistant trained to explain complex concepts simply and accurately."
    
    # Initialize processor with CSVMultiPromptHandler
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=CSVMultiPromptHandler(prompt_column="prompt"),
        output_handler=JSONOutputHandler(),
        batch_size=4
    )
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "01:00:00"
    
    # Run on SLURM with system prompt
    runner = SlurmRunner(processor, slurm_config)
    runner.run(
        input_path='data/questions.csv',
        output_path='results/multi_prompt_results.json',
        system_prompt=system_prompt
    )

def create_example_csv_data():
    """Create example CSV data for multi-prompt processing."""
    # Create example data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create example CSV input with different prompts
    csv_input = """prompt,temperature,max_tokens
"Explain the concept of quantum entanglement in simple terms.",0.7,1024
"What are the main differences between supervised and unsupervised learning?",0.8,1500
"Describe the significance of the Transformer architecture in natural language processing.",0.7,2048
"What is CRISPR technology and how is it revolutionizing genetic engineering?",0.6,1800
"""
    
    with open(data_dir / 'questions.csv', 'w') as f:
        f.write(csv_input)
    
    print("Example CSV with multiple prompts created at data/questions.csv")

if __name__ == '__main__':
    create_example_csv_data()
    print("Processing CSV with multiple prompts...")
    process_csv_multi_prompts()

"""
Expected Input Format:
-----------------------
CSV file with a 'prompt' column (or any specified column) containing the prompts:

```csv
prompt,temperature,max_tokens
"Explain the concept of quantum entanglement in simple terms.",0.7,1024
"What are the main differences between supervised and unsupervised learning?",0.8,1500
```

Expected Output Format:
-----------------------
JSON file with results:

```json
[
  {
    "input": {
      "prompt": "Explain the concept of quantum entanglement in simple terms.",
      "temperature": 0.7,
      "max_tokens": 1024
    },
    "output": "Quantum entanglement is like a magic connection between particles...",
    "timestamp": 1234567890.123
  },
  {
    "input": {
      "prompt": "What are the main differences between supervised and unsupervised learning?",
      "temperature": 0.8,
      "max_tokens": 1500
    },
    "output": "Supervised learning is when an algorithm learns from labeled data...",
    "timestamp": 1234567890.456
  }
]
```
""" 