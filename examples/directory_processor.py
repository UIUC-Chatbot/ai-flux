#!/usr/bin/env python3
"""Example script for directory processing with AI-Flux."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aiflux import BatchProcessor, SlurmRunner
from src.aiflux.core.config import Config
from src.aiflux.io import DirectoryHandler
from src.aiflux.io.base import JSONOutputHandler

def process_directory():
    """Example of processing a directory of text files."""
    # Load model configuration
    config = Config()
    model_config = config.get_model_config("llama3")
    
    # Define the prompt template for processing files
    prompt_template = (
        "Analyze the following text file and provide key insights:\n\n"
        "Filename: {filename}\n"
        "Content:\n{content}\n\n"
        "Please provide:\n"
        "1. A summary of the main topics\n"
        "2. Key entities mentioned\n"
        "3. The overall sentiment (positive, negative, or neutral)"
    )
    
    # Define system prompt for context
    system_prompt = "You are a content analysis expert with experience in extracting insights from documents."
    
    # Initialize processor with DirectoryHandler
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=DirectoryHandler(),
        output_handler=JSONOutputHandler(),
        batch_size=2
    )
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "01:30:00"
    
    # Process inputs with both prompt template and system prompt
    runner = SlurmRunner(processor, slurm_config)
    runner.run(
        input_path='data/articles/',
        output_path='results/articles_analysis.json',
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        file_pattern="*.txt"  # Process only .txt files
    )

def create_example_files():
    """Create example text files for directory processing."""
    # Create example data directory
    articles_dir = Path('data/articles')
    articles_dir.mkdir(exist_ok=True, parents=True)
    
    # Create example article files
    article1 = """# Latest Advancements in Artificial Intelligence
    
The field of artificial intelligence has seen remarkable progress in recent years.
Large language models are now capable of generating human-like text and answering
complex questions with impressive accuracy. These models are trained on vast amounts
of data and employ sophisticated neural network architectures.

Researchers continue to push the boundaries of what AI can accomplish, with applications
ranging from medical diagnosis to creative writing assistance.
"""
    
    article2 = """# Climate Change: Global Challenges and Solutions
    
Climate change remains one of the most pressing issues facing humanity today.
Rising global temperatures have led to more frequent extreme weather events,
melting polar ice caps, and disruption of ecosystems worldwide.

Many countries have pledged to reduce carbon emissions and invest in renewable
energy sources to mitigate the effects of climate change. However, coordinated
global action is necessary to address this complex challenge effectively.
"""
    
    # Write articles to files
    with open(articles_dir / 'ai_advancements.txt', 'w') as f:
        f.write(article1)
    
    with open(articles_dir / 'climate_change.txt', 'w') as f:
        f.write(article2)
    
    print("Example article files created in data/articles/")

if __name__ == '__main__':
    create_example_files()
    print("Processing directory of text files...")
    process_directory()

"""
Expected Input Format:
-----------------------
A directory containing text files:

```
data/articles/
├── ai_advancements.txt
└── climate_change.txt
```

The DirectoryHandler reads each file and applies the prompt template, 
replacing {content} with the file content and {filename} with the file name.

Expected Output Format:
-----------------------
JSON file with results:

```json
[
  {
    "input": {
      "prompt": "Analyze the following text file and provide key insights:\n\nFilename: ai_advancements.txt\nContent:\n# Latest Advancements in Artificial Intelligence...",
      "file": "data/articles/ai_advancements.txt"
    },
    "output": "1. Main topics: This text discusses recent progress in artificial intelligence...",
    "timestamp": 1234567890.123
  },
  {
    "input": {
      "prompt": "Analyze the following text file and provide key insights:\n\nFilename: climate_change.txt\nContent:\n# Climate Change: Global Challenges and Solutions...",
      "file": "data/articles/climate_change.txt"
    },
    "output": "1. Main topics: The text focuses on climate change as a major global challenge...",
    "timestamp": 1234567890.456
  }
]
```
""" 