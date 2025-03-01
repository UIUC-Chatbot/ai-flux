#!/usr/bin/env python3
"""Example for processing images with AI-Flux vision capabilities."""

import os
import sys
import json
import requests
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aiflux.core.config import ModelConfig, Config
from src.aiflux.processors.batch import BatchProcessor
from src.aiflux.io.vision import VisionHandler
from src.aiflux.io.base import JSONOutputHandler
from src.aiflux.runners.slurm import SlurmRunner

def process_images_with_custom_prompts(image_dir: str, output_path: str, prompts_file: str):
    """Process images using custom prompts for each image.
    
    Args:
        image_dir: Directory containing images to process
        output_path: Path to save analysis results
        prompts_file: JSON file with custom prompts for each image
    """
    # Load model config for vision
    config = Config()
    model_config = config.get_model_config("llama3")
    
    # Initialize vision handler with custom prompts file
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=VisionHandler(prompts_file=prompts_file),
        output_handler=JSONOutputHandler(),
        batch_size=4
    )
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.partition = "gpuA100x4"
    slurm_config.gpus_per_node = 1
    slurm_config.time = "01:00:00"  # 1 hour time limit
    
    # Run on SLURM
    runner = SlurmRunner(processor, slurm_config)
    runner.run(input_path=image_dir, output_path=output_path)

def process_images_with_prompt_map(image_dir: str, output_path: str):
    """Process images using a prompt map passed directly to the handler.
    
    Args:
        image_dir: Directory containing images to process
        output_path: Path to save analysis results
    """
    # Load model config for vision
    config = Config()
    model_config = config.get_model_config("llama3")
    
    # Define custom prompts for each image
    prompts_map = {
        "sample_1.jpg": "Describe this urban scene in detail, noting architecture and city features.",
        "sample_2.jpg": "Analyze this natural landscape and identify key environmental features.",
        "sample_3.jpg": "Extract and transcribe any text visible in this image."
    }
    
    # Define a system prompt for all images
    system_prompt = "You are a visual analysis expert with experience in photography, architecture, and nature."
    
    # Initialize vision handler with default prompt template
    # (will be used for any images not in the prompts_map)
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=VisionHandler(prompt_template="General image analysis"),
        output_handler=JSONOutputHandler(),
        batch_size=4
    )
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.partition = "gpuA100x4"
    slurm_config.gpus_per_node = 1
    slurm_config.time = "01:00:00"  # 1 hour time limit
    
    # Run on SLURM with custom prompts and system prompt passed as kwargs
    runner = SlurmRunner(processor, slurm_config)
    runner.run(
        input_path=image_dir, 
        output_path=output_path,
        prompts_map=prompts_map,
        system_prompt=system_prompt
    )

def download_sample_images():
    """Download sample images and create prompt files for vision processing.
    
    Returns:
        Tuple containing:
        - Path to directory containing downloaded images
        - Path to JSON file with custom prompts
    """
    # Create images directory
    image_dir = Path("data/images")
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample images to download
    image_urls = [
        "https://images.unsplash.com/photo-1597502616728-3a56bd0fe9c6",  # Cityscape
        "https://images.unsplash.com/photo-1598128558393-70ff21433be0",  # Nature
        "https://images.unsplash.com/photo-1622979135225-d2ba269cf1ac"   # Text
    ]
    
    # Download images
    filenames = []
    for i, url in enumerate(image_urls):
        image_path = image_dir / f"sample_{i+1}.jpg"
        filenames.append(image_path.name)
        
        if not image_path.exists():
            print(f"Downloading image {i+1}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                print(f"Downloaded {image_path}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
    
    # Create JSON file with custom prompts for each image
    prompts = {
        "sample_1.jpg": "Describe this urban landscape focusing on architectural styles and city planning.",
        "sample_2.jpg": "What type of biome or ecosystem is shown in this image? Describe the flora and fauna you might find here.",
        "sample_3.jpg": "Identify and transcribe any text visible in this image. If there are UI elements, describe their function."
    }
    
    prompts_file = image_dir / "image_prompts.json"
    with open(prompts_file, "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Created sample images in {image_dir}")
    print(f"Created prompts file at {prompts_file}")
    
    return str(image_dir), str(prompts_file)

if __name__ == "__main__":
    # Download sample images and create prompts file
    image_dir, prompts_file = download_sample_images()
    
    # Method 1: Process images with custom prompts from file
    output_path1 = "results/vision_results_file_prompts.json"
    print(f"\nProcessing images with prompts from file...")
    process_images_with_custom_prompts(image_dir, output_path1, prompts_file)
    
    # Method 2: Process images with custom prompts passed as arguments
    output_path2 = "results/vision_results_arg_prompts.json"
    print(f"\nProcessing images with prompts passed as arguments...")
    process_images_with_prompt_map(image_dir, output_path2)

"""
Example Input/Output Format:

INPUT:
------
1. Image files in a directory:
   - sample_1.jpg (cityscape)
   - sample_2.jpg (nature landscape)
   - sample_3.jpg (image with text)

2. Custom prompts file (image_prompts.json):
   {
     "sample_1.jpg": "Describe this urban landscape focusing on architectural styles...",
     "sample_2.jpg": "What type of biome or ecosystem is shown in this image?...",
     "sample_3.jpg": "Identify and transcribe any text visible in this image..."
   }

OUTPUT:
-------
JSON file with results:
[
  {
    "input": {
      "messages": [
        {"role": "system", "content": "You are a visual analysis expert..."},
        {"role": "user", "content": [
          {"type": "text", "text": "Describe this urban landscape..."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]}
      ],
      "metadata": {"filename": "sample_1.jpg", "path": "data/images/sample_1.jpg"}
    },
    "output": "This urban landscape features a modern city skyline with...",
    "metadata": {"model": "llama3", "timestamp": 1234567890.123}
  },
  ... (similar entries for other images)
]
""" 