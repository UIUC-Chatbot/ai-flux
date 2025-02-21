#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/batch_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, host: str = None, port: int = None):
        """Initialize Ollama client with optional host and port."""
        if host is None:
            host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        self.base_url = f"http://{host}"
        self.session = requests.Session()
    
    def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from the model."""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Disable streaming to get a single response
            **kwargs
        }
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            
            # Parse the response and extract just the generated text
            response_data = response.json()
            if isinstance(response_data, dict):
                return response_data.get('response', '')
            return str(response_data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {e}")
            return response.text

class BatchProcessor:
    def __init__(self, model: str, batch_size: int = 32):
        """Initialize batch processor with model and batch size."""
        self.model = model
        self.batch_size = batch_size
        self.client = OllamaClient()
        
        # Load default parameters from environment
        self.default_temperature = float(os.getenv('MODEL_TEMPERATURE', 0.7))
        self.default_top_p = float(os.getenv('MODEL_TOP_P', 0.9))
        self.default_max_tokens = int(os.getenv('MODEL_MAX_TOKENS', 2048))
        self.system_prompt = os.getenv('MODEL_SYSTEM_PROMPT', 
            "You are a helpful assistant.")
        
    def load_inputs(self, input_dir: str) -> List[Dict[str, Any]]:
        """Load input prompts from files in the input directory."""
        inputs = []
        input_path = Path(input_dir)
        
        for file in input_path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        inputs.extend(data)
                    else:
                        inputs.append(data)
            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")
                continue
        
        return inputs
    
    def save_outputs(self, outputs: List[Dict[str, Any]], output_dir: str):
        """Save processed outputs to the output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = output_path / f"batch_results_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(outputs, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of inputs."""
        results = []
        for item in batch:
            try:
                prompt = item.get('prompt', '')
                if not prompt:
                    logger.warning(f"Empty prompt in item: {item}")
                    continue
                
                # Add chat template with system prompt
                formatted_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                response = self.client.generate(
                    model=self.model,
                    prompt=formatted_prompt,
                    temperature=item.get('temperature', self.default_temperature),
                    top_p=item.get('top_p', self.default_top_p),
                    max_tokens=item.get('max_tokens', self.default_max_tokens),
                    stop=item.get('stop', None)
                )
                
                results.append({
                    'input': item,
                    'output': response,
                    'timestamp': time.time()
                })
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                results.append({
                    'input': item,
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        return results

    def process_all(self, input_dir: str, output_dir: str):
        """Process all inputs in batches."""
        inputs = self.load_inputs(input_dir)
        if not inputs:
            logger.error("No valid inputs found")
            return
        
        logger.info(f"Processing {len(inputs)} inputs in batches of {self.batch_size}")
        all_results = []
        
        for i in tqdm(range(0, len(inputs), self.batch_size)):
            batch = inputs[i:i + self.batch_size]
            results = self.process_batch(batch)
            all_results.extend(results)
            
            # Save intermediate results
            if i % (self.batch_size * 10) == 0:
                self.save_outputs(all_results, output_dir)
        
        # Save final results
        self.save_outputs(all_results, output_dir)
        logger.info("Batch processing completed")

def main():
    parser = argparse.ArgumentParser(description="Batch process inputs through LLM")
    parser.add_argument("--input-dir", required=True, help="Input directory containing JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    try:
        processor = BatchProcessor(args.model, args.batch_size)
        processor.process_all(args.input_dir, args.output_dir)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 