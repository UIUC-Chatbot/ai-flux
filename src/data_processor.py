#!/usr/bin/env python3
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from .model_executor import ModelExecutor

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles batch processing of inputs through a model executor."""
    
    def __init__(
        self, 
        model_executor: ModelExecutor,
        model: str,
        batch_size: int = 32
    ):
        """Initialize batch processor.
        
        Args:
            model_executor: Instance of ModelExecutor to use for generating responses
            model: Name of the model to use
            batch_size: Number of inputs to process in each batch
        """
        self.model_executor = model_executor
        self.model = model
        self.batch_size = batch_size
        
        # Load default parameters from environment
        self.default_temperature = float(os.getenv('MODEL_TEMPERATURE', 0.7))
        self.default_top_p = float(os.getenv('MODEL_TOP_P', 0.9))
        self.default_max_tokens = int(os.getenv('MODEL_MAX_TOKENS', 2048))
        self.system_prompt = os.getenv('MODEL_SYSTEM_PROMPT', 
            "You are a helpful assistant.")
    
    def load_inputs(self, input_dir: str) -> List[Dict[str, Any]]:
        """Load input prompts from files in the input directory.
        
        Args:
            input_dir: Directory containing JSON input files
            
        Returns:
            List of input dictionaries
        """
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
        """Save processed outputs to the output directory.
        
        Args:
            outputs: List of output dictionaries to save
            output_dir: Directory to save results in
        """
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
    
    def format_prompt(self, prompt: str) -> str:
        """Format a prompt with the system prompt and chat template.
        
        Args:
            prompt: Raw user prompt
            
        Returns:
            Formatted prompt string
        """
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of inputs.
        
        Args:
            batch: List of input dictionaries to process
            
        Returns:
            List of results with inputs, outputs, and timestamps
        """
        results = []
        for item in batch:
            try:
                prompt = item.get('prompt', '')
                if not prompt:
                    logger.warning(f"Empty prompt in item: {item}")
                    continue
                
                formatted_prompt = self.format_prompt(prompt)
                
                response = self.model_executor.generate(
                    model=self.model,
                    prompt=formatted_prompt,
                    temperature=item.get('temperature', self.default_temperature),
                    top_p=item.get('top_p', self.default_top_p),
                    max_tokens=item.get('max_tokens', self.default_max_tokens),
                    stop=item.get('stop', None)
                )
                
                results.append({
                    'input': item,
                    'output': response['response'],
                    'metadata': response.get('metadata', {}),
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
        """Process all inputs in batches.
        
        Args:
            input_dir: Directory containing input JSON files
            output_dir: Directory to save results in
        """
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