#!/usr/bin/env python3
"""Batch processor for AI-Flux."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Generator, Iterator
import tempfile
import json

from ..core.client import LLMClient
from ..core.config import ModelConfig
from ..io.base import InputHandler, OutputHandler, OutputResult

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processor for batch processing inputs with LLM."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        input_handler: InputHandler,
        output_handler: OutputHandler,
        batch_size: int = 4,
        save_frequency: int = 50,
        temp_dir: Optional[str] = None
    ):
        """Initialize batch processor.
        
        Args:
            model_config: Model configuration
            input_handler: Handler for processing input
            output_handler: Handler for processing output
            batch_size: Number of items to process in a batch
            save_frequency: How often to save intermediate results (items)
            temp_dir: Directory for storing temporary files
        """
        self.model_config = model_config
        self.input_handler = input_handler
        self.output_handler = output_handler
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.client = None
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.temp_file = os.path.join(self.temp_dir, f"aiflux_{int(time.time())}.jsonl")
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def setup(self):
        """Initialize LLM client and warm up model."""
        # Initialize client
        logger.info("Initializing LLM client")
        self.client = LLMClient()
        
        # Check if model exists and warm it up
        model = self.model_config.name
        logger.info(f"Warming up model: {model}")
        
        try:
            # Simple warmup query to ensure model is loaded
            warmup_messages = [{"role": "user", "content": "Hello, world!"}]
            self.client.generate(
                model=model,
                messages=warmup_messages,
                max_tokens=5
            )
            logger.info(f"Model {model} warmed up successfully")
        except Exception as e:
            logger.error(f"Error warming up model: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        if self.client:
            logger.info("Closing client session")
            self.client.session.close()
            self.client = None
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[OutputResult]:
        """Process a batch of items.
        
        Args:
            batch: List of input items to process
            
        Returns:
            List of processed output results
        """
        results = []
        model = self.model_config.name
        
        for item in batch:
            try:
                # Extract messages from the item
                messages = item.get("messages", [])
                
                # Generate response
                response = self.client.generate(
                    model=model,
                    messages=messages,
                    temperature=item.get("temperature", self.model_config.parameters.temperature),
                    max_tokens=item.get("max_tokens", self.model_config.parameters.max_tokens),
                    top_p=item.get("top_p", self.model_config.parameters.top_p),
                    stop=item.get("stop", self.model_config.parameters.stop_sequences),
                )
                
                # Create result
                result = OutputResult(
                    input=item,
                    output=response,
                    metadata={
                        "model": model,
                        "timestamp": time.time(),
                        **item.get("metadata", {})
                    }
                )
                results.append(result)
                logger.debug(f"Processed item with message content: {str(messages)[:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                # Add error result
                result = OutputResult(
                    input=item,
                    output=None,
                    error=str(e),
                    metadata={
                        "model": model,
                        "timestamp": time.time(),
                        "error": True,
                        **item.get("metadata", {})
                    }
                )
                results.append(result)
        
        return results
    
    def run(self, input_path: str, output_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Run batch processing on input data.
        
        Args:
            input_path: Path to input data
            output_path: Path to save output data
            **kwargs: Additional parameters to pass to input handler
            
        Returns:
            List of results
        """
        try:
            self.setup()
            
            # Process input
            logger.info(f"Processing input: {input_path}")
            all_results = []
            processed_count = 0
            last_save = 0
            batch = []
            
            # Pass additional arguments to the input handler
            input_iterator = self.input_handler.process(input_path, **kwargs)
            
            for item in input_iterator:
                batch.append(item)
                
                # Process batch when it reaches the batch size
                if len(batch) >= self.batch_size:
                    batch_results = self.process_batch(batch)
                    all_results.extend(batch_results)
                    processed_count += len(batch)
                    batch = []
                    
                    # Save intermediate results
                    if processed_count - last_save >= self.save_frequency:
                        self._save_intermediate_results(all_results, output_path)
                        last_save = processed_count
                    
                    logger.info(f"Processed {processed_count} items")
            
            # Process any remaining items
            if batch:
                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)
                processed_count += len(batch)
                logger.info(f"Processed {processed_count} items")
            
            # Save final results
            self._save_results(all_results, output_path)
            logger.info(f"Processing complete. Results saved to {output_path}")
            
            return all_results
        
        finally:
            self.cleanup()
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save intermediate results to a temporary file.
        
        Args:
            results: List of results to save
            output_path: Path to the final output file
        """
        try:
            # Convert results to serializable format
            serializable_results = [result.to_dict() for result in results]
            
            # Write to temporary file
            with open(self.temp_file, 'w') as f:
                json.dump(serializable_results, f)
            
            logger.info(f"Saved intermediate results to {self.temp_file}")
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def _save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save final results using the output handler.
        
        Args:
            results: List of results to save
            output_path: Path to save the results
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Use output handler to save results
            self.output_handler.save(results, output_path)
            
            # Clean up temporary file
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
                logger.debug(f"Removed temporary file: {self.temp_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise 