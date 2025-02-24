#!/usr/bin/env python3
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from ..core.processor import BaseProcessor
from ..core.client import LLMClient
from ..core.config import ModelConfig
from ..io.handlers import InputHandler, JSONBatchHandler
from ..io.output import OutputHandler, JSONOutputHandler, TimestampedOutputHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor(BaseProcessor):
    """Processor for batch processing inputs through LLM."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        input_handler: Optional[InputHandler] = None,
        output_handler: Optional[OutputHandler] = None,
        batch_size: Optional[int] = None
    ):
        """Initialize batch processor.
        
        Args:
            model_config: Model configuration
            input_handler: Handler for processing inputs
            output_handler: Handler for saving outputs
            batch_size: Size of batches to process
        """
        super().__init__(model_config.name)
        self.config = model_config
        self.input_handler = input_handler or JSONBatchHandler()
        self.output_handler = TimestampedOutputHandler(
            output_handler or JSONOutputHandler()
        )
        self.batch_size = batch_size or model_config.resources.batch_size
        self.client = None
    
    def setup(self) -> None:
        """Setup processor by initializing client."""
        self.client = LLMClient()
        
        # Warm up the model
        logger.info("Warming up model...")
        try:
            self.client.generate(
                model=self.model,
                prompt="warming up the model",
                system_prompt=self.config.system.prompt
            )
        except Exception as e:
            logger.error(f"Error warming up model: {e}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.client:
            self.client.session.close()
    
    def process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of inputs.
        
        Args:
            batch: List of input items to process
            
        Returns:
            List of processed results
        """
        results = []
        for item in batch:
            try:
                # Validate input
                if not self.validate_input(item):
                    logger.warning(f"Invalid input item: {item}")
                    continue
                
                # Get prompt and parameters
                prompt = item.get('prompt', '')
                if not prompt:
                    logger.warning(f"Empty prompt in item: {item}")
                    continue
                
                # Get generation parameters with defaults from config
                params = {
                    'temperature': item.get(
                        'temperature',
                        self.config.parameters.temperature
                    ),
                    'top_p': item.get(
                        'top_p',
                        self.config.parameters.top_p
                    ),
                    'max_tokens': item.get(
                        'max_tokens',
                        self.config.parameters.max_tokens
                    ),
                    'stop': item.get(
                        'stop',
                        self.config.parameters.stop_sequences
                    )
                }
                
                # Generate response
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    system_prompt=self.config.system.prompt,
                    **params
                )
                
                # Format and store result
                result = {
                    'input': item,
                    'output': response,
                    'timestamp': time.time()
                }
                results.append(self.format_output(result))
                
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                results.append({
                    'input': item,
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        return results
    
    def process_all(
        self,
        input_source: Union[str, Path],
        output_path: Union[str, Path],
        save_interval: int = 10,
        **kwargs
    ) -> None:
        """Process all inputs in batches.
        
        Args:
            input_source: Source of input data
            output_path: Path to save results
            save_interval: Number of batches between saves
            **kwargs: Additional parameters for input handler
        """
        try:
            # Setup processor
            self.setup()
            
            # Process inputs in batches
            all_results = []
            current_batch = []
            
            logger.info(f"Processing inputs from {input_source}")
            for item in tqdm(self.input_handler.process(input_source, **kwargs)):
                current_batch.append(item)
                
                # Process batch when full
                if len(current_batch) >= self.batch_size:
                    results = self.process_batch(current_batch)
                    all_results.extend(results)
                    current_batch = []
                    
                    # Save intermediate results
                    if len(all_results) % (self.batch_size * save_interval) == 0:
                        self.output_handler.save(all_results, output_path)
            
            # Process remaining items
            if current_batch:
                results = self.process_batch(current_batch)
                all_results.extend(results)
            
            # Save final results
            self.output_handler.save(all_results, output_path)
            logger.info("Batch processing completed")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
            
        finally:
            self.cleanup() 