#!/usr/bin/env python3
"""Timestamped output handler for AI-Flux."""

import time
import logging
from pathlib import Path
from typing import List, Any, Dict

from ..base import OutputHandler, OutputResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimestampedOutputHandler(OutputHandler):
    """Decorator for adding timestamps to output files."""
    
    def __init__(self, handler: OutputHandler):
        """Initialize with base handler.
        
        Args:
            handler: Base output handler to decorate
        """
        self.handler = handler
    
    def save(
        self,
        results: List[OutputResult],
        output_path: str,
        **kwargs
    ) -> None:
        """Save results with timestamp in filename.
        
        Args:
            results: List of results to save
            output_path: Base path for output file
            **kwargs: Additional save parameters
        """
        output_file = Path(output_path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Add timestamp to filename
        timestamped_path = output_file.parent / f"{output_file.stem}_{timestamp}{output_file.suffix}"
        
        self.handler.save(results, str(timestamped_path), **kwargs)
        logger.info(f"Results saved with timestamp to {timestamped_path}") 