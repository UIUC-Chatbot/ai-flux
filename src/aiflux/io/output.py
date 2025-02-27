#!/usr/bin/env python3
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OutputHandler(ABC):
    """Base class for output handlers."""
    
    @abstractmethod
    def save(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Save results to output destination.
        
        Args:
            results: List of results to save
            output_path: Path to save results
            **kwargs: Additional save parameters
        """
        pass

class JSONOutputHandler(OutputHandler):
    """Handler for saving results as JSON."""
    
    def save(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path],
        indent: int = 2,
        **kwargs
    ) -> None:
        """Save results as JSON.
        
        Args:
            results: List of results to save
            output_path: Path to save JSON file
            indent: JSON indentation level
            **kwargs: Unused
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=indent)
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise

class CSVOutputHandler(OutputHandler):
    """Handler for saving results as CSV."""
    
    def save(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path],
        flatten: bool = True,
        **kwargs
    ) -> None:
        """Save results as CSV.
        
        Args:
            results: List of results to save
            output_path: Path to save CSV file
            flatten: Whether to flatten nested dictionaries
            **kwargs: Additional pandas to_csv parameters
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            if flatten:
                # Flatten nested dictionaries
                flat_data = []
                for result in results:
                    flat_result = {}
                    for key, value in result.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                flat_result[f"{key}_{k}"] = v
                        else:
                            flat_result[key] = value
                    flat_data.append(flat_result)
                df = pd.DataFrame(flat_data)
            
            # Save to CSV
            df.to_csv(output_path, index=False, **kwargs)
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise

class TimestampedOutputHandler:
    """Decorator for adding timestamps to output files."""
    
    def __init__(self, handler: OutputHandler):
        """Initialize with base handler.
        
        Args:
            handler: Base output handler to decorate
        """
        self.handler = handler
    
    def save(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Save results with timestamp in filename.
        
        Args:
            results: List of results to save
            output_path: Base path for output file
            **kwargs: Additional save parameters
        """
        output_path = Path(output_path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Add timestamp to filename
        timestamped_path = output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"
        
        self.handler.save(results, timestamped_path, **kwargs) 