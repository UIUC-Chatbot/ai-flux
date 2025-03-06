#!/usr/bin/env python3
"""CSV output handler for AI-Flux."""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from ..base import OutputHandler, OutputResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVOutputHandler(OutputHandler):
    """Handler for saving results as CSV."""
    
    def save(
        self,
        results: List[OutputResult],
        output_path: str,
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
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert results to dict format
            results_dicts = [result.to_dict() for result in results]
            
            if flatten:
                # Flatten nested dictionaries
                flat_data = []
                for result_dict in results_dicts:
                    flat_result = {}
                    for key, value in result_dict.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                flat_result[f"{key}_{k}"] = v
                        else:
                            flat_result[key] = value
                    flat_data.append(flat_result)
                df = pd.DataFrame(flat_data)
            else:
                df = pd.DataFrame(results_dicts)
            
            # Save to CSV
            df.to_csv(output_file, index=False, **kwargs)
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise 