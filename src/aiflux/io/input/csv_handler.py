#!/usr/bin/env python3
"""CSV input handlers for AI-Flux."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import pandas as pd

from ..base import InputHandler

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVSinglePromptHandler(InputHandler):
    """Handler for CSV files to be processed with a single prompt template."""
    
    def process(
        self,
        input_source: str,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process CSV input file using a prompt template.
        
        Args:
            input_source: Path to CSV file
            prompt_template: Template string with placeholders matching CSV columns
            system_prompt: Optional system prompt to include in messages
            **kwargs: Additional parameters passed to pandas.read_csv
            
        Yields:
            Dict items with messages in OpenAI format
            
        Raises:
            FileNotFoundError: If input file does not exist
            ValueError: If prompt template is not provided
        """
        input_path = Path(input_source)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        if not prompt_template:
            raise ValueError("A prompt template is required for CSV processing")
            
        try:
            # Read CSV
            df = pd.read_csv(input_path, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {input_path}")
            
            # Process each row
            for i, row in df.iterrows():
                # Format prompt using row data
                try:
                    # Convert row to dict
                    row_dict = row.to_dict()
                    # Format prompt template with row values
                    formatted_prompt = prompt_template.format(**row_dict)
                    
                    # Create messages list
                    messages = []
                    
                    # Add system prompt if provided
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })
                    
                    # Add user prompt
                    messages.append({
                        "role": "user",
                        "content": formatted_prompt
                    })
                    
                    # Create item with any additional parameters from the row
                    item = {"messages": messages}
                    
                    # Add any other columns that might be parameters
                    for col, value in row_dict.items():
                        if col.lower() in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                            try:
                                item[col.lower()] = float(value)
                            except (ValueError, TypeError):
                                pass
                    
                    yield item
                    
                except KeyError as e:
                    logger.warning(f"Row {i} is missing column {e} required by prompt template, skipping")
                except Exception as e:
                    logger.warning(f"Error processing row {i}: {e}, skipping")
                    
        except Exception as e:
            logger.error(f"Error processing CSV file {input_path}: {e}")
            raise

class CSVMultiPromptHandler(InputHandler):
    """Handler for CSV files with multiple prompts."""
    
    def __init__(self, prompt_column: str = "prompt"):
        """Initialize with prompt column name.
        
        Args:
            prompt_column: Name of the column containing prompts
        """
        self.prompt_column = prompt_column
    
    def process(
        self,
        input_source: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process CSV file with multiple prompts.
        
        Args:
            input_source: Path to CSV file
            system_prompt: Optional system prompt to include in messages
            **kwargs: Additional parameters passed to pandas.read_csv
            
        Yields:
            Dict items with messages in OpenAI format
            
        Raises:
            FileNotFoundError: If input file does not exist
            ValueError: If prompt column is not found
        """
        input_path = Path(input_source)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        try:
            # Read CSV
            df = pd.read_csv(input_path, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {input_path}")
            
            # Check if prompt column exists
            if self.prompt_column not in df.columns:
                raise ValueError(f"Prompt column '{self.prompt_column}' not found in CSV")
                
            # Process each row
            for i, row in df.iterrows():
                try:
                    # Get prompt from specified column
                    prompt = row[self.prompt_column]
                    if not isinstance(prompt, str) or not prompt.strip():
                        logger.warning(f"Row {i} has an empty or non-string prompt, skipping")
                        continue
                        
                    # Create messages list
                    messages = []
                    
                    # Add system prompt if provided
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })
                    
                    # Add user prompt
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })
                    
                    # Create item with any additional parameters from the row
                    item = {"messages": messages}
                    
                    # Add any other columns that might be parameters
                    row_dict = row.to_dict()
                    for col, value in row_dict.items():
                        if col != self.prompt_column and col.lower() in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                            try:
                                item[col.lower()] = float(value)
                            except (ValueError, TypeError):
                                pass
                    
                    yield item
                    
                except Exception as e:
                    logger.warning(f"Error processing row {i}: {e}, skipping")
                    
        except Exception as e:
            logger.error(f"Error processing CSV file {input_path}: {e}")
            raise 