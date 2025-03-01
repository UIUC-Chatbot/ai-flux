#!/usr/bin/env python3
"""Input handlers for AI-Flux."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import pandas as pd

from .base import InputHandler

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JSONBatchHandler(InputHandler):
    """Handler for JSON files containing prompts in OpenAI format."""
    
    def process(
        self,
        input_source: str,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process JSON input file.
        
        Args:
            input_source: Path to JSON file with items in OpenAI format
            **kwargs: Additional parameters
            
        Yields:
            Input items from JSON with messages in OpenAI format
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            json.JSONDecodeError: If JSON parsing fails
            ValueError: If input is not in correct format
        """
        input_path = Path(input_source)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_source}")
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError(f"Input JSON must be a list of objects: {input_source}")
                
            for item in data:
                # Ensure each item has messages in OpenAI format
                if "messages" not in item:
                    # If no messages, try to convert from legacy format
                    messages = []
                    prompt_text = item.get("prompt", "")
                    
                    # Check if prompt contains system prompt
                    system_prompt = None
                    if isinstance(prompt_text, dict):
                        system_prompt = prompt_text.get("system")
                        prompt_text = prompt_text.get("prompt", "")
                    
                    # Add system message if available
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })
                    
                    # Add user message
                    messages.append({
                        "role": "user",
                        "content": prompt_text
                    })
                    
                    # Update item with messages
                    item["messages"] = messages
                    
                    # Log conversion
                    logger.debug(f"Converted legacy format to OpenAI format")
                
                # Add metadata if not present
                if "metadata" not in item:
                    item["metadata"] = {}
                
                yield item
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            raise

class CSVSinglePromptHandler(InputHandler):
    """Handler for running same prompt on CSV rows."""
    
    def process(
        self,
        input_source: str,
        prompt_template: str,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process CSV input file.
        
        Args:
            input_source: Path to CSV file
            prompt_template: Template string with {field} placeholders
            **kwargs: Additional parameters for prompt formatting
            
        Yields:
            Formatted prompts for each row in OpenAI-compatible format
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If prompt formatting fails
        """
        input_path = Path(input_source)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_source}")
        
        try:
            df = pd.read_csv(input_source)
            
            for _, row in df.iterrows():
                # Format prompt template with row data
                try:
                    prompt = prompt_template.format(**row.to_dict())
                    
                    # Get optional system prompt
                    system_prompt = kwargs.get("system_prompt")
                    
                    # Create messages array for OpenAI format
                    messages = []
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })
                    
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })
                    
                    # Prepare item with metadata
                    yield_item = {
                        "messages": messages,
                        "metadata": {
                            "row": row.to_dict()
                        }
                    }
                    
                    # Add model parameters if provided
                    for param in ["temperature", "max_tokens", "top_p", "stop"]:
                        if param in kwargs:
                            yield_item[param] = kwargs[param]
                    
                    yield yield_item
                    
                except KeyError as e:
                    logger.error(f"Missing field in template: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            raise

class CSVMultiPromptHandler(InputHandler):
    """Handler for CSV files where each row contains a prompt."""
    
    def process(
        self,
        input_source: str,
        prompt_column: str = "prompt",
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process CSV input file.
        
        Args:
            input_source: Path to CSV file
            prompt_column: Name of column containing prompts
            **kwargs: Additional parameters for each prompt
            
        Yields:
            Prompts from each row in OpenAI-compatible format
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If prompt column is not found
        """
        input_path = Path(input_source)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_source}")
        
        try:
            df = pd.read_csv(input_source)
            
            if prompt_column not in df.columns:
                raise ValueError(f"Prompt column '{prompt_column}' not found")
            
            for _, row in df.iterrows():
                # Get optional system prompt
                system_prompt = kwargs.get("system_prompt")
                
                # Create messages array for OpenAI format
                messages = []
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                messages.append({
                    "role": "user",
                    "content": row[prompt_column]
                })
                
                # Prepare item with metadata
                yield_item = {
                    "messages": messages,
                    "metadata": {
                        "row": row.to_dict()
                    }
                }
                
                # Add model parameters if provided
                for param in ["temperature", "max_tokens", "top_p", "stop"]:
                    if param in kwargs:
                        yield_item[param] = kwargs[param]
                
                yield yield_item
                
        except Exception as e:
            logger.error(f"Error processing {input_source}: {e}")
            raise

class DirectoryHandler(InputHandler):
    """Handler for processing files in a directory."""
    
    def process(
        self,
        input_source: str,
        prompt_template: str,
        file_pattern: str = "*",
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process files in directory.
        
        Args:
            input_source: Directory path
            prompt_template: Template string with {content} placeholder
            file_pattern: Glob pattern for files
            **kwargs: Additional parameters for each prompt
            
        Yields:
            Prompts for each file in OpenAI-compatible format
            
        Raises:
            FileNotFoundError: If input directory doesn't exist
        """
        input_dir = Path(input_source)
        
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_source}")
        
        try:
            for file_path in sorted(input_dir.glob(file_pattern)):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        prompt = prompt_template.format(
                            content=content,
                            filename=file_path.name
                        )
                        
                        # Get optional system prompt
                        system_prompt = kwargs.get("system_prompt")
                        
                        # Create messages array for OpenAI format
                        messages = []
                        if system_prompt:
                            messages.append({
                                "role": "system",
                                "content": system_prompt
                            })
                        
                        messages.append({
                            "role": "user",
                            "content": prompt
                        })
                        
                        # Prepare item with metadata
                        yield_item = {
                            "messages": messages,
                            "metadata": {
                                "file": str(file_path),
                                "filename": file_path.name
                            }
                        }
                        
                        # Add model parameters if provided
                        for param in ["temperature", "max_tokens", "top_p", "stop"]:
                            if param in kwargs:
                                yield_item[param] = kwargs[param]
                        
                        yield yield_item
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing directory {input_source}: {e}")
            raise 