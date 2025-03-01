#!/usr/bin/env python3
"""JSON input handler for AI-Flux."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from ..base import InputHandler

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
            input_source: Path to JSON file
            **kwargs: Additional processing options
            
        Yields:
            Dict items with prompt data
            
        Raises:
            FileNotFoundError: If input file does not exist
            ValueError: If JSON is invalid or has unexpected format
        """
        input_path = Path(input_source)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError(f"Expected a list in JSON file, got {type(data)}")
                
            logger.info(f"Loaded {len(data)} items from {input_path}")
            
            for i, item in enumerate(data):
                # Check if item is already in OpenAI format
                if "messages" in item:
                    yield item
                # Otherwise, convert from Ollama format
                elif "prompt" in item:
                    # Convert to OpenAI format
                    messages = [{
                        "role": "user",
                        "content": item.pop("prompt")
                    }]
                    # Add system prompt if present
                    if "system" in item:
                        messages.insert(0, {
                            "role": "system",
                            "content": item.pop("system")
                        })
                    
                    # Create new item with messages
                    item["messages"] = messages
                    yield item
                else:
                    logger.warning(f"Item {i} does not have 'messages' or 'prompt' field, skipping")
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {input_path}: {e}")
            raise ValueError(f"Invalid JSON in file {input_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing JSON file {input_path}: {e}")
            raise 