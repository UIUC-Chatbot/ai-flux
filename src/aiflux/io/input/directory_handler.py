#!/usr/bin/env python3
"""Directory input handler for AI-Flux."""

import logging
import os
import glob
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from ..base import InputHandler

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DirectoryHandler(InputHandler):
    """Handler for processing directories of text files."""
    
    def process(
        self,
        input_source: str,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        file_pattern: str = "*",
        max_file_size: int = 1024 * 1024,  # 1MB default limit
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process directory of text files.
        
        Args:
            input_source: Path to directory
            prompt_template: Template string with {filename} and {content} placeholders
            system_prompt: Optional system prompt to include in messages
            file_pattern: Glob pattern for matching files
            max_file_size: Maximum file size to process in bytes
            **kwargs: Additional processing options
            
        Yields:
            Dict items with messages in OpenAI format
            
        Raises:
            FileNotFoundError: If directory does not exist
            ValueError: If unable to process files
        """
        input_dir = Path(input_source)
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
            
        # Default prompt template if none provided
        if not prompt_template:
            prompt_template = "Process the following file:\n\nFilename: {filename}\n\nContent:\n{content}"
            
        try:
            # Find matching files
            pattern = os.path.join(input_dir, file_pattern)
            files = glob.glob(pattern)
            
            if not files:
                logger.warning(f"No files found matching pattern '{file_pattern}' in {input_dir}")
                return
                
            logger.info(f"Found {len(files)} files in {input_dir} matching pattern '{file_pattern}'")
            
            # Process each file
            for file_path in files:
                file_path = Path(file_path)
                
                try:
                    # Check file size
                    if file_path.stat().st_size > max_file_size:
                        logger.warning(f"Skipping {file_path}: exceeds size limit ({file_path.stat().st_size} > {max_file_size} bytes)")
                        continue
                        
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        
                    # Format prompt
                    formatted_prompt = prompt_template.format(
                        filename=file_path.name,
                        content=content
                    )
                    
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
                    
                    # Create item with metadata
                    item = {
                        "messages": messages,
                        "metadata": {
                            "file": str(file_path),
                            "filename": file_path.name
                        }
                    }
                    
                    yield item
                    
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}, skipping")
                    
        except Exception as e:
            logger.error(f"Error processing directory {input_dir}: {e}")
            raise 