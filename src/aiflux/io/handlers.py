#!/usr/bin/env python3
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InputHandler(ABC):
    """Base class for input handlers."""
    
    @abstractmethod
    def process(
        self,
        input_source: Union[str, Path],
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process input source and yield prompts/data.
        
        Args:
            input_source: Path to input source
            **kwargs: Additional processing parameters
            
        Yields:
            Processed input items
        """
        pass

class JSONBatchHandler(InputHandler):
    """Handler for JSON files containing multiple prompts."""
    
    def process(
        self,
        input_source: Union[str, Path],
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process JSON input file.
        
        Args:
            input_source: Path to JSON file
            **kwargs: Unused
            
        Yields:
            Input items from JSON
        """
        input_path = Path(input_source)
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                yield data
                
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            raise

class CSVSinglePromptHandler(InputHandler):
    """Handler for running same prompt on CSV rows."""
    
    def process(
        self,
        input_source: Union[str, Path],
        prompt_template: str,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process CSV input file.
        
        Args:
            input_source: Path to CSV file
            prompt_template: Template string with {field} placeholders
            **kwargs: Additional parameters for prompt formatting
            
        Yields:
            Formatted prompts for each row
        """
        try:
            #logger.info(f"Processing {input_source} with template: {prompt_template}")
            df = pd.read_csv(input_source)
            
            for _, row in df.iterrows():
                # Format prompt template with row data
                logger.info(f"Row: {row}")
                try:
                    prompt = prompt_template.format(**row.to_dict())
                    logger.info(f"Prompt in CSVSinglePromptHandler: {prompt}")
                    yield {
                        "prompt": prompt,
                        **kwargs
                    }
                except KeyError as e:
                    logger.error(f"Missing field in template: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error processing {input_source}: {e}")
            raise

class CSVMultiPromptHandler(InputHandler):
    """Handler for CSV files where each row contains a prompt."""
    
    def process(
        self,
        input_source: Union[str, Path],
        prompt_column: str = "prompt",
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process CSV input file.
        
        Args:
            input_source: Path to CSV file
            prompt_column: Name of column containing prompts
            **kwargs: Additional parameters for each prompt
            
        Yields:
            Prompts from each row
        """
        try:
            df = pd.read_csv(input_source)
            
            if prompt_column not in df.columns:
                raise ValueError(f"Prompt column '{prompt_column}' not found")
            
            for _, row in df.iterrows():
                yield {
                    "prompt": row[prompt_column],
                    **kwargs
                }
                
        except Exception as e:
            logger.error(f"Error processing {input_source}: {e}")
            raise

class DirectoryHandler(InputHandler):
    """Handler for processing files in a directory."""
    
    def process(
        self,
        input_source: Union[str, Path],
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
            Prompts for each file
        """
        input_dir = Path(input_source)
        
        try:
            for file_path in input_dir.glob(file_pattern):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        prompt = prompt_template.format(
                            content=content,
                            filename=file_path.name
                        )
                        
                        yield {
                            "prompt": prompt,
                            "file": str(file_path),
                            **kwargs
                        }
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing directory {input_source}: {e}")
            raise 