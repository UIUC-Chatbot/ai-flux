#!/usr/bin/env python3
"""Vision input handler for AI-Flux."""

import base64
import json
import logging
import os
import glob
from pathlib import Path
import mimetypes
from typing import Dict, Any, List, Iterator, Union, Optional

from ..base import InputHandler

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VisionHandler(InputHandler):
    """Handler for processing image files with vision models."""
    
    def __init__(self, prompt_template: Optional[str] = None, prompts_file: Optional[str] = None):
        """Initialize vision handler.
        
        Args:
            prompt_template: Optional template for prompting the vision model.
                             Defaults to a simple image analysis prompt.
            prompts_file: Optional path to a JSON file with custom prompts for each image.
                          Format should be a dictionary mapping image filenames to prompts.
        """
        self.prompt_template = prompt_template or "Analyze this image and describe what you see."
        self.custom_prompts = {}
        
        # Load custom prompts if provided
        if prompts_file:
            try:
                with open(prompts_file, 'r') as f:
                    self.custom_prompts = json.load(f)
                logger.info(f"Loaded {len(self.custom_prompts)} custom prompts from {prompts_file}")
            except Exception as e:
                logger.error(f"Failed to load custom prompts from {prompts_file}: {e}")
    
    def process(
        self,
        input_source: str,
        file_pattern: str = "*.jpg;*.jpeg;*.png",
        system_prompt: Optional[str] = None,
        prompts_map: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Process a directory of images or a single image file.
        
        Args:
            input_source: Path to directory or image file
            file_pattern: Glob pattern for matching image files (semicolon-separated for multiple)
            system_prompt: Optional system prompt to provide context for image analysis
            prompts_map: Optional dictionary mapping filenames to custom prompts
            **kwargs: Additional processing parameters
            
        Yields:
            Dict items with messages in OpenAI format including image data
            
        Raises:
            FileNotFoundError: If input directory or file does not exist
        """
        input_path = Path(input_source)
        
        # Combine file-specific prompts from all sources
        all_prompts = {}
        if prompts_map:
            all_prompts.update(prompts_map)
        if self.custom_prompts:
            all_prompts.update(self.custom_prompts)
            
        # Process directory of images
        if input_path.is_dir():
            # Split multiple patterns
            patterns = file_pattern.split(';')
            all_files = []
            
            # Get files for each pattern
            for pattern in patterns:
                all_files.extend(glob.glob(str(input_path / pattern.strip())))
                
            if not all_files:
                logger.warning(f"No image files found matching pattern '{file_pattern}' in {input_path}")
                return
                
            logger.info(f"Found {len(all_files)} image files in {input_path}")
            
            # Process each image
            for image_path in all_files:
                try:
                    image_path = Path(image_path)
                    filename = image_path.name
                    
                    # Get prompt for this image - custom prompt or default template
                    prompt = all_prompts.get(filename, self.prompt_template)
                    
                    yield self.process_image(
                        image_path, 
                        prompt=prompt,
                        system_prompt=system_prompt,
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    
        # Process single image file
        elif input_path.is_file():
            try:
                filename = input_path.name
                
                # Get prompt for this image - custom prompt or default template
                prompt = all_prompts.get(filename, self.prompt_template)
                
                yield self.process_image(
                    input_path, 
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error processing image {input_path}: {e}")
                
        else:
            raise FileNotFoundError(f"Image source not found: {input_path}")
            
    def process_image(
        self, 
        image_path: Union[str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single image file.
        
        Args:
            image_path: Path to image file
            prompt: Prompt text for image analysis
            system_prompt: Optional system prompt to provide context
            **kwargs: Additional parameters
            
        Returns:
            Dict with image data in OpenAI format
            
        Raises:
            FileNotFoundError: If image file does not exist
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Get mime type
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Default to JPEG if unknown
            
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
        # Build OpenAI-compatible message format with image
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        # Add user message with text and image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]
        })
        
        # Build request with messages
        request = {
            "messages": messages,
            "metadata": {
                "filename": image_path.name,
                "path": str(image_path)
            }
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ["messages", "metadata"]:
                request[key] = value
                
        return request 