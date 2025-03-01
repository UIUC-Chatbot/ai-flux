#!/usr/bin/env python3
"""Module for handling image data processing with AI-Flux."""

import base64
import json
import logging
import os
from pathlib import Path
import requests
from typing import Dict, Any, List, Iterator, Union, Optional
import mimetypes

from .base import InputHandler

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
            prompts_file: Optional JSON file path containing prompts for specific images.
                          Format: {"image_filename.jpg": "custom prompt for this image", ...}
                          If an image isn't in this map, the default prompt_template is used.
        """
        super().__init__()
        self.prompt_template = prompt_template or "Analyze this image in detail."
        self.image_prompts = {}
        
        # Load custom prompts from file if provided
        if prompts_file:
            try:
                with open(prompts_file, 'r') as f:
                    self.image_prompts = json.load(f)
                logger.info(f"Loaded {len(self.image_prompts)} custom prompts from {prompts_file}")
            except Exception as e:
                logger.error(f"Error loading prompts file {prompts_file}: {e}")
    
    def process(self, input_source: str, file_pattern: str = "*.[jp][pn]g", **kwargs) -> Iterator[Dict[str, Any]]:
        """Process image files in a directory or a single image file.
        
        Args:
            input_source: Path to directory containing images or path to a single image file
            file_pattern: File pattern for matching image files (default: images with jpg/png extension)
            **kwargs: Additional parameters:
                - prompts_map: Dict mapping filenames to custom prompts
                - system_prompt: Optional system prompt to include
            
        Yields:
            Dict containing the prompt data formatted for OpenAI-compatible vision API
            
        Raises:
            FileNotFoundError: If the input source does not exist
        """
        input_path = Path(input_source)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input source does not exist: {input_source}")
        
        # Allow overriding prompts from kwargs
        prompts_map = {**self.image_prompts, **(kwargs.get('prompts_map', {}))}
        system_prompt = kwargs.get('system_prompt')
        
        if input_path.is_dir():
            # Process all images in directory matching the pattern
            logger.info(f"Processing images in directory: {input_source}")
            for img_path in sorted(input_path.glob(file_pattern)):
                try:
                    # Get custom prompt for this image if available
                    prompt_text = prompts_map.get(img_path.name, self.prompt_template)
                    yield self.process_image(img_path, prompt_text, system_prompt)
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
        else:
            # Process a single image file
            logger.info(f"Processing single image: {input_source}")
            try:
                # Get custom prompt for this image if available
                prompt_text = prompts_map.get(input_path.name, self.prompt_template)
                yield self.process_image(input_path, prompt_text, system_prompt)
            except Exception as e:
                logger.error(f"Error processing image {input_source}: {e}")
    
    def process_image(self, image_path: Path, prompt_text: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process a single image file and prepare it for the vision model.
        
        Args:
            image_path: Path to the image file
            prompt_text: Text prompt to accompany the image
            system_prompt: Optional system prompt to include
            
        Returns:
            Dict containing the formatted prompt for OpenAI-compatible vision API
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be encoded
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        try:
            # Read and encode the image
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
            
            # Encode the image to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine the MIME type based on file extension
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type:
                # Default to a general image type if we can't determine
                mime_type = "image/jpeg" if image_path.suffix.lower() in ['.jpg', '.jpeg'] else "image/png"
            
            # Format in OpenAI-compatible vision API format
            # In OpenAI's format, we need to use a structured messages array with text and image content
            message_content = [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}"
                    }
                }
            ]
            
            # The full request structure for OpenAI vision API compatibility
            result = {
                "messages": [],
                "metadata": {
                    "filename": image_path.name,
                    "path": str(image_path)
                }
            }
            
            # Add system message if provided
            if system_prompt:
                result["messages"].append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user message with text and image
            result["messages"].append({
                "role": "user", 
                "content": message_content
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise ValueError(f"Failed to process image {image_path}: {e}")
    
    def __str__(self) -> str:
        """Return a string representation of the handler."""
        return f"VisionHandler(prompt_template='{self.prompt_template}', custom_prompts={len(self.image_prompts)})" 