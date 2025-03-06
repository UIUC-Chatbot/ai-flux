#!/usr/bin/env python3
"""OpenAI-compatible client for Ollama LLM service."""

import os
import logging
import time
import json
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMClient:
    """OpenAI-compatible client for LLM services."""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """Initialize LLM client.
        
        Args:
            host: Optional host address
            port: Optional port number
        """
        if host is None:
            host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        self.base_url = f"http://{host}"
        self.session = requests.Session()
    
    def list_models(self) -> List[str]:
        """List available models using OpenAI-compatible endpoint.
        
        Returns:
            List of available model names
        """
        url = f"{self.base_url}/v1/models"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        return model_name in self.list_models()
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if it doesn't exist.
        
        Note: This uses Ollama's native API since OpenAI doesn't have a pull endpoint.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        if self.model_exists(model_name):
            logger.info(f"Model {model_name} already exists")
            return True
        
        logger.info(f"Pulling model {model_name}...")
        url = f"{self.base_url}/api/pull"
        data = {"name": model_name}
        
        try:
            response = self.session.post(url, json=data, stream=True)
            response.raise_for_status()
            
            # Process streaming response to show progress
            for line in response.iter_lines():
                if line:
                    logger.debug(f"Pull progress: {line.decode('utf-8')}")
            
            # Verify model was pulled successfully
            if self.model_exists(model_name):
                logger.info(f"Successfully pulled model {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def ensure_model_available(self, model_name: str, max_retries: int = 3) -> bool:
        """Ensure a model is available, pulling it if necessary.
        
        Args:
            model_name: Name of the model to ensure
            max_retries: Maximum number of pull attempts
            
        Returns:
            True if model is available, False otherwise
        """
        # Check if model exists
        if self.model_exists(model_name):
            return True
        
        # Try to pull the model
        for attempt in range(max_retries):
            logger.info(f"Attempting to pull model {model_name} (attempt {attempt + 1}/{max_retries})")
            if self.pull_model(model_name):
                return True
            
            # Wait before retrying
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
        
        return False
    
    def generate(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Generate response using OpenAI-compatible chat completions endpoint.
        
        Args:
            model: Name of the model to use
            messages: Array of messages in OpenAI format
            **kwargs: Additional model parameters:
                - temperature: float
                - top_p: float
                - max_tokens: int
                - stop: List[str]
            
        Returns:
            Model response
            
        Raises:
            requests.exceptions.RequestException: If API call fails
            ValueError: If model is not available
        """
        # Ensure model is available
        if not self.ensure_model_available(model):
            error_msg = f"Model {model} is not available and could not be pulled"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        url = f"{self.base_url}/v1/chat/completions"
        
        # Create payload in OpenAI format
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        
        # Add other parameters if provided
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "stop" in kwargs and kwargs["stop"]:
            payload["stop"] = kwargs["stop"]
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            # Parse OpenAI response format
            response_data = response.json()
            
            # Extract the content from the response
            if (
                'choices' in response_data and 
                len(response_data['choices']) > 0 and
                'message' in response_data['choices'][0] and
                'content' in response_data['choices'][0]['message']
            ):
                return response_data['choices'][0]['message']['content']
            else:
                logger.warning(f"Unexpected response format: {response_data}")
                return ""
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error decoding response: {e}")
            return response.text 