#!/usr/bin/env python3
import json
import logging
import os
from typing import Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)

class ModelExecutor:
    """Generic interface for model execution."""
    def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from the model."""
        raise NotImplementedError("Subclasses must implement generate()")

class OllamaExecutor(ModelExecutor):
    """Implementation of ModelExecutor for Ollama API."""
    
    def __init__(self, host: Optional[str] = None):
        """Initialize Ollama client with optional host."""
        if host is None:
            host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        self.base_url = f"http://{host}"
        self.session = requests.Session()
    
    def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from the Ollama model.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt text
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Dict containing the model's response and metadata
            
        Raises:
            requests.exceptions.RequestException: If the API request fails
            json.JSONDecodeError: If the response cannot be parsed as JSON
        """
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Disable streaming to get a single response
            **kwargs
        }
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            if isinstance(response_data, dict):
                return {
                    'response': response_data.get('response', ''),
                    'metadata': {
                        k: v for k, v in response_data.items() 
                        if k != 'response'
                    }
                }
            return {'response': str(response_data), 'metadata': {}}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {e}")
            return {'response': response.text, 'metadata': {}} 