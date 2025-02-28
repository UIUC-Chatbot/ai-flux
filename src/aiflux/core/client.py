#!/usr/bin/env python3
import os
import logging
import time
from typing import Dict, Any, Optional, List
import requests

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMClient:
    """Base class for LLM clients."""
    
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
        """List available models.
        
        Returns:
            List of available model names
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
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
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response from the model.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional model parameters
            
        Returns:
            Model response
            
        Raises:
            requests.exceptions.RequestException: If API call fails
            json.JSONDecodeError: If response parsing fails
            ValueError: If model is not available
        """
        # Ensure model is available
        if not self.ensure_model_available(model):
            error_msg = f"Model {model} is not available and could not be pulled"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        url = f"{self.base_url}/api/generate"
        
        # Format prompt with system prompt if provided
        if system_prompt:
            formatted_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            formatted_prompt = prompt
        
        data = {
            "model": model,
            "prompt": formatted_prompt,
            "stream": False,  # Disable streaming for batch processing
            **kwargs
        }
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            if isinstance(response_data, dict):
                return response_data.get('response', '')
            return str(response_data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error decoding response: {e}")
            return response.text 