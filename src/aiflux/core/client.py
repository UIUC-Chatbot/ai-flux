#!/usr/bin/env python3
import os
import logging
from typing import Dict, Any, Optional
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
        """
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