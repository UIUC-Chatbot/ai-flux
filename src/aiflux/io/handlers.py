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
            df = pd.read_csv(input_source)
            
            for _, row in df.iterrows():
                # Format prompt template with row data
                try:
                    prompt = prompt_template.format(**row.to_dict())
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

class InteractiveHandler(InputHandler):
    """Handler for interactive sessions using Cloudflare tunnels.
    
    This handler creates an interactive endpoint that users can access
    during the job duration to interact with the LLM in real-time.
    """
    
    def process(
        self,
        input_source: Union[str, Path],
        tunnel_name: Optional[str] = None,
        tunnel_domain: Optional[str] = None,
        api_token_file: Optional[str] = None,
        subdomain_pattern: Optional[str] = None,
        account_name: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Set up an interactive endpoint using Cloudflare tunnels.
        
        Args:
            input_source: Path to configuration file or directory (optional)
            tunnel_name: Name for the Cloudflare tunnel (default: auto-generated)
            tunnel_domain: Domain for the tunnel (default: from config)
            api_token_file: Path to Cloudflare API token file (default: from env)
            subdomain_pattern: Pattern for subdomain (e.g., "{account}.{model}.ncsa.ai")
            account_name: Account name to use in subdomain pattern
            config_file: Path to Cloudflare config file
            **kwargs: Additional parameters for the interactive session
            
        Yields:
            Configuration for the interactive session
        """
        # This handler doesn't yield actual prompts but configuration
        # for the interactive session. The actual processing is handled
        # by a web server that's started as part of the job.
        
        # Generate a unique session ID
        import uuid
        import time
        session_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # Default tunnel name if not provided
        if not tunnel_name:
            tunnel_name = f"aiflux-{timestamp}-{session_id[:8]}"
        
        # Configuration for the interactive session
        config = {
            "session_id": session_id,
            "timestamp": timestamp,
            "tunnel_name": tunnel_name,
            "tunnel_domain": tunnel_domain,
            "api_token_file": api_token_file,
            "subdomain_pattern": subdomain_pattern,
            "account_name": account_name,
            "config_file": config_file,
            "interactive": True,
            **kwargs
        }
        
        # Try to load Cloudflare config file if provided
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    cf_config = json.load(f)
                    
                # Apply config values if not already set
                if not tunnel_domain and "domain" in cf_config:
                    config["tunnel_domain"] = cf_config["domain"]
                if not subdomain_pattern and "subdomain_pattern" in cf_config:
                    config["subdomain_pattern"] = cf_config["subdomain_pattern"]
                if not tunnel_name and "tunnel_name_prefix" in cf_config:
                    prefix = cf_config["tunnel_name_prefix"]
                    config["tunnel_name"] = f"{prefix}-{timestamp}-{session_id[:8]}"
            except Exception as e:
                logger.error(f"Error loading Cloudflare config: {e}")
        
        # If subdomain pattern is provided, format it with account and model
        if config.get("subdomain_pattern") and config.get("tunnel_domain"):
            model_name = kwargs.get("model", "model").replace(":", "-")
            account = config.get("account_name", "user")
            
            # Format the subdomain pattern
            pattern = config["subdomain_pattern"]
            subdomain = pattern.format(account=account, model=model_name, session=session_id[:8])
            
            # Combine with domain
            domain = config["tunnel_domain"]
            if not domain.startswith("http"):
                domain = f"https://{domain}"
                
            # Remove domain from subdomain if included
            if domain.replace("https://", "") in subdomain:
                config["tunnel_domain"] = f"https://{subdomain}"
            else:
                config["tunnel_domain"] = f"https://{subdomain}.{domain.replace('https://', '')}"
        
        # If input_source is provided, try to load additional config
        if input_source:
            input_path = Path(input_source)
            if input_path.is_file():
                try:
                    with open(input_path, 'r') as f:
                        if input_path.suffix.lower() == '.json':
                            user_config = json.load(f)
                        elif input_path.suffix.lower() in ['.yaml', '.yml']:
                            import yaml
                            user_config = yaml.safe_load(f)
                        else:
                            user_config = {}
                            
                        # Update config with user-provided values
                        config.update(user_config)
                except Exception as e:
                    logger.error(f"Error loading config from {input_path}: {e}")
        
        # Yield the configuration for the interactive session
        yield config 