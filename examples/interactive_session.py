#!/usr/bin/env python3
"""
Example of using the interactive handler to create an interactive LLM session.

This script demonstrates how to set up an interactive session using Cloudflare tunnels
that allows users to interact with the LLM through a web interface during the job duration.

Usage:
    python interactive_session.py --model qwen:7b --time 02:00:00

For custom domain with subdomain pattern:
    python interactive_session.py --model qwen:7b --time 02:00:00 \
        --domain ncsa.ai --subdomain-pattern "{account}.{model}" \
        --account myaccount --api-token-file ~/.cloudflare/token.txt

This will start an interactive session with the specified model that will run for the
specified time. The script will output a URL that can be used to access the web interface.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import aiflux
sys.path.insert(0, str(Path(__file__).parent.parent))

from aiflux import (
    InteractiveProcessor,
    SlurmRunner,
    Config,
    InteractiveHandler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the example."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Interactive LLM Session Example')
    parser.add_argument('--model', type=str, required=True, help='Model to use (e.g., qwen:7b)')
    parser.add_argument('--account', type=str, help='SLURM account')
    parser.add_argument('--partition', type=str, help='SLURM partition')
    parser.add_argument('--time', type=str, help='SLURM time limit')
    parser.add_argument('--memory', type=str, help='SLURM memory per node')
    
    # Cloudflare tunnel options
    parser.add_argument('--tunnel-name', type=str, help='Cloudflare tunnel name')
    parser.add_argument('--domain', type=str, help='Domain for the tunnel (e.g., ncsa.ai)')
    parser.add_argument('--subdomain-pattern', type=str, help='Pattern for subdomain (e.g., "{account}.{model}")')
    parser.add_argument('--account-name', type=str, help='Account name to use in subdomain pattern')
    parser.add_argument('--api-token-file', type=str, help='Path to Cloudflare API token file')
    parser.add_argument('--config-file', type=str, help='Path to Cloudflare config file')
    parser.add_argument('--port', type=int, default=8000, help='Port for the web server')
    
    args = parser.parse_args()
    
    # Parse model name
    if ':' not in args.model:
        logger.error("Model name must be in format 'type:size' (e.g., qwen:7b)")
        sys.exit(1)
    
    model_type, model_size = args.model.split(':')
    
    # Load configuration
    config = Config()
    try:
        model_config = config.load_model_config(model_type, model_size)
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        sys.exit(1)
    
    # Create SLURM configuration with overrides
    slurm_overrides = {}
    if args.account:
        slurm_overrides['account'] = args.account
    if args.partition:
        slurm_overrides['partition'] = args.partition
    if args.time:
        slurm_overrides['time'] = args.time
    if args.memory:
        slurm_overrides['memory'] = args.memory
    
    # Create interactive handler
    handler_kwargs = {
        'model': args.model  # Pass model name for subdomain pattern
    }
    
    # Add Cloudflare tunnel options if provided
    if args.tunnel_name:
        handler_kwargs['tunnel_name'] = args.tunnel_name
    if args.domain:
        handler_kwargs['tunnel_domain'] = args.domain
    if args.subdomain_pattern:
        handler_kwargs['subdomain_pattern'] = args.subdomain_pattern
    if args.account_name:
        handler_kwargs['account_name'] = args.account_name
    if args.api_token_file:
        handler_kwargs['api_token_file'] = args.api_token_file
    if args.config_file:
        handler_kwargs['config_file'] = args.config_file
    
    # Try to load default config file if exists and not explicitly provided
    if not args.config_file:
        default_config = Path.home() / '.cloudflare' / 'config.json'
        if default_config.exists():
            handler_kwargs['config_file'] = str(default_config)
            logger.info(f"Using default Cloudflare config: {default_config}")
    
    # Try to load default API token if exists and not explicitly provided
    if not args.api_token_file:
        default_token = Path.home() / '.cloudflare' / 'token.txt'
        if default_token.exists():
            handler_kwargs['api_token_file'] = str(default_token)
            logger.info(f"Using default Cloudflare API token: {default_token}")
    
    input_handler = InteractiveHandler()
    
    # Create interactive processor
    processor = InteractiveProcessor(
        model_config=model_config,
        input_handler=input_handler,
        port=args.port
    )
    
    # Create SLURM runner
    runner = SlurmRunner(
        config=config.get_slurm_config(slurm_overrides)
    )
    
    # Run the interactive session
    logger.info(f"Starting interactive session with model {args.model}")
    runner.run(
        processor=processor,
        input_source=None,  # No input source needed for interactive sessions
        **handler_kwargs
    )
    
    logger.info("Interactive session started")
    logger.info("Check the SLURM job output for the session URL")

if __name__ == "__main__":
    main() 