#!/usr/bin/env python3
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.processor import BaseProcessor
from ..core.client import LLMClient
from ..core.config import ModelConfig
from ..io.handlers import InteractiveHandler
from ..io.output import OutputHandler, JSONOutputHandler, TimestampedOutputHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InteractiveProcessor(BaseProcessor):
    """Processor for interactive LLM sessions using Cloudflare tunnels."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        input_handler: Optional[InteractiveHandler] = None,
        output_handler: Optional[OutputHandler] = None,
        port: int = 8000
    ):
        """Initialize interactive processor.
        
        Args:
            model_config: Model configuration
            input_handler: Handler for interactive session
            output_handler: Handler for saving outputs
            port: Port for the web server
        """
        super().__init__(model_config.name)
        self.config = model_config
        self.input_handler = input_handler or InteractiveHandler()
        self.output_handler = TimestampedOutputHandler(
            output_handler or JSONOutputHandler()
        )
        self.port = port
        self.client = None
        self.server_process = None
        self.cloudflared_process = None
        self.session_config = None
        self.endpoint_url = None
        self.stop_event = threading.Event()
        
    def setup(self) -> None:
        """Setup processor by initializing client and web server."""
        self.client = LLMClient()
        
        # Warm up the model
        logger.info("Warming up model...")
        try:
            self.client.generate(
                model=self.model,
                prompt="warming up the model",
                system_prompt=self.config.system.prompt
            )
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.error(f"Error warming up model: {e}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        self.stop_event.set()
        
        # Stop the web server
        if self.server_process:
            logger.info("Stopping web server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
        
        # Stop the cloudflared tunnel
        if self.cloudflared_process:
            logger.info("Stopping Cloudflare tunnel...")
            try:
                self.cloudflared_process.terminate()
                self.cloudflared_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.cloudflared_process.kill()
            self.cloudflared_process = None
        
        # Close the client session
        if self.client:
            self.client.session.close()
            self.client = None
    
    def process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of inputs.
        
        For interactive sessions, this method is used to set up the
        web server and Cloudflare tunnel.
        
        Args:
            batch: List of configuration items
            
        Returns:
            List of session information
        """
        if not batch:
            logger.error("No configuration provided for interactive session")
            return []
        
        # Use the first item as the session configuration
        self.session_config = batch[0]
        
        # Start the web server
        self._start_web_server()
        
        # Start the Cloudflare tunnel
        self._start_cloudflare_tunnel()
        
        # Return session information
        return [{
            "session_id": self.session_config.get("session_id"),
            "endpoint_url": self.endpoint_url,
            "model": self.model,
            "timestamp": time.time(),
            "status": "running"
        }]
    
    def _start_web_server(self) -> None:
        """Start the web server for the interactive session."""
        logger.info(f"Starting web server on port {self.port}...")
        
        # Create a temporary directory for the web server
        server_dir = Path(os.environ.get('LOGS_DIR', '.')) / 'server'
        server_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the web server script
        server_script = server_dir / "server.py"
        with open(server_script, 'w') as f:
            f.write(self._get_server_script())
        
        # Start the web server
        cmd = [
            "python", str(server_script),
            "--port", str(self.port),
            "--model", self.model,
            "--session-id", self.session_config.get("session_id", "unknown")
        ]
        
        # Add system prompt if available
        if hasattr(self.config, 'system') and hasattr(self.config.system, 'prompt'):
            cmd.extend(["--system-prompt", self.config.system.prompt])
        
        # Start the server process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        for _ in range(30):
            if self.server_process.poll() is not None:
                # Server process exited
                stdout, stderr = self.server_process.communicate()
                logger.error(f"Web server failed to start: {stderr}")
                raise RuntimeError(f"Web server failed to start: {stderr}")
            
            # Check if the server is running
            try:
                import requests
                response = requests.get(f"http://localhost:{self.port}/health")
                if response.status_code == 200:
                    logger.info(f"Web server started successfully on port {self.port}")
                    break
            except Exception:
                pass
            
            time.sleep(1)
        else:
            logger.error("Timed out waiting for web server to start")
            raise TimeoutError("Timed out waiting for web server to start")
    
    def _start_cloudflare_tunnel(self) -> None:
        """Start the Cloudflare tunnel for the interactive session."""
        logger.info("Starting Cloudflare tunnel...")
        
        # Get tunnel configuration
        tunnel_name = self.session_config.get("tunnel_name")
        tunnel_domain = self.session_config.get("tunnel_domain")
        api_token_file = self.session_config.get("api_token_file")
        
        if not tunnel_name:
            logger.error("No tunnel name provided")
            raise ValueError("No tunnel name provided")
        
        # Check if cloudflared is installed
        try:
            subprocess.run(
                ["cloudflared", "version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("cloudflared not found. Please install it first.")
            raise RuntimeError("cloudflared not found. Please install it first.")
        
        # Create a temporary directory for cloudflared
        tunnel_dir = Path(os.environ.get('LOGS_DIR', '.')) / 'cloudflared'
        tunnel_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up environment for cloudflared
        env = os.environ.copy()
        if api_token_file:
            try:
                with open(api_token_file, 'r') as f:
                    api_token = f.read().strip()
                env["CLOUDFLARE_API_TOKEN"] = api_token
            except Exception as e:
                logger.error(f"Error reading API token file: {e}")
                logger.warning("Proceeding without API token, will use quick tunnel")
        
        # Determine if we're using a named tunnel or quick tunnel
        using_named_tunnel = api_token_file and tunnel_domain
        
        if using_named_tunnel:
            # Using a named tunnel with custom domain
            logger.info(f"Setting up named tunnel '{tunnel_name}' with domain '{tunnel_domain}'")
            
            # Create the tunnel if it doesn't exist
            try:
                # Check if tunnel exists
                result = subprocess.run(
                    ["cloudflared", "tunnel", "list"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                tunnel_exists = tunnel_name in result.stdout
                
                if not tunnel_exists:
                    # Create the tunnel
                    logger.info(f"Creating tunnel '{tunnel_name}'...")
                    subprocess.run(
                        ["cloudflared", "tunnel", "create", tunnel_name],
                        env=env,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    logger.info(f"Tunnel '{tunnel_name}' created successfully")
                else:
                    logger.info(f"Tunnel '{tunnel_name}' already exists")
                
                # Create DNS record
                hostname = tunnel_domain.replace("https://", "").replace("http://", "")
                logger.info(f"Creating DNS record for '{hostname}'...")
                
                subprocess.run(
                    ["cloudflared", "tunnel", "route", "dns", tunnel_name, hostname],
                    env=env,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info(f"DNS record for '{hostname}' created successfully")
                
                # Start the tunnel
                cmd = [
                    "cloudflared", "tunnel", "run",
                    "--url", f"http://localhost:{self.port}",
                    tunnel_name
                ]
                
                self.endpoint_url = tunnel_domain
                if not self.endpoint_url.startswith("http"):
                    self.endpoint_url = f"https://{self.endpoint_url}"
                
            except subprocess.SubprocessError as e:
                logger.error(f"Error setting up named tunnel: {e}")
                logger.warning("Falling back to quick tunnel")
                using_named_tunnel = False
        
        if not using_named_tunnel:
            # Use quick tunnel
            logger.info("Using quick tunnel (no custom domain)")
            cmd = [
                "cloudflared", "tunnel",
                "--url", f"http://localhost:{self.port}",
                "--metrics", "localhost:8099",
                "--no-autoupdate",
                "--quick-tunnel"
            ]
        
        # Start the cloudflared process
        self.cloudflared_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # Wait for the tunnel to start and extract the URL
        for _ in range(60):  # Increased timeout for named tunnels
            if self.cloudflared_process.poll() is not None:
                # Cloudflared process exited
                stdout, stderr = self.cloudflared_process.communicate()
                logger.error(f"Cloudflare tunnel failed to start: {stderr}")
                raise RuntimeError(f"Cloudflare tunnel failed to start: {stderr}")
            
            # Check the output for the tunnel URL
            line = self.cloudflared_process.stdout.readline().strip()
            
            # For named tunnels, we already know the URL
            if using_named_tunnel and "connection" in line.lower() and "registered" in line.lower():
                logger.info(f"Named tunnel started successfully")
                break
                
            # For quick tunnels, extract the URL from output
            elif not using_named_tunnel and "https://" in line and "trycloudflare.com" in line:
                import re
                match = re.search(r'(https://[^\s]+)', line)
                if match:
                    self.endpoint_url = match.group(1)
                    logger.info(f"Quick tunnel started: {self.endpoint_url}")
                    break
            
            time.sleep(1)
        else:
            if not self.endpoint_url:
                logger.error("Timed out waiting for Cloudflare tunnel to start")
                raise TimeoutError("Timed out waiting for Cloudflare tunnel to start")
    
    def _get_server_script(self) -> str:
        """Generate the web server script for the interactive session."""
        return """#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description='Interactive LLM Server')
parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--model', type=str, required=True, help='Model to use')
parser.add_argument('--session-id', type=str, required=True, help='Session ID')
parser.add_argument('--system-prompt', type=str, default='', help='System prompt')
args = parser.parse_args()

# Create FastAPI app
app = FastAPI(title="AI-Flux Interactive LLM")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for chat requests
class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1024
    stop: Optional[list] = None

# Model for chat responses
class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: str
    timestamp: float

# Get Ollama host from environment
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'localhost:11434')
if not OLLAMA_HOST.startswith('http'):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Prepare request to Ollama
        url = f"{OLLAMA_HOST}/api/generate"
        
        # Format prompt with system prompt if provided
        system_prompt = request.system_prompt or args.system_prompt
        if system_prompt:
            formatted_prompt = (
                f"<|im_start|>system\\n{system_prompt}<|im_end|>\\n"
                f"<|im_start|>user\\n{request.prompt}<|im_end|>\\n"
                f"<|im_start|>assistant\\n"
            )
        else:
            formatted_prompt = request.prompt
        
        # Send request to Ollama
        ollama_request = {
            "model": args.model,
            "prompt": formatted_prompt,
            "stream": False,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": request.stop
        }
        
        response = requests.post(url, json=ollama_request)
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        if isinstance(response_data, dict):
            result = response_data.get('response', '')
        else:
            result = str(response_data)
        
        # Return response
        return ChatResponse(
            response=result,
            model=args.model,
            session_id=args.session_id,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
@app.get("/", response_class=HTMLResponse)
async def root():
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI-Flux Interactive LLM</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <header class="mb-8">
                <h1 class="text-3xl font-bold text-gray-800">AI-Flux Interactive LLM</h1>
                <p class="text-gray-600">Session ID: {args.session_id}</p>
                <p class="text-gray-600">Model: {args.model}</p>
            </header>
            
            <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                <div id="chat-history" class="mb-6 space-y-4 max-h-96 overflow-y-auto"></div>
                
                <div class="flex flex-col space-y-4">
                    <textarea 
                        id="prompt-input" 
                        class="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        rows="4"
                        placeholder="Enter your prompt here..."
                    ></textarea>
                    
                    <div class="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
                        <div class="w-full md:w-1/3">
                            <label class="block text-sm font-medium text-gray-700 mb-1">Temperature</label>
                            <input 
                                type="range" 
                                id="temperature" 
                                min="0" 
                                max="1" 
                                step="0.1" 
                                value="0.7"
                                class="w-full"
                            >
                            <div class="flex justify-between text-xs text-gray-500">
                                <span>0</span>
                                <span id="temperature-value">0.7</span>
                                <span>1</span>
                            </div>
                        </div>
                        
                        <div class="w-full md:w-1/3">
                            <label class="block text-sm font-medium text-gray-700 mb-1">Top P</label>
                            <input 
                                type="range" 
                                id="top-p" 
                                min="0" 
                                max="1" 
                                step="0.1" 
                                value="0.9"
                                class="w-full"
                            >
                            <div class="flex justify-between text-xs text-gray-500">
                                <span>0</span>
                                <span id="top-p-value">0.9</span>
                                <span>1</span>
                            </div>
                        </div>
                        
                        <div class="w-full md:w-1/3">
                            <label class="block text-sm font-medium text-gray-700 mb-1">Max Tokens</label>
                            <input 
                                type="number" 
                                id="max-tokens" 
                                min="1" 
                                max="4096" 
                                value="1024"
                                class="w-full p-2 border border-gray-300 rounded-lg"
                            >
                        </div>
                    </div>
                    
                    <button 
                        id="submit-btn"
                        class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                    >
                        Send
                    </button>
                </div>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', () => {
                const chatHistory = document.getElementById('chat-history');
                const promptInput = document.getElementById('prompt-input');
                const submitBtn = document.getElementById('submit-btn');
                const temperatureSlider = document.getElementById('temperature');
                const temperatureValue = document.getElementById('temperature-value');
                const topPSlider = document.getElementById('top-p');
                const topPValue = document.getElementById('top-p-value');
                const maxTokensInput = document.getElementById('max-tokens');
                
                // Update slider values
                temperatureSlider.addEventListener('input', () => {
                    temperatureValue.textContent = temperatureSlider.value;
                });
                
                topPSlider.addEventListener('input', () => {
                    topPValue.textContent = topPSlider.value;
                });
                
                // Submit prompt
                submitBtn.addEventListener('click', async () => {
                    const prompt = promptInput.value.trim();
                    if (!prompt) return;
                    
                    // Add user message to chat
                    addMessage('user', prompt);
                    
                    // Clear input
                    promptInput.value = '';
                    
                    // Disable submit button
                    submitBtn.disabled = true;
                    submitBtn.textContent = 'Generating...';
                    
                    try {
                        // Send request to API
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                prompt,
                                temperature: parseFloat(temperatureSlider.value),
                                top_p: parseFloat(topPSlider.value),
                                max_tokens: parseInt(maxTokensInput.value)
                            })
                        });
                        
                        if (!response.ok) {
                            throw new Error('Failed to generate response');
                        }
                        
                        const data = await response.json();
                        
                        // Add assistant message to chat
                        addMessage('assistant', data.response);
                    } catch (error) {
                        console.error('Error:', error);
                        addMessage('error', 'An error occurred while generating a response.');
                    } finally {
                        // Re-enable submit button
                        submitBtn.disabled = false;
                        submitBtn.textContent = 'Send';
                    }
                });
                
                // Handle Enter key
                promptInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        submitBtn.click();
                    }
                });
                
                // Add message to chat
                function addMessage(role, content) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'p-4 rounded-lg ' + 
                        (role === 'user' ? 'bg-blue-100 ml-12' : 
                         role === 'assistant' ? 'bg-gray-100 mr-12' : 
                         'bg-red-100 text-red-800');
                    
                    const roleSpan = document.createElement('div');
                    roleSpan.className = 'font-semibold mb-2';
                    roleSpan.textContent = role === 'user' ? 'You' : 
                                          role === 'assistant' ? 'AI' : 'Error';
                    
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'whitespace-pre-wrap';
                    contentDiv.textContent = content;
                    
                    messageDiv.appendChild(roleSpan);
                    messageDiv.appendChild(contentDiv);
                    chatHistory.appendChild(messageDiv);
                    
                    // Scroll to bottom
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
            });
        </script>
    </body>
    </html>
    '''

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
"""
    
    def process_all(
        self,
        input_source: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Process all inputs and start the interactive session.
        
        Args:
            input_source: Source of input data
            output_path: Path to save results
            **kwargs: Additional parameters for input handler
        """
        try:
            # Setup processor
            self.setup()
            
            # Process configuration
            logger.info(f"Setting up interactive session from {input_source}")
            config_items = list(self.input_handler.process(input_source, **kwargs))
            
            # Start the interactive session
            results = self.process_batch(config_items)
            
            # Save session information
            self.output_handler.save(results, output_path)
            
            # Print endpoint information
            if self.endpoint_url:
                logger.info(f"Interactive session available at: {self.endpoint_url}")
                logger.info(f"Session will remain active until the job ends")
            
            # Keep the session running until the job ends or is interrupted
            try:
                while not self.stop_event.is_set():
                    # Check if the processes are still running
                    if (self.server_process and self.server_process.poll() is not None or
                        self.cloudflared_process and self.cloudflared_process.poll() is not None):
                        logger.error("One of the processes has terminated unexpectedly")
                        break
                    
                    # Sleep to avoid busy waiting
                    time.sleep(5)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
            
        except Exception as e:
            logger.error(f"Error in interactive session: {e}")
            raise
            
        finally:
            self.cleanup() 