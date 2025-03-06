#!/usr/bin/env python3
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

from ..core.config import Config, SlurmConfig
from ..core.processor import BaseProcessor
from ..io import JSONBatchHandler, CSVSinglePromptHandler, VisionHandler, JSONOutputHandler, DirectoryHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SlurmRunner:
    """Runner for executing processors on SLURM."""
    
    def __init__(
        self,
        config: Optional[SlurmConfig] = None,
        workspace: Optional[str] = None
    ):
        """Initialize SLURM runner.
        
        Args:
            config: SLURM configuration
            workspace: Path to workspace directory
        """
        # Initialize config
        self.config_manager = Config()
        self.slurm_config = config or self.config_manager.get_slurm_config()
        
        # Set workspace
        self.workspace = Path(workspace) if workspace else self.config_manager.workspace
        
        # Get paths using config manager (following precedence rules)
        self.data_dir = self.config_manager.get_path('DATA_DIR', self.workspace / "data")
        self.data_input_dir = self.config_manager.get_path('DATA_INPUT_DIR', self.data_dir / "input")
        self.data_output_dir = self.config_manager.get_path('DATA_OUTPUT_DIR', self.data_dir / "output")
        self.models_dir = self.config_manager.get_path('MODELS_DIR', self.workspace / "models")
        self.logs_dir = self.config_manager.get_path('LOGS_DIR', self.workspace / "logs")
        self.containers_dir = self.config_manager.get_path('CONTAINERS_DIR', self.workspace / "containers")
        
        # Create directories
        for directory in [
            self.data_dir,
            self.data_input_dir,
            self.data_output_dir,
            self.models_dir,
            self.logs_dir,
            self.containers_dir,
            self.workspace / "tmp",
            self.workspace / "tmp" / "cache"
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_environment(self) -> Dict[str, str]:
        """Setup environment variables for SLURM job.
        
        Returns:
            Dictionary of environment variables
        """
        # Get package root directory for container definition
        package_root = Path(__file__).parent.parent
        container_def = package_root / "container" / "container.def"
        
        # Use config manager to get environment with proper precedence
        # Map slurm_config fields to their corresponding environment variables
        overrides = {
            # Map SlurmConfig fields to SLURM_* environment variables
            'SLURM_ACCOUNT': self.slurm_config.account,
            'SLURM_PARTITION': self.slurm_config.partition,
            'SLURM_NODES': str(self.slurm_config.nodes),
            'SLURM_GPUS_PER_NODE': str(self.slurm_config.gpus_per_node),
            'SLURM_TIME': self.slurm_config.time,
            'SLURM_MEM': self.slurm_config.memory,
            'SLURM_CPUS_PER_TASK': str(self.slurm_config.cpus_per_task),
            
            # Add workspace paths
            'PROJECT_ROOT': str(self.workspace),
            'DATA_INPUT_DIR': str(self.data_input_dir),
            'DATA_OUTPUT_DIR': str(self.data_output_dir),
            'MODELS_DIR': str(self.models_dir),
            'LOGS_DIR': str(self.logs_dir),
            'CONTAINERS_DIR': str(self.containers_dir),
            'CONTAINER_DEF': str(container_def),
            
            # Set Apptainer paths explicitly
            'APPTAINER_TMPDIR': str(self.workspace / "tmp"),
            'APPTAINER_CACHEDIR': str(self.workspace / "tmp" / "cache"),
            'SINGULARITY_TMPDIR': str(self.workspace / "tmp"),
            'SINGULARITY_CACHEDIR': str(self.workspace / "tmp" / "cache"),
        }
        
        # Get environment with proper precedence
        env = self.config_manager.get_environment(overrides)
        
        return env
    
    def _find_available_port(self) -> int:
        """Find an available port for Ollama server.
        
        Returns:
            Available port number
        """
        import socket
        
        # Try ports in range 40000-50000
        for port in range(40000, 50000):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return port
                except socket.error:
                    continue
        
        raise RuntimeError("No available ports found")
    
    def run(
        self,
        processor: BaseProcessor,
        input_source: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """Run processor on SLURM.
        
        Args:
            processor: Processor to run
            input_source: Source of input data
            output_path: Path to save results
            **kwargs: Additional parameters for processor, including prompt_template for CSVSinglePromptHandler
        """
        # Setup paths following precedence: code paths > environment variables > defaults
        # Use config manager to resolve paths
        
        # 1. For input:
        # If input_source is a file path, use it directly
        input_path = Path(input_source)
        if not input_path.exists():
            # If it doesn't exist, check if it's relative to the data input directory
            data_input_dir = self.config_manager.get_path('DATA_INPUT_DIR')
            potential_path = data_input_dir / input_path.name
            if potential_path.exists():
                input_path = potential_path
            else:
                # If still not found, use the input_source as is
                # It might be created by the script or specified as an output location
                pass
        
        # 2. For output:
        if output_path:
            output_path = Path(output_path)
        else:
            output_dir = self.config_manager.get_path('DATA_OUTPUT_DIR')
            output_path = output_dir / f"results_{int(time.time())}.json"
        
        # Ensure directories exist
        self.config_manager.ensure_directory(input_path.parent if input_path.is_file() else input_path)
        self.config_manager.ensure_directory(output_path.parent)
        
        # Copy input to workspace if needed
        if not input_path.is_relative_to(self.workspace) and input_path.exists():
            workspace_input = self.data_input_dir / input_path.name
            if input_path.is_file():
                workspace_input.parent.mkdir(parents=True, exist_ok=True)
                workspace_input.write_bytes(input_path.read_bytes())
            else:
                import shutil
                # If directory exists, remove it first to avoid FileExistsError
                if workspace_input.exists():
                    shutil.rmtree(workspace_input)
                shutil.copytree(input_path, workspace_input)
            input_path = workspace_input
        
        # Setup environment
        env = self._setup_environment()
        
        # Add processor configuration to environment
        env['MODEL_NAME'] = processor.model_config.name
        env['HANDLER_CLASS_NAME'] = processor.input_handler.__class__.__name__
        env['BATCH_SIZE'] = str(processor.batch_size)
        env['OLLAMA_MODEL_NAME'] = processor.model_config.name
        
        # Find available port
        port = self._find_available_port()
        env['OLLAMA_PORT'] = str(port)
        env['OLLAMA_HOST'] = f"0.0.0.0:{port}"
        
        # Add prompt_template to environment if provided
        if 'prompt_template' in kwargs:
            env['PROMPT_TEMPLATE'] = str(kwargs['prompt_template'])
            
        # Add prompts_file to environment if provided
        if hasattr(processor.input_handler, 'prompts_file') and processor.input_handler.prompts_file:
            env['prompts_file'] = str(processor.input_handler.prompts_file)
            
        # Add prompts_map if it's provided in the kwargs
        if 'prompts_map' in kwargs:
            env['prompts_map'] = json.dumps(kwargs['prompts_map'])
            
        # Add system_prompt if it's provided in the kwargs
        if 'system_prompt' in kwargs:
            env['system_prompt'] = str(kwargs['system_prompt'])
            
        # Add file_pattern if it's provided in the kwargs
        if 'file_pattern' in kwargs:
            env['file_pattern'] = str(kwargs['file_pattern'])
        
        #logger.info(f"Environment: {env}") 
        # logger.info(f"Environment var prompt_template: {env['PROMPT_TEMPLATE']}")
        
        # Create SLURM job script
        job_script = [
            "#!/bin/bash",
            f"#SBATCH --job-name=llm_processor",
            f"#SBATCH --account={self.slurm_config.account}",
            f"#SBATCH --partition={self.slurm_config.partition}",
            f"#SBATCH --nodes={self.slurm_config.nodes}",
            f"#SBATCH --gpus-per-node={self.slurm_config.gpus_per_node}",
            f"#SBATCH --time={self.slurm_config.time}",
            f"#SBATCH --mem={self.slurm_config.memory}",
            f"#SBATCH --cpus-per-task={self.slurm_config.cpus_per_task}",
            f"#SBATCH --output={self.logs_dir}/%j.out",
            f"#SBATCH --error={self.logs_dir}/%j.err",
            "",
            "# Load required modules",
            "module purge",
            "",
            "# Try loading GCC",
            "for gcc_version in '11.4.0' '11.3.0'; do",
            "    if module load gcc/$gcc_version &>/dev/null; then",
            "        echo \"Loaded gcc/$gcc_version\"",
            "        break",
            "    fi",
            "done",
            "",
            "# Try loading CUDA",
            "for cuda_version in '12.2.1' '11.7.0'; do",
            "    if module load cuda/$cuda_version &>/dev/null; then",
            "        echo \"Loaded cuda/$cuda_version\"",
            "        break",
            "    fi",
            "done",
            "",
            "# Create all necessary directories",
            "mkdir -p $DATA_INPUT_DIR $DATA_OUTPUT_DIR $MODELS_DIR $LOGS_DIR $CONTAINERS_DIR $APPTAINER_TMPDIR $APPTAINER_CACHEDIR",
            "",
            "# Start Ollama server",
            "mkdir -p $OLLAMA_HOME $OLLAMA_MODELS",
            "",
            "# Build container if needed",
            "if [ ! -f \"$CONTAINERS_DIR/llm_processor.sif\" ]; then",
            "    APPTAINER_DEBUG=1 apptainer build --force $CONTAINERS_DIR/llm_processor.sif $CONTAINER_DEF",
            "fi",
            "",
            "# Start server",
            "OLLAMA_DEBUG=1 apptainer exec --nv \\",
            "    --env OLLAMA_HOST=$OLLAMA_HOST \\",
            "    --env OLLAMA_ORIGINS=* \\",
            "    --env OLLAMA_MODELS=$OLLAMA_MODELS \\",
            "    --env OLLAMA_HOME=$OLLAMA_HOME \\",
            "    --env OLLAMA_INSECURE=true \\",
            "    --env CURL_CA_BUNDLE= \\",
            "    --env SSL_CERT_FILE= \\",
            "    --bind $DATA_INPUT_DIR:/app/data/input,$DATA_OUTPUT_DIR:/app/data/output,$MODELS_DIR:/app/models,$LOGS_DIR:/app/logs,$OLLAMA_HOME:$OLLAMA_HOME \\",
            "    $CONTAINERS_DIR/llm_processor.sif \\",
            "    ollama serve &",
            "",
            "OLLAMA_PID=$!",
            "",
            "# Wait for server",
            "for i in {1..60}; do",
            "    if curl -s \"http://localhost:$OLLAMA_PORT/api/version\" &>/dev/null; then",
            "        echo \"Ollama server started\"",
            "        break",
            "    fi",
            "    if ! ps -p $OLLAMA_PID > /dev/null; then",
            "        echo \"Ollama server died\"",
            "        exit 1",
            "    fi",
            "    echo \"Waiting... ($i/60)\"",
            "    sleep 1",
            "done",
            "",
            "# Pull model if needed",
            "MODEL_NAME=\"${OLLAMA_MODEL_NAME:-llama3.2-vision:11b}\"",
            "echo \"Checking if model ${MODEL_NAME} exists...\"",
            "if ! curl -s \"http://localhost:$OLLAMA_PORT/api/tags\" | grep -q \"${MODEL_NAME}\"; then",
            "    echo \"Pulling model ${MODEL_NAME}...\"",
            "    curl -X POST \"http://localhost:$OLLAMA_PORT/api/pull\" -d '{\"name\": \"${MODEL_NAME}\"}' -H \"Content-Type: application/json\"",
            "    if [ $? -ne 0 ]; then",
            "        echo \"Failed to pull model ${MODEL_NAME}\"",
            "        exit 1",
            "    fi",
            "else",
            "    echo \"Model ${MODEL_NAME} already exists\"",
            "fi",
            "",
            "# Run processor",
            f"python3 -c \"",
            "import sys",
            "import json",
            "import os",
            "sys.path.append('$PROJECT_ROOT')",
            "from aiflux.core.config import Config",
            "from aiflux.processors import BatchProcessor",
            "from aiflux.io import JSONBatchHandler, CSVSinglePromptHandler, VisionHandler, JSONOutputHandler, DirectoryHandler",
            "",
            "# Load model configuration",
            "config = Config()",
            "model_name = os.environ.get('MODEL_NAME', 'llama3')",
            f"# Handle special vision model names that contain hyphens",
            f"if '-' in model_name.split(':')[0]:",
            f"    model_parts = model_name.split(':')",
            f"    model_type = model_parts[0]",
            f"    model_size = model_parts[1] if len(model_parts) > 1 else '11b'",
            f"else:",
            f"    model_type = '{processor.model_config.name.split(':')[0] if ':' in processor.model_config.name else 'qwen2.5'}'",
            f"    model_size = '{processor.model_config.name.split(':')[1] if ':' in processor.model_config.name else '7b'}'",
            "try:",
            "    model_config = config.load_model_config(model_type, model_size)",
            "except Exception as e:",
            "    print(f'Error loading model config for {model_type}:{model_size}: {e}')",
            "    # Fallback to default model",
            "    model_config = config.load_model_config('qwen', '7b')",
            "",
            "# Get input handler class",
            "handler_class_name = os.environ.get('HANDLER_CLASS_NAME', 'VisionHandler')",
            "handler_class = None",
            "if handler_class_name == 'JSONBatchHandler':",
            "    handler_class = JSONBatchHandler",
            "elif handler_class_name == 'CSVSinglePromptHandler':",
            "    handler_class = CSVSinglePromptHandler",
            "elif handler_class_name == 'VisionHandler':",
            "    handler_class = VisionHandler",
            "elif handler_class_name == 'DirectoryHandler':",
            "    from aiflux.io import DirectoryHandler",
            "    handler_class = DirectoryHandler",
            "else:",
            "    raise ValueError(f'Unknown input handler class: {handler_class_name}')",
            "",
            "# Prepare processor kwargs",
            "processor_kwargs = {",
            "    'input_handler': handler_class(),",
            f"    'batch_size': int(os.environ.get('BATCH_SIZE', '4')),",
            "'output_handler': JSONOutputHandler(),",
            "}",
            "",
            "# Special handling for VisionHandler initialization",
            "if handler_class_name == 'VisionHandler':",
            "    # Check for prompts_file in environment or passed kwargs",
            "    if 'prompts_file' in os.environ:",
            "        processor_kwargs['input_handler'] = handler_class(prompts_file=os.environ['prompts_file'])",
            "    elif 'prompt_template' in os.environ:",
            "        processor_kwargs['input_handler'] = handler_class(prompt_template=os.environ['prompt_template'])",
            "",
            "    # No longer add prompts_map to processor_kwargs, it will be added to process_all_kwargs below",
            "",
            "# Special handling for DirectoryHandler initialization",
            "if handler_class_name == 'DirectoryHandler':",
            "    # DirectoryHandler takes no arguments in constructor",
            "    processor_kwargs['input_handler'] = handler_class()",
            "    # file_pattern will be passed as an argument to process_all_kwargs",
            "",
            "# Create processor",
            "batch_processor = BatchProcessor(",
            "    model_config=model_config,",
            "    **processor_kwargs",
            ")",
            "",
            "# Prepare process_all kwargs",
            "process_all_kwargs = {}",
            "",
            "# Add prompt_template if using CSVSinglePromptHandler",
            "if handler_class_name == 'CSVSinglePromptHandler':",
            "    prompt_template = os.environ.get('PROMPT_TEMPLATE', '')",
            "    if prompt_template:",
            "        process_all_kwargs['prompt_template'] = prompt_template",
            "",
            "# Add VisionHandler specific parameters",
            "if handler_class_name == 'VisionHandler':",
            "    if 'prompts_map' in os.environ:",
            "        import json",
            "        process_all_kwargs['prompts_map'] = json.loads(os.environ['prompts_map'])",
            "    if 'system_prompt' in os.environ:",
            "        process_all_kwargs['system_prompt'] = os.environ['system_prompt']",
            "",
            "# Add DirectoryHandler specific parameters",
            "if handler_class_name == 'DirectoryHandler':",
            "    if 'system_prompt' in os.environ:",
            "        process_all_kwargs['system_prompt'] = os.environ['system_prompt']",
            "    if 'file_pattern' in os.environ:",
            "        process_all_kwargs['file_pattern'] = os.environ['file_pattern']",
            "",
            "# Add any other kwargs from environment variables",
            "import json  # Ensure json is imported",
            "",
            "# Check for other common parameters",
            "for key in ['prompt_template', 'max_tokens', 'temperature', 'top_p', 'top_k', 'file_pattern']:",
            "    if key in os.environ:",
            "        process_all_kwargs[key] = os.environ[key]",
            "",
            f"batch_processor.run('{input_path}', '{output_path}', **process_all_kwargs)",
            "\"",
            "",
            "# Cleanup",
            "pkill -f \"ollama serve\" || true",
            "# Only remove temporary directories that we created",
            "if [ -d \"$APPTAINER_TMPDIR\" ] && [ -w \"$APPTAINER_TMPDIR\" ]; then",
            "    rm -rf \"$APPTAINER_TMPDIR\"",
            "fi",
            "if [ -d \"$APPTAINER_CACHEDIR\" ] && [ -w \"$APPTAINER_CACHEDIR\" ]; then",
            "    rm -rf \"$APPTAINER_CACHEDIR\"",
            "fi"
        ]
        
        # Write job script
        job_script_path = self.workspace / "job.sh"
        with open(job_script_path, 'w') as f:
            f.write('\n'.join(job_script))
        
        # Submit job
        try:
            subprocess.run(
                ['sbatch', str(job_script_path)],
                env=env,
                check=True
            )
            logger.info("Job submitted successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error submitting job: {e}")
            raise
        
        finally:
            # Cleanup job script
            job_script_path.unlink() 