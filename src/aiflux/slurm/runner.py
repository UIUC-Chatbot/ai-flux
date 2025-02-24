#!/usr/bin/env python3
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

from ..core.config import Config, SlurmConfig
from ..core.processor import BaseProcessor

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
        self.config = config or Config().get_slurm_config()
        self.workspace = Path(workspace or os.getcwd())
        
        # Create workspace directories
        self.data_dir = self.workspace / "data"
        self.models_dir = self.workspace / "models"
        self.logs_dir = self.workspace / "logs"
        self.containers_dir = self.workspace / "containers"
        
        for directory in [
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.containers_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_environment(self) -> Dict[str, str]:
        """Setup environment variables for SLURM job.
        
        Returns:
            Dictionary of environment variables
        """
        # Get current environment
        env = os.environ.copy()
        
        # Get package root directory for container definition
        package_root = Path(__file__).parent.parent
        container_def = package_root / "container" / "container.def"
        
        # Add SLURM variables
        env.update({
            'SLURM_ACCOUNT': self.config.account,
            'SLURM_PARTITION': self.config.partition,
            'SLURM_NODES': str(self.config.nodes),
            'SLURM_GPUS_PER_NODE': str(self.config.gpus_per_node),
            'SLURM_TIME': self.config.time,
            'SLURM_MEM': self.config.memory,
            'SLURM_CPUS_PER_TASK': str(self.config.cpus_per_task),
            
            # Workspace paths
            'PROJECT_ROOT': str(self.workspace),
            'DATA_INPUT_DIR': str(self.data_dir / "input"),
            'DATA_OUTPUT_DIR': str(self.data_dir / "output"),
            'MODELS_DIR': str(self.models_dir),
            'LOGS_DIR': str(self.logs_dir),
            'CONTAINERS_DIR': str(self.containers_dir),
            'CONTAINER_DEF': str(container_def),
            
            # Apptainer settings
            'APPTAINER_TMPDIR': str(self.workspace / "tmp"),
            'APPTAINER_CACHEDIR': str(self.workspace / "tmp/cache"),
            
            # Ollama settings
            'OLLAMA_MODELS': str(self.models_dir),
            'OLLAMA_HOME': str(self.workspace / ".ollama"),
            'OLLAMA_ORIGINS': "*",
            'OLLAMA_INSECURE': "true",
            'CURL_CA_BUNDLE': "",
            'SSL_CERT_FILE': ""
        })
        
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
            **kwargs: Additional parameters for processor
        """
        # Setup paths
        # 1. For input:
        input_path = Path(input_source) if input_source else None
        if input_path is None:
            # Try env var
            if 'DATA_INPUT_DIR' in os.environ:
                input_path = Path(os.environ['DATA_INPUT_DIR'])
            else:
                # Default
                input_path = self.data_dir / "input"

        # 2. For output:
        output_path = Path(output_path) if output_path else None
        if output_path is None:
            # Try env var
            if 'DATA_OUTPUT_DIR' in os.environ:
                output_path = Path(os.environ['DATA_OUTPUT_DIR']) / f"results_{int(time.time())}.json"
            else:
                # Default
                output_path = self.data_dir / "output" / f"results_{int(time.time())}.json"
        
        # Copy input to workspace if needed
        if not input_path.is_relative_to(self.workspace):
            workspace_input = self.data_dir / "input" / input_path.name
            if input_path.is_file():
                workspace_input.parent.mkdir(parents=True, exist_ok=True)
                workspace_input.write_bytes(input_path.read_bytes())
            else:
                import shutil
                shutil.copytree(input_path, workspace_input)
            input_path = workspace_input
        
        # Setup environment
        env = self._setup_environment()
        
        # Find available port
        port = self._find_available_port()
        env['OLLAMA_PORT'] = str(port)
        env['OLLAMA_HOST'] = f"0.0.0.0:{port}"
        
        # Create SLURM job script
        job_script = [
            "#!/bin/bash",
            f"#SBATCH --job-name=llm_processor",
            f"#SBATCH --account={self.config.account}",
            f"#SBATCH --partition={self.config.partition}",
            f"#SBATCH --nodes={self.config.nodes}",
            f"#SBATCH --gpus-per-node={self.config.gpus_per_node}",
            f"#SBATCH --time={self.config.time}",
            f"#SBATCH --mem={self.config.memory}",
            f"#SBATCH --cpus-per-task={self.config.cpus_per_task}",
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
            "# Create temporary directories",
            "mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR",
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
            "# Run processor",
            f"python3 -c \"",
            "import sys",
            "sys.path.append('$PROJECT_ROOT')",
            "from aiflux.processors import BatchProcessor",
            "processor = BatchProcessor(",
            "    model_config=model_config,",
            "    **processor_kwargs",
            ")",
            f"processor.process_all('{input_path}', '{output_path}', **kwargs)",
            "\"",
            "",
            "# Cleanup",
            "pkill -f \"ollama serve\" || true",
            "rm -rf $APPTAINER_TMPDIR $APPTAINER_CACHEDIR"
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