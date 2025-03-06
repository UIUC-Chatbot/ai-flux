# AI-Flux: LLM Batch Processing Pipeline for HPC Systems

A streamlined solution for running Large Language Models (LLMs) in batch mode on HPC systems powered by Slurm. AI-Flux uses the OpenAI-compatible API format with a JSONL-first architecture for all interactions.

## Architecture

```
      JSONL Input                    Batch Processing                    Results
   (OpenAI Format)                  (Ollama + Model)                   (JSON Output)
         │                                 │                                 │
         │                                 │                                 │
         ▼                                 ▼                                 ▼
    ┌──────────┐                   ┌──────────────┐                   ┌──────────┐
    │  Batch   │                   │              │                   │  Output  │
    │ Requests │─────────────────▶ │   Model on   │─────────────────▶│  Results │
    │  (JSONL) │                   │    GPU(s)    │                   │  (JSON)  │
    └──────────┘                   │              │                   └──────────┘
                                   └──────────────┘                    
```

AI-Flux processes JSONL files in a standardized OpenAI-compatible batch API format, enabling efficient processing of thousands of prompts on HPC systems with minimal overhead.

## Installation

1. **Create and Activate Conda Environment:**
   ```bash
   conda create -n aiflux python=3.11 -y
   conda activate aiflux
   ```

2. **Install Package:**
   ```bash
   pip install -e .
   ```

3. **Environment Setup:**
   ```bash
   cp .env.example .env
   # Edit .env with your SLURM account and model details
   ```

## Quick Start

### Core Batch Processing on SLURM

The primary workflow for AI-Flux is submitting JSONL files for batch processing on SLURM:

```python
from aiflux.slurm import SlurmRunner
from aiflux.core.config import Config

# Setup SLURM configuration
config = Config()
slurm_config = config.get_slurm_config()
slurm_config.account = "myaccount"

# Initialize runner
runner = SlurmRunner(config=slurm_config)

# Submit JSONL file directly for processing
job_id = runner.run(
    input_path="prompts.jsonl",
    output_path="results.json",
    model="llama3.2:3b",
    batch_size=4
)
print(f"Job submitted with ID: {job_id}")
```

JSONL input format follows the OpenAI Batch API specification:
```jsonl
{"custom_id":"request1","method":"POST","url":"/v1/chat/completions","body":{"model":"llama3.2:3b","messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Explain quantum computing"}],"temperature":0.7,"max_tokens":500}}
{"custom_id":"request2","method":"POST","url":"/v1/chat/completions","body":{"model":"llama3.2:3b","messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is machine learning?"}],"temperature":0.7,"max_tokens":500}}
```

### Alternative: Using BatchProcessor Directly

For more control, you can also use the BatchProcessor directly:

```python
from aiflux import BatchProcessor, SlurmRunner
from aiflux.core.config import Config

# Initialize processor
config = Config()
model_config = config.load_model_config("llama3.2", "3b")

processor = BatchProcessor(
    model_config=model_config,
    batch_size=4
)

# Run on SLURM
runner = SlurmRunner(account="myaccount")
runner.run(
    processor=processor,
    input_path="prompts.jsonl",
    output_path="results.json"
)
```

For more detailed examples, see the [examples directory](examples/).

## Repository Structure

```
aiflux/
├── src/
│   └── aiflux/                 
│       ├── core/              
│       │   ├── processor.py   # Base processor interface
│       │   ├── config.py      # Configuration management
│       │   ├── config_manager.py # Configuration priority system
│       │   └── client.py      # LLM client interface
│       ├── processors/        # Built-in processors
│       │   └── batch.py       # JSONL batch processor
│       ├── slurm/             # SLURM integration
│       │   ├── runner.py      # SLURM job management
│       │   └── scripts/       # SLURM scripts
│       ├── converters/        # Format converters (utilities)
│       │   ├── csv.py         # CSV to JSONL converter
│       │   ├── json.py        # JSON to JSONL converter
│       │   ├── directory.py   # Directory to JSONL converter
│       │   ├── vision.py      # Vision to JSONL converter
│       │   └── utils.py       # JSONL utilities
│       ├── io/                # Input/Output handling
│       │   ├── base.py        # Base output classes
│       │   └── output/        # Output handlers
│       │       └── json_output.py # JSON output handler
│       ├── templates/         # Model templates
│       │   ├── llama3.2/
│       │   ├── llama3.3/
│       │   └── qwen2.5/
│       └── utils/            
│           └── env.py         # Environment utilities
├── examples/                  # Example implementations
├── tests/                    
└── pyproject.toml
```

## Command-Line Interface

AI-Flux includes a command-line interface for submitting batch processing jobs:

```bash
# Process JSONL file directly (core functionality)
aiflux run --model llama3.2:3b --input data/prompts.jsonl --output results/output.json
```

For detailed command options:
```bash
aiflux --help
```

## Available Models

Model configurations in `aiflux/templates/`:

| Model | Config File | GPU Memory | Notes |
|-------|-------------|------------|-------|
| Qwen 2.5 7B | qwen2.5/7b.yaml | ~15GB | Default setup |
| Qwen 2.5 72B | qwen2.5/72b.yaml | ~35GB | Requires A100 80GB |
| Llama 3.2 7B | llama3.2/7b.yaml | ~13GB | Good for general use |
| Llama 3.3 70B | llama3.3/70b.yaml | ~38GB | Requires A100 80GB |

## Configuration

The `.env` file controls system-level settings:

```bash
# SLURM Settings (Required)
SLURM_ACCOUNT=your-account    # Your SLURM account
SLURM_PARTITION=gpuA100x4     # GPU partition
SLURM_NODES=1                 # Number of nodes
SLURM_GPUS_PER_NODE=1        # GPUs per node
SLURM_TIME=04:00:00          # Time limit
SLURM_MEM=32G                # Memory per node

# Model Selection
MODEL_NAME=llama3.2:3b        # Model to use
GPU_LAYERS=35                 # GPU layers to use
BATCH_SIZE=8                  # Prompts per batch

# System Resources
MAX_CONCURRENT_REQUESTS=2     # Parallel requests
CPU_THREADS=4                 # CPU threads to use
```

## Output Format

Results are saved in the user's workspace:
```json
[
  {
    "input": {
      "custom_id": "request1",
      "method": "POST",
      "url": "/v1/chat/completions",
      "body": {
        "model": "llama3.2:3b",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant"},
          {"role": "user", "content": "Original prompt text"}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
      },
      "metadata": {
        "source_file": "example.txt"
      }
    },
    "output": {
      "id": "chat-cmpl-123",
      "object": "chat.completion",
      "created": 1699123456,
      "model": "llama3.2:3b",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Generated response text"
          },
          "finish_reason": "stop"
        }
      ]
    },
    "metadata": {
      "model": "llama3.2:3b",
      "timestamp": "2023-11-04T12:34:56.789Z",
      "processing_time": 1.23
    }
  }
]
```

## Utility Converters

AI-Flux provides utility converters to help prepare JSONL files from various input formats:

### CSV to JSONL
```python
from aiflux.converters import csv_to_jsonl

result = csv_to_jsonl(
    input_path="data.csv",
    output_path="data.jsonl",
    prompt_template="Analyze this text: {text}",
    system_prompt="You are a text analysis expert",
    model="llama3.2:3b"
)
```

### JSON to JSONL
```python
from aiflux.converters import json_to_jsonl

result = json_to_jsonl(
    input_path="data.json",
    output_path="data.jsonl"
)
```

### Directory to JSONL
```python
from aiflux.converters import directory_to_jsonl

result = directory_to_jsonl(
    input_path="documents/",
    output_path="documents.jsonl",
    prompt_template="Summarize this document: {content}",
    system_prompt="You are a document summarization specialist",
    recursive=True,
    extensions=['.txt', '.md']
)
```

### CLI for Converters
```bash
# Convert CSV to JSONL
aiflux convert csv --input data/papers.csv --output data/papers.jsonl --template "Summarize: {text}"

# Convert directory to JSONL
aiflux convert dir --input data/documents/ --output data/docs.jsonl --recursive
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT License](LICENSE) 
