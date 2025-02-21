# LLM Batch Processing Pipeline for HPC Systems

A streamlined solution for running Large Language Models (LLMs) in batch mode on HPC systems powered by Slurm. 

## Architecture

```
Input JSON                         Batch Processing                    Output JSON
(data/input/*.json)               (Ollama + Model)                   (data/output/)
     │                                   │                                 │
     │                                   │                                 │
     ▼                                   ▼                                 ▼
[                                ┌──────────────┐                   [
  {                              │              │                      {
    "prompt": "...",       ────▶ │   Model on   │────▶                 "input": {...},
    "temperature": 0.7           │    GPU(s)    │                      "output": "...",
    "top_p": 0.9,                │              │                       "timestamp": ...
    "max_tokens": 1024           └──────────────┘                     },
  }                                                                 ...]
]
```

## Installation

1. **Create and Activate Conda Environment:**
   ```bash
   conda create -n llm_runner python=3.11 -y
   conda activate llm_runner
   ```

2. **Install Dependencies:**
   The project uses `pyproject.toml` for dependency management. Install using:
   ```bash
   pip install -e .
   ```

3. **Environment Setup:**
   ```bash
   cp .env.example .env
   # Edit .env with your SLURM account and model details
   ```

## Quick Start

1. **Prepare Input Data:**
   Create JSON files in `data/input/` with your prompts. Each file should contain an array of prompt objects:
   ```json
   [
     {
       "prompt": "Explain quantum computing in simple terms.",
       "temperature": 0.7,        # Controls randomness (optional)
       "top_p": 0.9,             # Controls diversity (optional)
       "max_tokens": 1024,       # Max response length (optional)
       "stop": ["###"]           # Stop sequences (optional)
     }
   ]
   ```

2. **Submit Job:**
   ```bash
   bash scripts/llm_pipeline.sh [model_type] [model_size] [account]
   ```
   Example:
   ```bash
   bash scripts/llm_pipeline.sh qwen 7b acc_name
   ```

## Repository Structure

```
.
├── config/                # Configuration files
│   ├── container.def     # Apptainer container definition
│   └── models/           # Model-specific YAML configs
│       ├── qwen/         # Qwen model configurations
│       └── llama/        # Llama model configurations
├── data/                 # Data directories
│   ├── input/           # Place input JSON files here
│   └── output/          # Generated outputs with timestamps
├── logs/                # SLURM and processing logs
├── models/              # Downloaded model cache
├── scripts/             # SLURM and utility scripts
│   ├── llm_pipeline.sh  # Main job submission script
│   └── run_llm.sh       # Job execution script
├── src/                 # Python source code
│   ├── batch_processor.py  # Batch processing implementation
│   └── config_validator.py # Configuration validation
├── pyproject.toml       # Project dependencies and metadata
└── requirements.txt     # Direct dependency requirements
```

## Available Models

Model configurations in `config/models/`:

| Model | Config File | GPU Memory | Notes |
|-------|-------------|------------|--------|
| Qwen 2.5 7B | qwen.env | ~15GB | Default setup |
| Qwen 2.5 72B | qwen.env | ~35GB | Requires A100 80GB |
| Llama 3.2 7B | llama.env | ~13GB | Good for general use |
| Llama 3.3 70B | llama.env | ~38GB | Requires A100 80GB |

To use a different model, copy settings from `config/models/<model>.env` to your `.env` file.

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
MODEL_NAME=qwen2.5:7b         # Model to use
GPU_LAYERS=35                 # GPU layers to use
BATCH_SIZE=8                  # Prompts per batch

# System Resources
MAX_CONCURRENT_REQUESTS=2     # Parallel requests
CPU_THREADS=4                 # CPU threads to use
```

## Input Parameters

Each prompt in your input JSON can have these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | (required) | The input text |
| temperature | float | 0.7 | Randomness (0.0-1.0) |
| top_p | float | 0.9 | Diversity (0.0-1.0) |
| max_tokens | integer | 2048 | Max response length |
| stop | string[] | null | Stop sequences |

## Output Format

Results are saved in `data/output/batch_results_<timestamp>.json`:
```json
[
  {
    "input": {
      "prompt": "Original prompt text",
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 1024
    },
    "output": "Generated response text",
    "timestamp": 1234567890.123
  }
]
```

## License

[MIT License](LICENSE)