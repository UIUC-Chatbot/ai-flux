# AI-Flux: LLM Batch Processing Pipeline for HPC Systems

A streamlined solution for running Large Language Models (LLMs) in batch mode on HPC systems powered by Slurm. 

## Architecture

```
Input Data                        Batch Processing                    Output Data
(Multiple Formats)               (Ollama + Model)                   (User Workspace)
     │                                   │                                 │
     │                                   │                                 │
     ▼                                   ▼                                 ▼
[                                ┌──────────────┐                   [
  • JSON Files                   │              │                      • JSON Results
  • CSV Files              ────▶ │   Model on   │────▶                • CSV Results
  • Text Files                   │    GPU(s)    │                     • Custom Formats
  • Custom Input                 │              │                      
]                                └──────────────┘                    ]
```

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

1. **Using Built-in Batch Processor:**
   
   a. With JSON input:
   ```python
   from aiflux import BatchProcessor, SlurmRunner
   from aiflux.io import JSONBatchHandler

   # Initialize processor
   processor = BatchProcessor(
       model="qwen:7b",
       input_handler=JSONBatchHandler()
   )

   # Run on SLURM
   runner = SlurmRunner(account="myaccount")
   runner.run(processor, input_source="prompts.json")
   ```

   b. With CSV input:
   ```python
   from aiflux.io import CSVSinglePromptHandler

   processor = BatchProcessor(
       model="qwen:7b",
       input_handler=CSVSinglePromptHandler(),
       prompt_template="Analyze this text: {text}"
   )
   runner.run(processor, input_source="data.csv")
   ```

2. **Using Custom Processor:**
   ```python
   from aiflux import BaseProcessor, SlurmRunner
   
   class MyProcessor(BaseProcessor):
       def process_batch(self, batch):
           # Your custom processing logic
           pass

   processor = MyProcessor(model="qwen:7b")
   runner = SlurmRunner(account="myaccount")
   runner.run(processor, input_source="data/")
   ```

## Repository Structure

```
aiflux/
├── src/
│   └── aiflux/                 
│       ├── core/              
│       │   ├── processor.py   # Base processor interface
│       │   ├── config.py      # Configuration management
│       │   └── client.py      # LLM client interface
│       ├── processors/        # Built-in processors
│       │   ├── batch.py       # Basic batch processor
│       │   └── stream.py      # Streaming processor
│       ├── io/               
│       │   ├── handlers.py    # Input handlers
│       │   └── output.py      # Output handlers
│       ├── slurm/             # SLURM integration
│       │   ├── runner.py      # SLURM job management
│       │   └── scripts/       # SLURM scripts
│       ├── templates/         # Model templates
│       │   ├── llama/
│       │   └── qwen/
│       ├── container/         # Container management
│       │   └── container.def  # Default container definition
│       └── utils/            
│           └── env.py         # Environment utilities
├── examples/                  # Example implementations
├── tests/                    
└── pyproject.toml
```

## Available Models

Model configurations in `aiflux/templates/`:

| Model | Config File | GPU Memory | Notes |
|-------|-------------|------------|--------|
| Qwen 2.5 7B | qwen/7b.yaml | ~15GB | Default setup |
| Qwen 2.5 72B | qwen/72b.yaml | ~35GB | Requires A100 80GB |
| Llama 3.2 7B | llama/7b.yaml | ~13GB | Good for general use |
| Llama 3.3 70B | llama/70b.yaml | ~38GB | Requires A100 80GB |

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

## Input Handlers

| Handler | Description | Input Format |
|---------|-------------|--------------|
| JSONBatchHandler | Process JSON files with multiple prompts | JSON array of prompt objects |
| CSVSinglePromptHandler | Run same prompt on CSV rows | CSV with data to format prompt |
| CSVMultiPromptHandler | Each CSV row has a prompt | CSV with prompt column |
| DirectoryHandler | Process files in directory | Directory of files |
| CustomHandler | User-defined processing | Any format |

## Output Format

Results are saved in the user's workspace:
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

## Input/Output Path Handling

The system handles input and output paths with the following precedence:

1. **User-Specified Paths (Highest Priority)**
   ```python
   runner.run(
       processor,
       input_source='data/prompts.json',
       output_path='results/batch_results.json'
   )
   ```
   If paths are specified in code, they take precedence over environment variables.

2. **Environment Variables (Middle Priority)**
   ```bash
   # In .env file
   DATA_INPUT_DIR=/path/to/input
   DATA_OUTPUT_DIR=/path/to/output
   ```
   Used if no paths are specified in code.

3. **Default Paths (Lowest Priority)**
   ```
   data/input/  # Default input directory
   data/output/ # Default output directory
   ```
   Used if neither code paths nor environment variables are set.

## License

[MIT License](LICENSE) 