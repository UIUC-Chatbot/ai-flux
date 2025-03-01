# AI-Flux: LLM Batch Processing Pipeline for HPC Systems

A streamlined solution for running Large Language Models (LLMs) in batch mode on HPC systems powered by Slurm. AI-Flux uses the OpenAI-compatible API format for all interactions.

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
  • Image Files                  └──────────────┘                    ]
]                                
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
   
   a. With JSON input (using OpenAI format with messages array):
   ```python
   from aiflux import BatchProcessor, SlurmRunner
   from aiflux.io import JSONBatchHandler

   # Initialize processor
   processor = BatchProcessor(
       model="qwen2.5:7b",
       input_handler=JSONBatchHandler()
   )

   # Run on SLURM
   runner = SlurmRunner(account="myaccount")
   runner.run(processor, input_source="prompts.json")
   ```

   Expected JSON format:
   ```json
   [
     {
       "messages": [
         {"role": "system", "content": "You are a helpful assistant"},
         {"role": "user", "content": "Explain quantum computing"}
       ],
       "temperature": 0.7,
       "max_tokens": 500
     }
   ]
   ```

   b. With CSV input:
   ```python
   from aiflux.io import CSVSinglePromptHandler

   processor = BatchProcessor(
       model="llama3.2:7b",
       input_handler=CSVSinglePromptHandler(),
       prompt_template="Analyze this text: {text}"
   )
   runner.run(processor, input_source="data.csv", system_prompt="You are a text analysis expert")
   ```

   c. With image input (vision capabilities):
   ```python
   from aiflux.io import VisionHandler
   
   # Process images with a standard prompt
   processor = BatchProcessor(
       model="llama3.2:7b",
       input_handler=VisionHandler(prompt_template="Describe this image in detail"),
       batch_size=1  # Process one image at a time due to token limits
   )
   runner.run(processor, input_source="images/")
   
   # Or with custom prompts for each image
   processor = BatchProcessor(
       model="llama3.2:7b",
       input_handler=VisionHandler(prompts_file="image_prompts.json"),
       batch_size=1
   )
   runner.run(processor, input_source="images/")
   ```

2. **Using Custom Processor:**
   ```python
   from aiflux import BaseProcessor, SlurmRunner
   
   class MyProcessor(BaseProcessor):
       def process_batch(self, batch):
           # Your custom processing logic
           pass

   processor = MyProcessor(model="qwen2.5:7b")
   runner = SlurmRunner(account="myaccount")
   runner.run(processor, input_source="data/")
   ```

For more detailed examples, see the [examples directory](examples/README.md).

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
│       │   ├── output.py      # Output handlers
│       │   └── vision.py      # Vision handlers
│       ├── slurm/             # SLURM integration
│       │   ├── runner.py      # SLURM job management
│       │   └── scripts/       # SLURM scripts
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
# Basic usage
aiflux run --model qwen2.5:7b --input data/prompts.json --output results/output.json

# Specify SLURM account and resources
aiflux run --model llama3.2:7b --input data/papers.csv --output results/summaries.json --account myaccount --time 02:00:00 --gpus 2

# Run with a template for CSV data
aiflux run --model qwen2.5:7b --input data/papers.csv --template "Summarize: {text}" --output results/

# Process images with vision models
aiflux run --model llama3.2:7b --input data/images/ --output results/image_analysis.json --vision --detail high
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
| JSONBatchHandler | Process JSON files with multiple prompts | JSON array with OpenAI-compatible messages format |
| CSVSinglePromptHandler | Run same prompt on CSV rows | CSV with data to format into OpenAI messages |
| CSVMultiPromptHandler | Each CSV row has a prompt | CSV with prompt column converted to OpenAI messages |
| DirectoryHandler | Process files in directory | Directory of files processed into OpenAI messages |
| VisionHandler | Process images | Directory of images or single image with OpenAI vision format |
| CustomHandler | User-defined processing | Any format converted to OpenAI messages |

## Output Format

Results are saved in the user's workspace:
```json
[
  {
    "input": {
      "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Original prompt text"}
      ],
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 1024,
      "metadata": {
        "filename": "example.txt"
      }
    },
    "output": "Generated response text",
    "metadata": {
      "model": "llama3:7b",
      "timestamp": 1234567890.123
    }
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