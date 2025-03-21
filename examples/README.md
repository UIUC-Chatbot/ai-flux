# AI-Flux Examples

This directory contains example scripts demonstrating how to use AI-Flux for batch processing with LLMs on HPC systems.

## Available Examples

1. **Direct JSONL Processing**: [`jsonl_slurm_direct.py`](jsonl_slurm_direct.py) - Core workflow submitting JSONL directly to SLURM
2. **CSV to JSONL**: [`csv_jsonl_example.py`](csv_jsonl_example.py) - Convert CSV data to JSONL format
3. **JSON to JSONL**: [`json_jsonl_example.py`](json_jsonl_example.py) - Convert JSON data to JSONL format
4. **Directory to JSONL**: [`directory_jsonl_example.py`](directory_jsonl_example.py) - Convert directory contents to JSONL

## Configuration Parameters

All examples accept the following configuration parameters, which follow the priority system (code parameters > environment variables > defaults):

### Core Parameters

| Parameter | Type | Example | Description | Priority |
|-----------|------|---------|-------------|----------|
| `input_path` | string | `"data/input.jsonl"` | Path to input JSONL file | Required |
| `output_path` | string | `"results/output.json"` | Path for results | Required |
| `model` | string | `"llama3.2:3b"` | Model to use (format: `model:size`) | Code > ENV > Default |
| `batch_size` | int | `4` | Number of prompts to process in parallel | Code > ENV > Default |
| `max_retries` | int | `3` | Number of retries for failed requests | Code > ENV > Default |
| `retry_delay` | float | `1.0` | Delay between retries (seconds) | Code > ENV > Default |

### SLURM Parameters

| Parameter | Type | Example | Description | Priority |
|-----------|------|---------|-------------|----------|
| `account` | string | `"myaccount"` | SLURM account name | Required |
| `partition` | string | `"a100"` | GPU partition | Code > ENV > Default |
| `time` | string | `"01:00:00"` | Job time limit (HH:MM:SS) | Code > ENV > Default |
| `mem` | string | `"16G"` | Memory allocation | Code > ENV > Default |
| `gpus_per_node` | int | `1` | GPUs per node | Code > ENV > Default |
| `nodes` | int | `1` | Number of nodes | Code > ENV > Default |
| `cpus_per_task` | int | `4` | CPUs per task | Code > ENV > Default |

### Model Parameters

| Parameter | Type | Example | Description | Priority |
|-----------|------|---------|-------------|----------|
| `temperature` | float | `0.7` | Sampling temperature | Code > ENV > Default |
| `top_p` | float | `0.9` | Top-p sampling parameter | Code > ENV > Default |
| `top_k` | int | `40` | Top-k sampling parameter | Code > ENV > Default |
| `max_tokens` | int | `2048` | Maximum tokens to generate | Code > ENV > Default |
| `stop_sequences` | list | `["###"]` | Stop sequences | Code > ENV > Default |

## Setting Parameters

Parameters can be specified in three ways, following the priority system:

### 1. Direct in Code (Highest Priority)

```python
slurm_runner.run(
    input_path="data/prompts.jsonl",
    output_path="results/output.json",
    model="llama3.2:3b",
    batch_size=4,
    temperature=0.5,
    max_tokens=4096
)
```

### 2. Via Environment Variables (.env file)

```bash
# In .env file
MODEL_NAME=llama3.2:3b
BATCH_SIZE=4
TEMPERATURE=0.5
MAX_TOKENS=4096
```

### 3. Defaults (Lowest Priority)

Default values are used if no parameter is specified in code or environment.

## Running Examples

1. Ensure you have set up your environment variables in `.env`
2. Run an example:
   ```bash
   conda activate aiflux
   python examples/jsonl_slurm_direct.py
   ```

Each example script will create sample data files if they don't exist and then submit a SLURM job for processing.

## Further Documentation

For complete documentation of all configuration options, see:
- [Configuration Guide](../docs/CONFIGURATION.md)
- [Models Guide](../docs/MODELS.md) 