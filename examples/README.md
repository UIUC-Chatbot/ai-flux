# AI-Flux Examples

This directory contains example scripts demonstrating how to use AI-Flux for batch processing with LLMs on HPC systems.

## Available Examples

### 1. Basic Batch Processing (`batch_processing.py`)

A comprehensive example that demonstrates both JSON and CSV batch processing in a single script:

```python
# Process JSON prompts
config = Config()
model_config = config.load_model_config('qwen2.5', '7b')

processor = BatchProcessor(
    model_config=model_config,
    input_handler=JSONBatchHandler(),
    batch_size=8
)

runner = SlurmRunner(
    config=config.get_slurm_config({
        'account': os.getenv('SLURM_ACCOUNT'),
        'time': '01:00:00'
    })
)

runner.run(
    processor,
    input_source='data/prompts.json',
    output_path='results/batch_results.json'
)
```

### 2. JSON Batch Processing (`json_batch_processor.py`)

Example for processing a batch of JSON prompts:

```python
# Initialize processor with JSON handler
processor = BatchProcessor(
    model_config=model_config,
    input_handler=JSONBatchHandler(),
    batch_size=8
)

# Process inputs
runner.run(
    processor,
    input_source='data/prompts.json',
    output_path='results/batch_results.json'
)
```

Expected JSON input format:
```json
[
    {
        "prompt": "Explain quantum computing in simple terms.",
        "temperature": 0.7,
        "max_tokens": 1024
    },
    {
        "prompt": "What are the main applications of machine learning in healthcare?",
        "temperature": 0.8,
        "max_tokens": 2048
    }
]
```

### 3. CSV Batch Processing (`csv_batch_processor.py`)

Example for processing CSV files with a template prompt:

```python
# Initialize processor with CSV handler
processor = BatchProcessor(
    model_config=model_config,
    input_handler=CSVSinglePromptHandler(),
    batch_size=4
)

# Process inputs with template
runner.run(
    processor,
    input_source='data/papers.csv',
    output_path='results/paper_summaries.json',
    prompt_template=(
        "Please summarize the following research paper:\n\n"
        "Title: {title}\n"
        "Abstract: {abstract}\n\n"
        "Provide a concise summary focusing on:\n"
        "1. Main research question\n"
        "2. Key methodology\n"
        "3. Main findings\n"
        "4. Significance of results"
    )
)
```

Expected CSV input format:
```csv
title,abstract
"Quantum Supremacy Using a Programmable Superconducting Processor","The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor..."
"Large Language Models in Machine Learning","Recent advances in transformer architectures and pre-training techniques have led to significant improvements in natural language processing tasks..."
```

## Running the Examples

1. Ensure you have set up your environment variables in `.env`
2. Run an example:
   ```bash
   python examples/batch_processing.py
   ```

Each example script will create sample data files if they don't exist and then submit a SLURM job for processing. 