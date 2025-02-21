#!/bin/bash

# Get the directory of the script and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Debug: Print paths
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Current PWD: $PWD"

# Load environment variables
set -a
source "$PROJECT_ROOT/.env"
set +a

# Debug: Print environment variables
echo "SLURM_ACCOUNT: $SLURM_ACCOUNT"
echo "DATA_INPUT_DIR: $DATA_INPUT_DIR"
echo "MODELS_DIR: $MODELS_DIR"

# Parse command line arguments
MODEL_TYPE=${1:-"qwen"}  # qwen or llama
MODEL_SIZE=${2:-"7b"}    # 7b, 70b, or 72b

# Load model config and set variables
eval $(python3 -c "
import yaml
with open('$PROJECT_ROOT/config/models/${MODEL_TYPE}/${MODEL_SIZE}.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(f'export MODEL_NAME={config[\"name\"]}')
    print(f'export GPU_LAYERS={config[\"resources\"][\"gpu_layers\"]}')
    print(f'export BATCH_SIZE={config[\"resources\"][\"batch_size\"]}')
    print(f'export MAX_CONCURRENT_REQUESTS={config[\"resources\"][\"max_concurrent\"]}')
    print(f'export MODEL_SYSTEM_PROMPT=\"{config[\"system\"][\"prompt\"]}\"')
    print(f'export SLURM_MEM={config[\"requirements\"][\"min_gpu_memory\"]}')
    print(f'export SLURM_CPUS_PER_TASK={config.get(\"requirements\", {}).get(\"cpu_threads\", 4)}')
")

# Export variables for the job
export PROJECT_ROOT="$PROJECT_ROOT"
export MODEL_NAME="$MODEL_NAME"
export GPU_LAYERS="$GPU_LAYERS"
export BATCH_SIZE="$BATCH_SIZE"
export MAX_CONCURRENT_REQUESTS="$MAX_CONCURRENT_REQUESTS"
export MODEL_SYSTEM_PROMPT="$MODEL_SYSTEM_PROMPT"
export DATA_INPUT_DIR="${DATA_INPUT_DIR:-"data/input"}"
export DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-"data/output"}"
export MODELS_DIR="${MODELS_DIR:-"models"}"
export LOGS_DIR="${LOGS_DIR:-"logs"}"
export CONTAINER_NAME="${CONTAINER_NAME:-"llm_processor"}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-300}"

# Submit the job directly with sbatch command
sbatch \
    --job-name="llm_${MODEL_TYPE}_${MODEL_SIZE}" \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION:-gpuA100x4}" \
    --nodes=1 \
    --gpus-per-node=1 \
    --time="${SLURM_TIME:-00:30:00}" \
    --mem="${SLURM_MEM}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --output="${LOGS_DIR:-logs}/%j.out" \
    --error="${LOGS_DIR:-logs}/%j.err" \
    --export=ALL \
    "$SCRIPT_DIR/run_llm.sh" 