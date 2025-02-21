#!/bin/bash

# Error handling
set -e
trap 'cleanup' EXIT
trap 'echo "Error occurred at line $LINENO"' ERR

# Load required modules
module purge
module load cuda/12.2.1
module load gcc/11.4.0

# Create directories only if they don't exist
for dir in "${DATA_INPUT_DIR}" "${DATA_OUTPUT_DIR}" "${MODELS_DIR}" "${LOGS_DIR}" "${CONTAINERS_DIR:-containers}"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Set temporary directory for Apptainer
export APPTAINER_TMPDIR="${PROJECT_ROOT}/tmp"
export APPTAINER_CACHEDIR="${PROJECT_ROOT}/tmp/cache"
if [ ! -d "$APPTAINER_TMPDIR" ]; then
    mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
fi

# Find available port
port=$((RANDOM % 10000 + 40000))
while nc -z localhost $port 2>/dev/null; do
    port=$((RANDOM % 10000 + 40000))
done

# Export Ollama variables
export OLLAMA_PORT=$port
export OLLAMA_HOST="0.0.0.0:$port"
export OLLAMA_MODELS="${PROJECT_ROOT}/${MODELS_DIR}"
export OLLAMA_HOME="${PROJECT_ROOT}/.ollama"
export OLLAMA_ORIGINS="*"
export OLLAMA_INSECURE=true
export OLLAMA_CONCURRENT_REQUESTS=$MAX_CONCURRENT_REQUESTS
export OLLAMA_TIMEOUT=${OLLAMA_TIMEOUT:-300}
export CURL_CA_BUNDLE=""
export SSL_CERT_FILE=""

# Build container if needed
if [ ! -f "${CONTAINERS_DIR:-containers}/${CONTAINER_NAME}.sif" ]; then
    echo "Building container..."
    APPTAINER_DEBUG=1 apptainer build --force "${CONTAINERS_DIR:-containers}/${CONTAINER_NAME}.sif" config/container.def
fi

# Start Ollama server
echo "Starting Ollama server on port $OLLAMA_PORT..."
mkdir -p "$OLLAMA_HOME"
mkdir -p "${PROJECT_ROOT}/${MODELS_DIR}"

# Start server with debug output
echo "Starting Ollama server with models directory: ${PROJECT_ROOT}/${MODELS_DIR}"
OLLAMA_DEBUG=1 apptainer exec --nv \
    --env OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT" \
    --env OLLAMA_ORIGINS="*" \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    --env OLLAMA_HOME="$OLLAMA_HOME" \
    --env OLLAMA_INSECURE=true \
    --env CURL_CA_BUNDLE="" \
    --env SSL_CERT_FILE="" \
    --bind "${PROJECT_ROOT}/${DATA_INPUT_DIR}:/app/data/input,${PROJECT_ROOT}/${DATA_OUTPUT_DIR}:/app/data/output,${PROJECT_ROOT}/${MODELS_DIR}:/app/models,${PROJECT_ROOT}/${LOGS_DIR}:/app/logs,$OLLAMA_HOME:$OLLAMA_HOME" \
    "${CONTAINERS_DIR:-containers}/${CONTAINER_NAME}.sif" \
    ollama serve &

OLLAMA_PID=$!

# Wait for server with better verification
echo "Waiting for server to start..."
for i in {1..60}; do
    if curl -s "http://localhost:$OLLAMA_PORT/api/version" &>/dev/null; then
        echo "Ollama server started successfully"
        break
    fi
    if ! ps -p $OLLAMA_PID > /dev/null; then
        echo "Ollama server process died"
        exit 1
    fi
    echo "Waiting... ($i/60)"
    sleep 1
done

if ! curl -s "http://localhost:$OLLAMA_PORT/api/version" &>/dev/null; then
    echo "Failed to start Ollama server after 60 seconds"
    exit 1
fi

# Check if model exists in cache
echo "Checking for model $MODEL_NAME in cache..."
if ! apptainer exec --nv \
    --env OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT" \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    --env OLLAMA_HOME="$OLLAMA_HOME" \
    --env OLLAMA_INSECURE=true \
    --env CURL_CA_BUNDLE="" \
    --env SSL_CERT_FILE="" \
    --bind "${PROJECT_ROOT}/${DATA_INPUT_DIR}:/app/data/input,${PROJECT_ROOT}/${DATA_OUTPUT_DIR}:/app/data/output,${PROJECT_ROOT}/${MODELS_DIR}:/app/models,${PROJECT_ROOT}/${LOGS_DIR}:/app/logs,$OLLAMA_HOME:$OLLAMA_HOME" \
    "${CONTAINERS_DIR:-containers}/${CONTAINER_NAME}.sif" \
    ollama list | grep -q "$MODEL_NAME"; then
    
    echo "Model $MODEL_NAME not found in cache, pulling..."
    if ! apptainer exec --nv \
        --env OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT" \
        --env OLLAMA_MODELS="$OLLAMA_MODELS" \
        --env OLLAMA_HOME="$OLLAMA_HOME" \
        --env OLLAMA_INSECURE=true \
        --env CURL_CA_BUNDLE="" \
        --env SSL_CERT_FILE="" \
        --bind "${PROJECT_ROOT}/${DATA_INPUT_DIR}:/app/data/input,${PROJECT_ROOT}/${DATA_OUTPUT_DIR}:/app/data/output,${PROJECT_ROOT}/${MODELS_DIR}:/app/models,${PROJECT_ROOT}/${LOGS_DIR}:/app/logs,$OLLAMA_HOME:$OLLAMA_HOME" \
        "${CONTAINERS_DIR:-containers}/${CONTAINER_NAME}.sif" \
        ollama pull "$MODEL_NAME"; then
        echo "Failed to pull model $MODEL_NAME"
        exit 1
    fi
else
    echo "Model $MODEL_NAME found in cache"
fi

# Warm up the model
echo "Warming up the model..."
if ! curl -s -X POST "http://localhost:$OLLAMA_PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"warming up the model\",\"stream\":false}"; then
    echo "Failed to warm up the model"
    exit 1
fi

# Process data
echo "Processing input data..."
python3 src/batch_processor.py \
    --input-dir "${DATA_INPUT_DIR}" \
    --output-dir "${DATA_OUTPUT_DIR}" \
    --model "$MODEL_NAME" \
    --batch-size "$BATCH_SIZE"

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    pkill -f "ollama serve" || true
    rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
}

# Report completion
echo "Pipeline completed successfully!" 