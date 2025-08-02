#!/bin/bash

# Knowledge Distillation Training Script for Mac Mini
# This script sets up the environment and runs the distillation training

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/configs/distillation_config.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
 ____  _     _   _ _ _       _   _             
|  _ \(_)___| |_(_) | | __ _| |_(_) ___  _ __  
| | | | / __| __| | | |/ _` | __| |/ _ \| '_ \ 
| |_| | \__ \ |_| | | | (_| | |_| | (_) | | | |
|____/|_|___/\__|_|_|_|\__,_|\__|_|\___/|_| |_|

xinSLM v04 - Knowledge Distillation Training
EOF
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    log "Python version: $PYTHON_VERSION"
    
    # Check if we're on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log "Running on macOS"
        
        # Check for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            log "Apple Silicon detected"
        else
            log "Intel Mac detected"
        fi
        
        # Check available memory
        MEMORY_GB=$(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc)
        log "Available memory: ${MEMORY_GB} GB"
        
        if [[ $MEMORY_GB -lt 8 ]]; then
            log_warning "Less than 8GB memory detected. Training may be slow or fail."
        fi
    else
        log "Running on non-macOS system"
    fi
    
    # Check disk space
    DISK_AVAILABLE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    log "Available disk space: $DISK_AVAILABLE"
}

# Setup Python environment
setup_environment() {
    log "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "${PROJECT_ROOT}/venv" ]]; then
        log "Creating virtual environment..."
        python3 -m venv "${PROJECT_ROOT}/venv"
    fi
    
    # Activate virtual environment
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        log "Installing requirements..."
        pip install -r "${PROJECT_ROOT}/requirements.txt"
    else
        log_warning "requirements.txt not found. Installing basic dependencies..."
        pip install torch transformers datasets accelerate wandb
    fi
    
    log_success "Environment setup complete"
}

# Prepare directories
prepare_directories() {
    log "Preparing directories..."
    
    # Create necessary directories
    mkdir -p "$LOG_DIR"
    mkdir -p "${PROJECT_ROOT}/checkpoints"
    mkdir -p "${PROJECT_ROOT}/data"
    
    log_success "Directories prepared"
}

# Check configuration
check_config() {
    log "Checking configuration..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Validate YAML syntax
    python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        yaml.safe_load(f)
    print('Configuration file is valid')
except Exception as e:
    print(f'Configuration file error: {e}')
    exit(1)
"
    
    log_success "Configuration validated"
}

# Monitor system resources during training
monitor_resources() {
    local pid=$1
    local log_file="${LOG_DIR}/resource_usage.log"
    
    echo "timestamp,cpu_percent,memory_gb,memory_percent" > "$log_file"
    
    while kill -0 "$pid" 2>/dev/null; do
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local cpu_percent=$(ps -p "$pid" -o %cpu --no-headers | tr -d ' ')
        local memory_kb=$(ps -p "$pid" -o rss --no-headers | tr -d ' ')
        local memory_gb=$(echo "scale=2; $memory_kb / 1024 / 1024" | bc -l)
        local memory_percent=$(ps -p "$pid" -o %mem --no-headers | tr -d ' ')
        
        echo "$timestamp,$cpu_percent,$memory_gb,$memory_percent" >> "$log_file"
        sleep 30  # Log every 30 seconds
    done
}

# Main training function
run_training() {
    log "Starting distillation training..."
    
    # Activate virtual environment
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Prepare training command
    TRAIN_CMD="python3 scripts/train_distillation.py --config $CONFIG_FILE"
    
    # Add debug mode if requested
    if [[ "${DEBUG:-}" == "true" ]]; then
        TRAIN_CMD="$TRAIN_CMD --debug"
        log_warning "Running in debug mode with reduced dataset"
    fi
    
    # Add resume checkpoint if specified
    if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CHECKPOINT"
        log "Resuming from checkpoint: $RESUME_CHECKPOINT"
    fi
    
    log "Training command: $TRAIN_CMD"
    
    # Start training with resource monitoring
    (
        eval "$TRAIN_CMD" 2>&1 | tee "${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
    ) &
    
    local train_pid=$!
    
    # Start resource monitoring in background
    monitor_resources "$train_pid" &
    local monitor_pid=$!
    
    # Wait for training to complete
    wait "$train_pid"
    local train_exit_code=$?
    
    # Stop resource monitoring
    kill "$monitor_pid" 2>/dev/null || true
    
    if [[ $train_exit_code -eq 0 ]]; then
        log_success "Training completed successfully!"
    else
        log_error "Training failed with exit code: $train_exit_code"
        exit $train_exit_code
    fi
}

# Post-training analysis
post_training_analysis() {
    log "Running post-training analysis..."
    
    # Check if best model exists
    local best_model_dir="${PROJECT_ROOT}/checkpoints/best_model"
    
    if [[ -d "$best_model_dir" ]]; then
        log "Best model found at: $best_model_dir"
        
        # Get model size
        local model_size=$(du -sh "$best_model_dir" | cut -f1)
        log "Model size: $model_size"
        
        # Check model files
        if [[ -f "$best_model_dir/pytorch_model.bin" ]]; then
            local model_file_size=$(ls -lh "$best_model_dir/pytorch_model.bin" | awk '{print $5}')
            log "Model file size: $model_file_size"
        fi
        
        # Generate model info
        python3 -c "
import sys
sys.path.append('$PROJECT_ROOT')
from models.distilled_llama import DistilledLlamaForCausalLM
import json
import torch

try:
    model = DistilledLlamaForCausalLM.from_pretrained('$best_model_dir')
    info = model.get_model_info()
    print(f'Model parameters: {info[\"total_parameters\"]:,}')
    print(f'Model size: {info[\"parameter_size_mb\"]:.1f} MB')
    
    # Save detailed info
    with open('$best_model_dir/model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
        
except Exception as e:
    print(f'Error loading model: {e}')
"
    else
        log_warning "Best model not found. Training may have failed."
    fi
    
    # Generate training summary
    if [[ -f "${LOG_DIR}/training_*.log" ]]; then
        local latest_log=$(ls -t "${LOG_DIR}"/training_*.log | head -n1)
        log "Latest training log: $latest_log"
        
        # Extract key metrics
        grep -E "(epoch|loss|perplexity|validation)" "$latest_log" | tail -20 > "${LOG_DIR}/training_summary.txt"
    fi
    
    log_success "Post-training analysis complete"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Deactivate virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        deactivate
    fi
}

# Main execution
main() {
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Print banner
    print_banner
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --resume)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            --debug)
                DEBUG="true"
                shift
                ;;
            --skip-setup)
                SKIP_SETUP="true"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --config FILE       Use specific config file (default: configs/distillation_config.yaml)"
                echo "  --resume CHECKPOINT Resume training from checkpoint"
                echo "  --debug             Run in debug mode with reduced dataset"
                echo "  --skip-setup        Skip environment setup"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Main execution flow
    log "Starting xinSLM v04 distillation training pipeline"
    log "Project root: $PROJECT_ROOT"
    log "Configuration: $CONFIG_FILE"
    
    check_requirements
    
    if [[ "${SKIP_SETUP:-}" != "true" ]]; then
        setup_environment
    fi
    
    prepare_directories
    check_config
    run_training
    post_training_analysis
    
    log_success "Training pipeline completed successfully!"
    log "Check the following directories for results:"
    log "  - Checkpoints: ${PROJECT_ROOT}/checkpoints"
    log "  - Logs: ${LOG_DIR}"
    log "  - Resource usage: ${LOG_DIR}/resource_usage.log"
}

# Execute main function
main "$@"