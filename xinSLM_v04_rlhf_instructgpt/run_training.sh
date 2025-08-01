#!/bin/bash

# RLHF Training Script for xinSLM v04
# Run the complete InstructGPT-style RLHF pipeline

echo "======================================"
echo "xinSLM v04 RLHF Training Pipeline"
echo "======================================"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "Error: PyTorch not found. Please install requirements:"
    echo "pip install -r requirements.txt"
    exit 1
}

# Run integration test first
echo ""
echo "Running integration tests..."
python3 test_pipeline.py

if [ $? -ne 0 ]; then
    echo "Integration tests failed. Please fix issues before training."
    exit 1
fi

echo ""
echo "Integration tests passed! Starting RLHF training..."
echo ""

# Run RLHF training with fast configuration
# Change to --config default for full training
python3 train_rlhf.py --config fast --stage all

echo ""
echo "Training completed!"
echo "Check the following directories for outputs:"
echo "  - sft_checkpoints/ : Supervised fine-tuning models"
echo "  - reward_checkpoints/ : Reward model checkpoints"
echo "  - ppo_checkpoints/ : PPO-trained final models"
echo ""
echo "Training logs and metrics:"
echo "  - rlhf_training.log : Complete training log"
echo "  - sft_metrics.json : SFT training metrics"
echo "  - reward_metrics.json : Reward model metrics"
echo "  - ppo_metrics.json : PPO training metrics"