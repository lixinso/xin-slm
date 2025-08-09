#!/bin/bash
# Memory-Safe Training Launcher for GPT-OSS MoE on Mac Mini
# This script provides additional safety checks and monitoring

set -e  # Exit on error

echo "🚀 Starting Memory-Safe GPT-OSS MoE Training"
echo "============================================"

# Check if we're in the right directory
if [ ! -f "scripts/safe_train.py" ]; then
    echo "❌ Error: Please run this script from the xinSLM_v06_gpt_oss directory"
    exit 1
fi

# Check Python environment
echo "🔍 Checking Python environment..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "❌ Error: PyTorch not found. Please install requirements first."
    exit 1
}

# Check available memory
echo "🔍 Checking system resources..."
python3 -c "
import psutil
mem = psutil.virtual_memory()
avail_gb = mem.available / (1024**3)
print(f'Available memory: {avail_gb:.1f}GB')
if avail_gb < 6:
    print('⚠️  Warning: Less than 6GB available. Training may fail.')
    import sys
    response = input('Continue anyway? (y/N): ')
    if response.lower() != 'y':
        sys.exit(1)
"

# Check if MPS is available
echo "🔍 Checking Metal Performance Shaders..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Metal Performance Shaders) is available')
else:
    print('❌ MPS not available. Training will be slow on CPU.')
    import sys
    response = input('Continue anyway? (y/N): ')
    if response.lower() != 'y':
        sys.exit(1)
"

# Option to run system check only
if [ "$1" = "--check-only" ]; then
    echo "🔍 Running system readiness check only..."
    python3 scripts/safe_train.py --check-only
    exit $?
fi

# Show configuration being used
echo "📋 Training configuration:"
echo "  Model config: configs/memory_optimized_model_config.yaml"
echo "  Training config: configs/memory_safe_training_config.yaml"
echo "  Model variant: light (150M active parameters)"
echo "  Max memory allocation: ~8GB"
echo ""

# Option to create a backup of current state
if [ "$1" = "--with-backup" ]; then
    echo "💾 Creating backup of current state..."
    backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    cp -r configs models scripts "$backup_dir/"
    echo "✅ Backup created in: $backup_dir"
fi

# Clear any existing logs
echo "🧹 Cleaning up previous logs..."
rm -f training.log safe_training.log memory_monitoring.json

# Warning about system usage
echo ""
echo "⚠️  IMPORTANT RECOMMENDATIONS:"
echo "   • Close other applications to free up memory"
echo "   • Keep the system plugged in (training may take hours)"
echo "   • Don't put the system to sleep during training"
echo "   • Monitor the terminal for memory warnings"
echo ""
echo "📊 Training will be monitored for:"
echo "   • Memory usage (critical threshold: 90%)"
echo "   • Training duration (max: 4 hours)"
echo "   • System stability"
echo ""

# Ask for confirmation
read -p "🚦 Ready to start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled."
    exit 0
fi

# Start the memory-safe training
echo "🎯 Starting memory-safe training..."
echo "   Log files: training.log, safe_training.log"
echo "   Press Ctrl+C to stop training safely"
echo ""

# Run the training with error handling
if python3 scripts/safe_train.py \
    --config configs/memory_safe_training_config.yaml \
    --model-config configs/memory_optimized_model_config.yaml; then
    
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "📁 Check the following directories for results:"
    echo "   • ./checkpoints_memory_safe/ (model checkpoints)"
    echo "   • ./logs_memory_safe/ (training logs)"
    echo ""
    echo "🔍 To test the trained model:"
    echo "   python3 scripts/test_model.py --checkpoint ./checkpoints_memory_safe/best_model.pt"
    
else
    echo ""
    echo "❌ Training failed or was interrupted."
    echo ""
    echo "🔍 Check the following for debugging:"
    echo "   • training.log (detailed training logs)"
    echo "   • safe_training.log (safety monitoring logs)"
    echo ""
    echo "💡 If you encountered memory issues:"
    echo "   • Try the 'ultra_light' model variant"
    echo "   • Close more applications"
    echo "   • Restart the system and try again"
    
    exit 1
fi