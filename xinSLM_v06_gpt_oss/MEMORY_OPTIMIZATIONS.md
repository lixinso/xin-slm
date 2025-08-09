# Memory Optimizations for GPT-OSS MoE Training on Mac Mini

## Problem Analysis
The v06 training failed due to memory exhaustion, causing the Mac Mini to reboot. Analysis revealed:

- **Model size**: 3.09B total parameters, 259M active (MoE architecture)
- **Memory estimate**: ~989MB for model + 4-6GB total during training (optimizer, gradients, activations)
- **Available memory**: Only ~75MB free when training started
- **Result**: System OOM kill and reboot

## Solutions Implemented

### 1. Memory-Optimized Model Configuration
**File**: `configs/memory_optimized_model_config.yaml`

**Key changes**:
- Reduced experts from 32 to 8 (75% reduction)
- Single expert routing instead of top-2
- Smaller model variants available
- Conservative memory limits (10GB max allocation)

**Model variants**:
- **Micro**: ~50M active params (for testing)
- **Light**: ~150M active params (recommended)
- **Standard**: ~250M active params (max safe size)

### 2. Memory-Safe Training Configuration  
**File**: `configs/memory_safe_training_config.yaml`

**Key optimizations**:
- Higher gradient accumulation (64 steps)
- Shorter sequence length (512 tokens)
- Aggressive memory monitoring
- Frequent cache clearing (every 25 steps)
- Conservative memory allocation (8GB limit)

### 3. Ultra-Safe Configuration
**File**: `configs/ultra_safe_training_config.yaml`

**For systems with <6GB available**:
- Micro model variant (2 experts)
- Very short sequences (256 tokens) 
- Minimal training (100 steps for testing)
- 3GB memory allocation limit

### 4. Enhanced Training Script
**File**: `scripts/train_gpt_oss_moe.py`

**Improvements**:
- Integrated resource monitoring
- Memory pressure detection
- Automatic cache clearing
- Memory usage logging
- Graceful error handling

### 5. Memory-Safe Training Wrapper
**File**: `scripts/safe_train.py`

**Safety features**:
- Pre-training system checks
- Real-time memory monitoring  
- Automatic training termination at 90% memory
- Process cleanup and recovery
- Comprehensive logging

### 6. Easy Launch Script
**File**: `start_memory_safe_training.sh`

**Features**:
- System readiness verification
- Memory and MPS checks
- User-friendly interface
- Error handling and reporting
- Post-training analysis

## Usage Instructions

### Option 1: Quick Start (Recommended)
```bash
cd xinSLM_v06_gpt_oss
./start_memory_safe_training.sh
```

### Option 2: Manual Launch
```bash
# Check system first
python3 scripts/safe_train.py --check-only

# Start training with memory-safe config
python3 scripts/safe_train.py \
  --config configs/memory_safe_training_config.yaml \
  --model-config configs/memory_optimized_model_config.yaml
```

### Option 3: Ultra-Safe Mode (Limited Memory)
```bash
python3 scripts/safe_train.py \
  --config configs/ultra_safe_training_config.yaml \
  --model-config configs/memory_optimized_model_config.yaml
```

## Memory Requirements by Configuration

| Configuration | Model Size | Memory Needed | Recommended For |
|---------------|------------|---------------|-----------------|
| Ultra-safe    | ~50M params | 3-4GB | Testing, <6GB available |
| Memory-safe   | ~150M params | 6-8GB | Production, 8GB+ available |
| Standard      | ~250M params | 8-10GB | Performance, 12GB+ available |

## Monitoring and Safety Features

### Automatic Protections
- **Memory monitoring**: Checks every 10-30 seconds
- **Cache clearing**: Every 10-25 steps  
- **Garbage collection**: At memory thresholds
- **Process termination**: At 90% memory usage
- **Time limits**: 4-hour maximum training

### Warning Thresholds
- **75%**: Memory warning logged
- **85%**: Aggressive cleanup triggered  
- **90%**: Training terminated automatically

### Recovery Options
- Automatic model variant downgrade
- CPU offloading for memory relief
- Checkpoint-based recovery
- Graceful shutdown procedures

## Expected Performance

### Memory Usage
- **Micro model**: ~200MB baseline + 2-3GB training overhead
- **Light model**: ~500MB baseline + 4-6GB training overhead  
- **Standard model**: ~800MB baseline + 6-8GB training overhead

### Training Speed
- **Micro**: ~10-15 tokens/sec
- **Light**: ~5-10 tokens/sec
- **Standard**: ~3-7 tokens/sec

## Troubleshooting

### If Training Still Fails
1. **Use ultra-safe config**: Micro model with 100-step test
2. **Close other apps**: Free up more system memory
3. **Restart system**: Clear any memory leaks
4. **Check logs**: Review `safe_training.log` for details

### Common Issues
- **"Insufficient memory"**: Use ultra-safe config or free more RAM
- **"MPS not available"**: Training will be slower on CPU
- **"Training timeout"**: Normal for very large models, adjust time limits

### Log Files
- `training.log`: Detailed training progress
- `safe_training.log`: Memory monitoring and safety events
- `memory_monitoring.json`: Resource usage data

## Next Steps

1. **Test with micro model**: Verify system stability
2. **Scale up gradually**: Try light → standard variants
3. **Monitor performance**: Check memory usage patterns
4. **Optimize further**: Adjust batch size, sequence length as needed

## Files Created/Modified

### New Files
- `configs/memory_optimized_model_config.yaml`
- `configs/memory_safe_training_config.yaml` 
- `configs/ultra_safe_training_config.yaml`
- `scripts/safe_train.py`
- `start_memory_safe_training.sh`
- `MEMORY_OPTIMIZATIONS.md`

### Modified Files
- `scripts/train_gpt_oss_moe.py` (added resource monitoring)

All changes maintain compatibility with existing code while providing comprehensive memory safety for Mac Mini training.