#!/usr/bin/env python3
"""
Memory-Safe Training Wrapper for GPT-OSS MoE Model
Provides additional safety checks and memory management for Mac Mini training
"""

import os
import sys
import time
import psutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import signal
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SAFE_TRAIN - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safe_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemorySafeTrainer:
    """Wrapper that provides memory safety for training"""
    
    def __init__(self, config_path: str, model_config_path: str = None):
        self.config_path = config_path
        self.model_config_path = model_config_path or "configs/memory_optimized_model_config.yaml"
        self.process = None
        self.monitoring_active = False
        self.start_time = None
        
        # Safety thresholds
        self.memory_warning_threshold = 0.8   # 80%
        self.memory_critical_threshold = 0.9  # 90%
        self.temperature_threshold = 80       # Celsius
        self.max_training_time = 4 * 3600    # 4 hours max
        
        logger.info("Memory-safe trainer initialized")
    
    def check_system_readiness(self) -> bool:
        """Check if system is ready for training"""
        logger.info("Checking system readiness...")
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 3.0:  # Need at least 3GB free for ultra-safe mode
            logger.error(f"Insufficient memory: {available_gb:.1f}GB available, need at least 3GB")
            return False
        elif available_gb < 6.0:
            logger.warning(f"Limited memory: {available_gb:.1f}GB available, recommend using ultra-safe config")
        else:
            logger.info(f"Sufficient memory: {available_gb:.1f}GB available")
        
        logger.info(f"Memory check: {available_gb:.1f}GB available ✓")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        
        if free_gb < 5.0:  # Need at least 5GB free for checkpoints
            logger.error(f"Insufficient disk space: {free_gb:.1f}GB free, need at least 5GB")
            return False
        
        logger.info(f"Disk check: {free_gb:.1f}GB free ✓")
        
        # Check if training configs exist
        if not Path(self.config_path).exists():
            logger.error(f"Training config not found: {self.config_path}")
            return False
            
        if not Path(self.model_config_path).exists():
            logger.error(f"Model config not found: {self.model_config_path}")
            return False
        
        logger.info("Configuration files found ✓")
        
        # Check for other intensive processes
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 50:
            logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
            logger.warning("Consider closing other applications before training")
        
        logger.info("System readiness check completed ✓")
        return True
    
    def cleanup_before_training(self):
        """Clean up system resources before training"""
        logger.info("Performing pre-training cleanup...")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear any existing MPS cache
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("MPS cache cleared ✓")
        except:
            pass
        
        # Log current resource usage
        memory = psutil.virtual_memory()
        logger.info(f"Pre-training memory: {memory.percent:.1f}% used")
        logger.info(f"Available memory: {memory.available / (1024**3):.1f}GB")
    
    def monitor_training_process(self):
        """Monitor the training process for memory and safety issues"""
        if not self.process:
            return
        
        while self.process.poll() is None and self.monitoring_active:
            try:
                # Check system memory
                memory = psutil.virtual_memory()
                
                if memory.percent > self.memory_critical_threshold * 100:
                    logger.critical(f"Critical memory usage: {memory.percent:.1f}%")
                    logger.critical("Terminating training to prevent system crash")
                    self.terminate_training("Critical memory usage")
                    break
                
                elif memory.percent > self.memory_warning_threshold * 100:
                    logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
                # Check training time
                if self.start_time and time.time() - self.start_time > self.max_training_time:
                    logger.warning("Maximum training time reached, stopping training")
                    self.terminate_training("Time limit reached")
                    break
                
                # Check CPU temperature (Mac-specific)
                try:
                    # This is a simplified check - real implementation would use system-specific APIs
                    cpu_percent = psutil.cpu_percent(interval=1)
                    if cpu_percent > 90:
                        logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                except:
                    pass
                
                # Log periodic status
                logger.debug(f"Monitoring: Memory={memory.percent:.1f}%, Process running")
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def terminate_training(self, reason: str):
        """Safely terminate the training process"""
        logger.warning(f"Terminating training: {reason}")
        
        if self.process:
            try:
                # Send SIGTERM first for graceful shutdown
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=30)
                    logger.info("Training process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning("Force killing training process")
                    self.process.kill()
                    self.process.wait()
            
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
        
        self.monitoring_active = False
    
    def start_training(self) -> bool:
        """Start the training process with monitoring"""
        if not self.check_system_readiness():
            return False
        
        self.cleanup_before_training()
        
        # Prepare training command
        training_script = Path(__file__).parent / "train_gpt_oss_moe.py"
        cmd = [
            sys.executable, 
            str(training_script),
            "--config", self.config_path
        ]
        
        logger.info(f"Starting training: {' '.join(cmd)}")
        
        try:
            # Start training process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.start_time = time.time()
            self.monitoring_active = True
            
            logger.info(f"Training process started (PID: {self.process.pid})")
            
            # Start monitoring in a separate thread
            import threading
            monitor_thread = threading.Thread(target=self.monitor_training_process, daemon=True)
            monitor_thread.start()
            
            # Stream output and log it
            for line in self.process.stdout:
                line = line.strip()
                if line:
                    print(line)  # Print to console
                    
                    # Log important messages
                    if "ERROR" in line or "CRITICAL" in line:
                        logger.error(f"Training: {line}")
                    elif "WARNING" in line:
                        logger.warning(f"Training: {line}")
                    elif "Step" in line and "Loss" in line:
                        logger.info(f"Training: {line}")
            
            # Wait for process completion
            return_code = self.process.wait()
            self.monitoring_active = False
            
            if return_code == 0:
                logger.info("Training completed successfully ✓")
                return True
            else:
                logger.error(f"Training failed with return code: {return_code}")
                return False
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.terminate_training("User interrupt")
            return False
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            if self.process:
                self.terminate_training("Unexpected error")
            return False
        
        finally:
            self.monitoring_active = False
            if self.process:
                try:
                    self.process.terminate()
                except:
                    pass
    
    def post_training_report(self):
        """Generate post-training report"""
        logger.info("=" * 60)
        logger.info("POST-TRAINING REPORT")
        logger.info("=" * 60)
        
        # Memory status
        memory = psutil.virtual_memory()
        logger.info(f"Final memory usage: {memory.percent:.1f}%")
        logger.info(f"Available memory: {memory.available / (1024**3):.1f}GB")
        
        # Check for checkpoints
        checkpoint_dirs = [
            "./checkpoints_memory_safe",
            "./checkpoints_prod",
            "./checkpoints"
        ]
        
        for checkpoint_dir in checkpoint_dirs:
            if Path(checkpoint_dir).exists():
                checkpoints = list(Path(checkpoint_dir).glob("*.pt"))
                if checkpoints:
                    logger.info(f"Checkpoints found in {checkpoint_dir}: {len(checkpoints)}")
                    latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    logger.info(f"Latest checkpoint: {latest}")
        
        # Training duration
        if self.start_time:
            duration = time.time() - self.start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            logger.info(f"Training duration: {hours}h {minutes}m")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Memory-safe trainer for GPT-OSS MoE")
    parser.add_argument(
        "--config", 
        default="configs/memory_safe_training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--model-config", 
        default="configs/memory_optimized_model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check system readiness, don't start training"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MemorySafeTrainer(args.config, args.model_config)
    
    if args.check_only:
        success = trainer.check_system_readiness()
        sys.exit(0 if success else 1)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        trainer.terminate_training("Shutdown signal")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start training
    success = trainer.start_training()
    trainer.post_training_report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()