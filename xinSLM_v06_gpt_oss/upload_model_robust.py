#!/usr/bin/env python3
"""
Robust Model Upload to Hugging Face Hub
Handles network issues and large files with retry mechanism
"""

import os
import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import HfApi, login, upload_file
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.gpt_oss_moe import create_gpt_oss_moe

def upload_with_retry(file_path, repo_id, path_in_repo, token=None, max_retries=5):
    """Upload file with retry mechanism"""
    
    for attempt in range(max_retries):
        try:
            print(f"📤 Uploading {path_in_repo} (attempt {attempt + 1}/{max_retries})...")
            
            result = upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                repo_type="model"
            )
            
            print(f"✅ Successfully uploaded {path_in_repo}")
            return result
            
        except Exception as e:
            print(f"❌ Upload attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"💥 All {max_retries} upload attempts failed")
                raise e

def create_simple_model_files(checkpoint_path: str, output_dir: Path):
    """Create simplified model files for upload"""
    
    print(f"🔄 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        training_config = config_dict.get('training', {})
        model_variant = training_config.get('model_variant', 'micro')
        variant_config = config_dict.get('model_variants', {}).get(model_variant, {})
    else:
        # Default micro configuration
        variant_config = {
            'hidden_size': 384,
            'intermediate_size': 1024,
            'num_hidden_layers': 8,
            'num_attention_heads': 6,
            'num_key_value_heads': 1,
            'num_experts': 4,
            'num_experts_per_tok': 1,
            'reasoning_effort': 'low'
        }
        model_variant = 'micro'
    
    # Save just the state dict as PyTorch model
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    print("💾 Saving PyTorch model...")
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    
    # Create simple config.json
    config_json = {
        "architectures": ["GPTOSSMoEModel"],
        "model_type": "gpt_oss_moe",
        "vocab_size": 50257,
        "hidden_size": variant_config.get('hidden_size', 384),
        "intermediate_size": variant_config.get('intermediate_size', 1024),
        "num_hidden_layers": variant_config.get('num_hidden_layers', 8),
        "num_attention_heads": variant_config.get('num_attention_heads', 6),
        "num_key_value_heads": variant_config.get('num_key_value_heads', 1),
        "max_position_embeddings": 512,
        "num_experts": variant_config.get('num_experts', 4),
        "num_experts_per_tok": variant_config.get('num_experts_per_tok', 1),
        "reasoning_effort": variant_config.get('reasoning_effort', 'low'),
        "use_cache": True,
        "pad_token_id": 50256,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "torch_dtype": "float32",
        "transformers_version": "4.54.0"
    }
    
    import json
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    
    # Create README.md
    readme_content = f"""---
license: apache-2.0
tags:
- mixture-of-experts
- moe
- gpt
- language-model
- mac-optimized
language:
- en
pipeline_tag: text-generation
---

# GPT-OSS MoE - {model_variant.title()} Variant

A Mixture of Experts (MoE) language model optimized for Mac Mini (16GB RAM).

## Model Details
- **Variant**: {model_variant.title()}
- **Active Parameters**: ~{variant_config.get('num_experts', 4) * 10}M
- **Total Experts**: {variant_config.get('num_experts', 4)}
- **Experts per Token**: {variant_config.get('num_experts_per_tok', 1)}
- **Hidden Size**: {variant_config.get('hidden_size', 384)}
- **Layers**: {variant_config.get('num_hidden_layers', 8)}

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load model state dict
state_dict = torch.load("pytorch_model.bin", map_location="cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Note: This model requires the custom GPT-OSS MoE architecture
# See: https://github.com/xin-slm/xinSLM_v06_gpt_oss
```

## Training Details
- Platform: Mac Mini 16GB, Apple Silicon MPS
- Memory Optimization: Real-time monitoring and cleanup
- Architecture: GPT-OSS with reduced experts for memory efficiency

## Citation
```bibtex
@misc{{xinslm-gpt-oss-moe-{model_variant},
    title={{GPT-OSS MoE {model_variant.title()}: Memory-Optimized Mixture of Experts}},
    author={{Xinson Li}},
    year={{2025}},
    url={{https://huggingface.co/lixinso/xinslm-gpt-oss-moe-{model_variant}}}
}}
```
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Created model files in {output_dir}")
    return ["pytorch_model.bin", "config.json", "README.md"]

def main():
    parser = argparse.ArgumentParser(description="Robust upload to Hugging Face Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--repo-name", required=True, help="Repository name")
    parser.add_argument("--login", action="store_true", help="Use login")
    
    args = parser.parse_args()
    
    print("🚀 Starting Robust Model Upload")
    print("=" * 50)
    
    if args.login:
        print("🔐 Please authenticate with Hugging Face...")
        try:
            login()
            print("✅ Authentication successful!")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return
    
    # Get repository info
    try:
        api = HfApi()
        user_info = api.whoami()
        username = user_info["name"]
        full_repo_name = f"{username}/{args.repo_name}"
        print(f"📋 Target repository: {full_repo_name}")
    except Exception as e:
        print(f"❌ Could not get user info: {e}")
        return
    
    # Create temporary directory
    temp_dir = Path("temp_upload")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create model files
        files_to_upload = create_simple_model_files(args.checkpoint, temp_dir)
        
        print(f"\n⬆️ Uploading {len(files_to_upload)} files to {full_repo_name}")
        
        # Upload each file with retry
        for filename in files_to_upload:
            file_path = temp_dir / filename
            
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"\n📤 Uploading {filename} ({file_size_mb:.1f} MB)...")
                
                upload_with_retry(
                    file_path=file_path,
                    repo_id=full_repo_name,
                    path_in_repo=filename,
                    max_retries=3
                )
            else:
                print(f"⚠️ File not found: {filename}")
        
        print(f"\n🎉 Upload completed successfully!")
        print(f"🔗 Model available at: https://huggingface.co/{full_repo_name}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temporary files")

if __name__ == "__main__":
    main()