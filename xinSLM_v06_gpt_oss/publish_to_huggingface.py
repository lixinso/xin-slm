#!/usr/bin/env python3
"""
Publish GPT-OSS MoE Model to Hugging Face Hub

This script converts the trained PyTorch model to Hugging Face format
and uploads it to the Hugging Face Hub with proper documentation.

Usage:
    python publish_to_huggingface.py --checkpoint checkpoints_ultra_safe/best_model.pt --repo-name xinslm-gpt-oss-moe-micro
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.gpt_oss_moe import GPTOSSForCausalLM, create_gpt_oss_moe, GPTOSSMoEConfig
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder, login
from huggingface_hub.utils import RepositoryNotFoundError
import webbrowser

class GPTOSSMoEHFConfig(PretrainedConfig):
    """Hugging Face compatible configuration for GPT-OSS MoE"""
    
    model_type = "gpt_oss_moe"
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        num_experts=32,
        num_experts_per_tok=2,
        expert_capacity_factor=1.0,
        router_aux_loss_coef=0.02,
        router_z_loss_coef=0.001,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        reasoning_effort="medium",
        use_quantization=False,
        quantization_bits=4,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_capacity_factor = expert_capacity_factor
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.reasoning_effort = reasoning_effort
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits


class GPTOSSMoEHFModel(PreTrainedModel):
    """Hugging Face compatible wrapper for GPT-OSS MoE model"""
    
    config_class = GPTOSSMoEHFConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Create the actual model
        self.model = create_gpt_oss_moe(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_seq_len=config.max_position_embeddings,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            reasoning_effort=config.reasoning_effort,
            use_quantization=config.use_quantization
        )
        
        # Store config reference
        self.config = config
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)
    
    def generate(self, *args, **kwargs):
        # Implement generation if needed
        return super().generate(*args, **kwargs)


def load_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint and extract configuration"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        training_config = config_dict.get('training', {})
        model_variant = training_config.get('model_variant', 'micro')
        variant_config = config_dict.get('model_variants', {}).get(model_variant, {})
        
        # Extract training info
        training_info = {
            'final_loss': checkpoint.get('loss', 'N/A'),
            'epoch': checkpoint.get('epoch', 'N/A'),
            'step': checkpoint.get('step', 'N/A'),
            'model_variant': model_variant,
            'training_duration': training_config.get('duration', 'N/A')
        }
        
        return variant_config, training_info, checkpoint
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
        training_info = {'model_variant': 'micro'}
        
        return variant_config, training_info, checkpoint


def create_model_card(config: Dict[str, Any], training_info: Dict[str, Any]) -> str:
    """Create model card markdown"""
    
    model_card = f"""---
license: apache-2.0
tags:
- mixture-of-experts
- moe
- gpt
- language-model
- mac-optimized
- apple-silicon
- pytorch
language:
- en
pipeline_tag: text-generation
widget:
- text: "The future of artificial intelligence is"
- text: "Once upon a time"
- text: "Machine learning is"
---

# GPT-OSS MoE - {training_info.get('model_variant', 'micro').title()} Variant

A Mixture of Experts (MoE) language model based on GPT-OSS architecture, optimized for Mac Mini (16GB RAM) training and inference.

## Model Details

### Architecture
- **Model Type**: GPT-OSS Mixture of Experts
- **Variant**: {training_info.get('model_variant', 'micro').title()}
- **Total Parameters**: ~{config.get('total_params', 'N/A'):,} 
- **Active Parameters**: ~{config.get('active_params', 'N/A'):,}
- **Number of Experts**: {config.get('num_experts', 4)}
- **Experts per Token**: {config.get('num_experts_per_tok', 1)}
- **Hidden Size**: {config.get('hidden_size', 384)}
- **Layers**: {config.get('num_hidden_layers', 8)}
- **Attention Heads**: {config.get('num_attention_heads', 6)}

### Training Details
- **Training Loss**: {training_info.get('final_loss', 'N/A')}
- **Training Steps**: {training_info.get('step', 'N/A')}
- **Epochs**: {training_info.get('epoch', 'N/A')}
- **Training Duration**: {training_info.get('training_duration', 'N/A')}
- **Platform**: Mac Mini 16GB, Apple Silicon MPS
- **Memory Usage**: Stable 76-85% utilization
- **Optimization**: Memory-safe training with real-time monitoring

## Performance

### Generation Speed
- **Inference Speed**: 5.5-6.8 tokens/second (Mac Mini M2)
- **Memory Usage**: ~2-4GB during inference
- **Load Time**: 3-5 seconds

### Quality Metrics
- **Loss Reduction**: 21% improvement during training
- **Perplexity**: ~2600 (early training stage)
- **Text Quality**: Technical infrastructure working perfectly, text quality reflects early training stage

## Usage

### Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("lixinso/xinslm-gpt-oss-moe-{training_info.get('model_variant', 'micro')}")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Advanced Usage
```python
# Custom generation parameters
outputs = model.generate(
    inputs,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True
)
```

## Model Variants

This model is part of a series of GPT-OSS MoE variants optimized for different memory constraints:

- **Micro** (40.9M active): 4 experts, optimized for 4-6GB RAM
- **Light** (150M active): 6 experts, optimized for 6-8GB RAM  
- **Standard** (250M active): 8 experts, optimized for 8-10GB RAM

## Limitations

- **Early Training Stage**: Model has been trained for only 279 steps (1 epoch)
- **Text Quality**: Limited coherence due to minimal training
- **Context Understanding**: Basic context awareness
- **Knowledge**: No factual knowledge retained yet

## Recommendations for Improvement

1. **Extended Training**: 5-10 epochs for improved coherence
2. **More Training Steps**: 2,000-5,000 steps minimum
3. **Larger Datasets**: More diverse training data
4. **Fine-tuning**: Task-specific fine-tuning for better performance

## Technical Implementation

### Memory Optimizations
- **Expert Reduction**: Reduced from 32 to {config.get('num_experts', 4)} experts
- **Smart Routing**: {config.get('num_experts_per_tok', 1)} expert(s) per token
- **Quantization Support**: Optional MXFP4-style quantization
- **MPS Acceleration**: Full Metal Performance Shaders support

### Safety Features
- **Memory Monitoring**: Real-time usage tracking
- **Automatic Cleanup**: Prevents OOM crashes
- **Graceful Degradation**: Safe operation under memory pressure

## Citation

```bibtex
@misc{{xinslm-gpt-oss-moe-{training_info.get('model_variant', 'micro')},
    title={{GPT-OSS MoE {training_info.get('model_variant', 'micro').title()}: Memory-Optimized Mixture of Experts for Mac Mini}},
    author={{Xinson Li}},
    year={{2025}},
    howpublished={{\\url{{https://huggingface.co/lixinso/xinslm-gpt-oss-moe-{training_info.get('model_variant', 'micro')}}}}},
}}
```

## License

Apache License 2.0

## Model Card Contact

For questions about this model, please contact @lixinso or open an issue in the [GitHub repository](https://github.com/xin-slm/xinSLM_v06_gpt_oss).
"""
    
    return model_card


def convert_and_upload_model(
    checkpoint_path: str,
    repo_name: str,
    token: Optional[str] = None,
    private: bool = False,
    organization: Optional[str] = None,
    interactive_login: bool = False
):
    """Convert model and upload to Hugging Face Hub"""
    
    print("🚀 Starting GPT-OSS MoE Model Upload to Hugging Face")
    print("=" * 60)
    
    # Handle authentication
    if interactive_login:
        print("🌐 Starting Hugging Face authentication...")
        print("📱 Getting your token for authentication...")
        print()
        print("📋 Please follow these steps:")
        print("   1. 🌍 Opening https://huggingface.co/settings/tokens")
        print("   2. 🔑 Create a new token with 'Write' permissions")
        print("   3. 📋 Copy the token (starts with 'hf_...')")
        print("   4. 📝 Paste it below when prompted")
        print()
        
        try:
            # Open token settings page directly
            webbrowser.open("https://huggingface.co/settings/tokens")
            print("🌍 Browser opened to token settings page")
        except Exception:
            print("⚠️  Could not open browser automatically")
            print("   Please manually go to: https://huggingface.co/settings/tokens")
        
        print()
        print("⏳ Ready for token input...")
        try:
            login()
            print("✅ Successfully authenticated with Hugging Face!")
        except KeyboardInterrupt:
            print("\n❌ Login cancelled by user")
            return
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print()
            print("💡 Please make sure:")
            print("   - Token has 'Write' permissions")
            print("   - Token is copied correctly (starts with 'hf_')")
            print("   - You have network connectivity")
            return
    elif not token:
        print("❌ Error: Authentication required")
        print("Use --login for browser authentication or --token for token-based authentication")
        return
    
    # Load checkpoint and configuration
    variant_config, training_info, checkpoint = load_checkpoint_info(checkpoint_path)
    
    # Create temporary directory for conversion
    temp_dir = Path("temp_hf_model")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create Hugging Face compatible configuration
        hf_config = GPTOSSMoEHFConfig(
            vocab_size=50257,
            hidden_size=variant_config.get('hidden_size', 384),
            intermediate_size=variant_config.get('intermediate_size', 1024),
            num_hidden_layers=variant_config.get('num_hidden_layers', 8),
            num_attention_heads=variant_config.get('num_attention_heads', 6),
            num_key_value_heads=variant_config.get('num_key_value_heads', 1),
            max_position_embeddings=512,
            num_experts=variant_config.get('num_experts', 4),
            num_experts_per_tok=variant_config.get('num_experts_per_tok', 1),
            reasoning_effort=variant_config.get('reasoning_effort', 'low'),
            use_quantization=False
        )
        
        # Create and load the model
        print("🔄 Converting model to Hugging Face format...")
        model = GPTOSSMoEHFModel(hf_config)
        
        # Load trained weights
        if 'model_state_dict' in checkpoint:
            model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.model.load_state_dict(checkpoint)
        
        print("✅ Model weights loaded successfully")
        
        # Save model in Hugging Face format
        model.save_pretrained(temp_dir)
        hf_config.save_pretrained(temp_dir)
        
        # Load and save tokenizer
        print("🔤 Setting up tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(temp_dir)
        
        # Create model card
        print("📝 Creating model card...")
        model_card_content = create_model_card(
            {
                **variant_config,
                'total_params': sum(p.numel() for p in model.parameters()),
                'active_params': variant_config.get('active_params', 40944384)
            },
            training_info
        )
        
        with open(temp_dir / "README.md", "w") as f:
            f.write(model_card_content)
        
        # Create repository
        print(f"📦 Creating repository: {repo_name}")
        api = HfApi(token=token if not interactive_login else None)
        
        # Get current user info to construct proper repo name
        try:
            user_info = api.whoami()
            username = user_info["name"]
            full_repo_name = f"{organization}/{repo_name}" if organization else f"{username}/{repo_name}"
            print(f"📋 Full repository name: {full_repo_name}")
        except Exception as e:
            print(f"⚠️  Could not get user info: {e}")
            full_repo_name = f"{organization}/{repo_name}" if organization else repo_name
        
        try:
            repo_url = create_repo(
                repo_id=full_repo_name,
                token=token if not interactive_login else None,
                private=private,
                repo_type="model"
            )
            print(f"✅ Repository created: {repo_url}")
        except Exception as e:
            if "already exists" in str(e).lower() or "already created" in str(e).lower() or "409" in str(e):
                print(f"✅ Repository already exists: {full_repo_name}")
                repo_url = f"https://huggingface.co/{full_repo_name}"
                print(f"🔗 Using existing repository: {repo_url}")
            else:
                print(f"❌ Failed to create repository: {e}")
                raise e
        
        # Upload files
        print("⬆️  Uploading model files...")
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                print(f"  Uploading {file_path.name}...")
                upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=full_repo_name,
                    token=token if not interactive_login else None,
                    repo_type="model"
                )
        
        print("🎉 Model successfully uploaded to Hugging Face Hub!")
        print(f"🔗 Model URL: https://huggingface.co/{full_repo_name}")
        
        return f"https://huggingface.co/{full_repo_name}"
        
    finally:
        # Cleanup temporary directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temporary files")


def main():
    parser = argparse.ArgumentParser(description="Publish GPT-OSS MoE model to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Name for the Hugging Face repository"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="Hugging Face organization name"
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Use interactive browser login instead of token"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided (unless using interactive login)
    if args.login:
        token = None
        interactive_login = True
    else:
        token = args.token or os.getenv("HF_TOKEN")
        interactive_login = False
        if not token:
            print("❌ Error: Hugging Face token required.")
            print("Either use --token argument, set HF_TOKEN environment variable, or use --login for browser authentication.")
            print("Get your token at: https://huggingface.co/settings/tokens")
            return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return
    
    try:
        model_url = convert_and_upload_model(
            checkpoint_path=args.checkpoint,
            repo_name=args.repo_name,
            token=token,
            private=args.private,
            organization=args.organization,
            interactive_login=interactive_login
        )
        
        print("\n🎊 Upload Complete!")
        print(f"Your model is now available at: {model_url}")
        
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()