# Hugging Face Publishing Guide for xinSLM GPT-OSS MoE

This guide helps you publish your trained GPT-OSS MoE model to Hugging Face Hub under the username **lixinso**.

## Quick Start

### 1. Setup Environment
```bash
./setup_huggingface.sh
```

### 2. Publish Your Model
```bash
python publish_to_huggingface.py \
  --checkpoint checkpoints_ultra_safe/best_model.pt \
  --repo-name xinslm-gpt-oss-moe-micro
```

### 3. Your Model URL
After publishing, your model will be available at:
**https://huggingface.co/lixinso/xinslm-gpt-oss-moe-micro**

## Authentication Options

### Option A: Interactive Login
```bash
huggingface-cli login
```

### Option B: Environment Variable
```bash
export HF_TOKEN=your_token_here
python publish_to_huggingface.py --checkpoint checkpoints_ultra_safe/best_model.pt --repo-name xinslm-gpt-oss-moe-micro
```

### Option C: Direct Token
```bash
python publish_to_huggingface.py \
  --checkpoint checkpoints_ultra_safe/best_model.pt \
  --repo-name xinslm-gpt-oss-moe-micro \
  --token your_token_here
```

## Publishing Variations

### Public Model (Default)
```bash
python publish_to_huggingface.py \
  --checkpoint checkpoints_ultra_safe/best_model.pt \
  --repo-name xinslm-gpt-oss-moe-micro
```

### Private Model
```bash
python publish_to_huggingface.py \
  --checkpoint checkpoints_ultra_safe/best_model.pt \
  --repo-name xinslm-gpt-oss-moe-micro \
  --private
```

### Different Model Variants
For future model variants, use different repository names:

```bash
# Light variant (when you train it)
python publish_to_huggingface.py \
  --checkpoint checkpoints_light/best_model.pt \
  --repo-name xinslm-gpt-oss-moe-light

# Standard variant (when you train it)
python publish_to_huggingface.py \
  --checkpoint checkpoints_standard/best_model.pt \
  --repo-name xinslm-gpt-oss-moe-standard
```

## What Gets Published

The script will create a complete Hugging Face repository with:

### 📁 Model Files
- `pytorch_model.bin` - Your trained model weights
- `config.json` - Model configuration
- `tokenizer.json` + `tokenizer_config.json` - GPT-2 tokenizer files

### 📄 Documentation
- `README.md` - Complete model card with:
  - Architecture details (40.9M active params, 4 experts)
  - Training metrics (279 steps, 8.71 final loss)
  - Performance benchmarks (6+ tokens/sec)
  - Usage examples
  - Limitations and recommendations

### 🏷️ Metadata
- License: Apache 2.0
- Tags: mixture-of-experts, moe, gpt, mac-optimized
- Pipeline: text-generation

## Using Your Published Model

Once published, anyone can use your model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model
model = AutoModelForCausalLM.from_pretrained("lixinso/xinslm-gpt-oss-moe-micro")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=100, temperature=0.7)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

## Model Information

Your model will be published with these details:

- **Username**: lixinso
- **Model Name**: xinslm-gpt-oss-moe-micro
- **Full URL**: https://huggingface.co/lixinso/xinslm-gpt-oss-moe-micro
- **Architecture**: GPT-OSS Mixture of Experts
- **Parameters**: 97.5M total, 40.9M active
- **Experts**: 4 experts, 1 active per token
- **Training**: 279 steps, Mac Mini optimized
- **Performance**: 6+ tokens/sec on Apple Silicon

## Troubleshooting

### Token Issues
- Get your token at: https://huggingface.co/settings/tokens
- Make sure you have "Write" permissions

### Model Too Large
- Current model: ~1.17GB (should upload fine)
- If issues, try `--private` flag first

### Authentication Errors
```bash
# Re-login if needed
huggingface-cli logout
huggingface-cli login
```

### Repository Already Exists
The script will update existing repositories, so you can run it multiple times safely.

## Next Steps After Publishing

1. **Test the Model**: Try loading it from Hugging Face to verify upload
2. **Share the Link**: Your model will be at https://huggingface.co/lixinso/xinslm-gpt-oss-moe-micro
3. **Update Documentation**: Edit the model card on Hugging Face web interface if needed
4. **Train Larger Variants**: Use the same script for light/standard variants when ready

## Example Complete Workflow

```bash
# 1. Setup (one-time)
./setup_huggingface.sh

# 2. Login (one-time)
huggingface-cli login

# 3. Publish
python publish_to_huggingface.py \
  --checkpoint checkpoints_ultra_safe/best_model.pt \
  --repo-name xinslm-gpt-oss-moe-micro

# 4. Test (verify it works)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('lixinso/xinslm-gpt-oss-moe-micro')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print('✅ Model loaded successfully from Hugging Face!')
"
```

Ready to publish your model to the world! 🚀