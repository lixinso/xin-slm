#!/bin/bash

echo "🚀 Setting up Hugging Face publishing environment"
echo "=================================================="

# Install required packages
echo "📦 Installing required packages..."
pip install huggingface_hub transformers accelerate

# Check if user is logged in to Hugging Face
echo ""
echo "🔐 Checking Hugging Face authentication..."
if huggingface-cli whoami &>/dev/null; then
    echo "✅ Already logged in to Hugging Face"
    huggingface-cli whoami
else
    echo "❌ Not logged in to Hugging Face"
    echo ""
    echo "To publish models, you need to:"
    echo "1. Create account at https://huggingface.co/join"
    echo "2. Get your token at https://huggingface.co/settings/tokens"
    echo "3. Run: huggingface-cli login"
    echo "   OR set environment variable: export HF_TOKEN=your_token_here"
    echo ""
    read -p "Would you like to login now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 To publish your model, run:"
echo "python publish_to_huggingface.py \\"
echo "  --checkpoint checkpoints_ultra_safe/best_model.pt \\"
echo "  --repo-name xinslm-gpt-oss-moe-micro"
echo ""
echo "🌐 Your model will be published at:"
echo "https://huggingface.co/lixinso/xinslm-gpt-oss-moe-micro"
echo ""
echo "📚 Optional parameters:"
echo "  --private          # Make repository private"
echo "  --organization ORG # Upload to organization"
echo "  --token TOKEN      # Use specific token"
echo ""
echo "💡 Example with private repo:"
echo "python publish_to_huggingface.py \\"
echo "  --checkpoint checkpoints_ultra_safe/best_model.pt \\"
echo "  --repo-name xinslm-gpt-oss-moe-micro \\"
echo "  --private"