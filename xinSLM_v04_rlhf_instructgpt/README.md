# SLM v04: RLHF Implementation (InstructGPT-style)

Implementation of Reinforcement Learning from Human Feedback (RLHF) following the InstructGPT methodology by Ouyang et al., 2022.

## Overview

This implementation follows the three-step RLHF process:

1. **Supervised Fine-tuning (SFT)**: Train on human demonstrations
2. **Reward Model Training**: Learn human preferences from comparisons
3. **PPO Optimization**: Use reinforcement learning to optimize policy

## Key Features

- **Reward Model**: Bradley-Terry preference learning
- **PPO Implementation**: Proximal Policy Optimization for stable training
- **KL Divergence Control**: Prevent model drift from original behavior
- **Instruction Following**: Optimized for following human instructions
- **Scalable Architecture**: Efficient implementation for smaller models

## Model Architecture

- Based on SLM v03 architecture (25M parameters)
- Optimized for instruction following tasks
- Reward model with scalar output head
- PPO-compatible policy and value heads

## Training Pipeline

1. **SFT Phase**: Fine-tune on instruction-response pairs
2. **Reward Phase**: Train reward model on preference comparisons
3. **RLHF Phase**: PPO optimization using reward model feedback

## Theoretical Background

### Reinforcement Learning from Human Feedback (RLHF)
RLHF is a paradigm that trains AI systems to align with human preferences by combining reinforcement learning with human feedback. Instead of optimizing traditional metrics like perplexity or BLEU scores, RLHF learns directly from human judgments about which outputs are better. The process involves three stages: first, supervised fine-tuning creates a baseline model; second, a reward model learns to predict human preferences from comparison data; and third, reinforcement learning optimizes the policy using the learned reward function while preventing the model from deviating too far from its original behavior through KL divergence constraints.

### Supervised Fine-Tuning (SFT)  
Supervised Fine-Tuning is the first stage of RLHF where a pre-trained language model is fine-tuned on a curated dataset of high-quality instruction-response pairs. This process teaches the model to follow instructions and produce helpful responses by learning from human demonstrations. SFT establishes the foundation for instruction-following behavior by exposing the model to diverse examples of how humans expect AI assistants to respond to various prompts, creating a baseline policy that can then be further improved through preference learning and reinforcement learning.

### Reward Model
The reward model is a neural network trained to predict human preferences by learning from comparison data where humans rank different responses to the same prompt. Using the Bradley-Terry model for preference learning, it takes a prompt-response pair as input and outputs a scalar reward score indicating the quality of the response. The reward model essentially learns to internalize human judgment, serving as an automated evaluator that can provide feedback signals during reinforcement learning. This allows the system to optimize for human preferences at scale without requiring constant human evaluation during training.

### Proximal Policy Optimization (PPO)
PPO is a reinforcement learning algorithm that optimizes the language model policy using rewards from the reward model while maintaining training stability through careful constraint mechanisms. PPO uses a clipped objective function that prevents the policy from changing too dramatically in a single update, avoiding the instability issues common in policy gradient methods. In the RLHF context, PPO balances maximizing rewards (human preference scores) with a KL divergence penalty that keeps the optimized model close to the original SFT model, ensuring the model improves its responses without losing coherence or developing unexpected behaviors.

## Key Innovations from InstructGPT

- Human preference learning over traditional metrics
- KL penalty to maintain coherent behavior
- Multi-step training process for alignment
- Robust evaluation on instruction-following tasks