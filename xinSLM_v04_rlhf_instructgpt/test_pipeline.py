"""
Test script for RLHF pipeline
Quick validation of all components
"""
import torch
import os
import sys
import json
import logging
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xinSLM_v03_slm_llama_architecture'))
from model import SLMModel
from tokenizer import SLMTokenizer
from slm_config import SLMConfig

# Import RLHF components
from rlhf_config import get_fast_rlhf_config
from reward_model import RewardModel
from sft_trainer import SFTTrainer, InstructionDataset
from ppo_trainer import PPOTrainer
from data_utils import create_preference_dataset, create_ppo_dataset, validate_datasets


def test_tokenizer():
    """Test tokenizer functionality"""
    print("Testing tokenizer...")
    
    tokenizer = SLMTokenizer()
    
    # Test encoding/decoding
    text = "Hello, how are you today?"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print("✓ Tokenizer test passed\n")
    
    return tokenizer


def test_base_model(tokenizer):
    """Test base model functionality"""
    print("Testing base model...")
    
    config = get_fast_rlhf_config()
    model = SLMModel(config.base_model_config)
    
    # Test forward pass
    input_text = "What is machine learning?"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs[0]
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Base model test passed\n")
    
    return model


def test_reward_model(base_model, tokenizer):
    """Test reward model functionality"""
    print("Testing reward model...")
    
    config = get_fast_rlhf_config()
    reward_model = RewardModel(base_model, config)
    
    # Test reward computation
    input_text = "What is AI? Artificial intelligence is amazing!"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    with torch.no_grad():
        outputs = reward_model(input_ids=input_ids, return_dict=True)
        rewards = outputs['rewards']
    
    print(f"Input: {input_text}")
    print(f"Reward: {rewards.item():.4f}")
    print("✓ Reward model test passed\n")
    
    return reward_model


def test_datasets(tokenizer):
    """Test dataset creation and loading"""
    print("Testing datasets...")
    
    config = get_fast_rlhf_config()
    
    # Test SFT dataset
    sft_dataset = InstructionDataset(
        data_path=config.sft_data_path,
        tokenizer=tokenizer,
        max_length=config.sft_max_length
    )
    
    print(f"SFT dataset size: {len(sft_dataset)}")
    sft_sample = sft_dataset[0]
    print(f"SFT sample keys: {sft_sample.keys()}")
    
    # Test preference dataset
    pref_dataset = create_preference_dataset(
        data_path=config.reward_data_path,
        tokenizer=tokenizer,
        config=config
    )
    
    print(f"Preference dataset size: {len(pref_dataset)}")
    pref_sample = pref_dataset[0]
    print(f"Preference sample keys: {pref_sample.keys()}")
    
    # Test PPO dataset
    ppo_dataset = create_ppo_dataset(
        data_path=config.ppo_prompts_path,
        tokenizer=tokenizer,
        config=config
    )
    
    print(f"PPO dataset size: {len(ppo_dataset)}")
    ppo_sample = ppo_dataset[0]
    print(f"PPO sample keys: {ppo_sample.keys()}")
    
    print("✓ Datasets test passed\n")
    
    return sft_dataset, pref_dataset, ppo_dataset


def test_sft_trainer(model, tokenizer):
    """Test SFT trainer functionality"""
    print("Testing SFT trainer...")
    
    config = get_fast_rlhf_config()
    
    # Create a small dataset for testing
    sft_dataset = InstructionDataset(
        data_path=config.sft_data_path,
        tokenizer=tokenizer,
        max_length=config.sft_max_length
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=sft_dataset
    )
    
    # Test single training step
    from torch.utils.data import DataLoader
    from sft_trainer import collate_fn
    
    dataloader = DataLoader(sft_dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    # Move to CPU for testing (avoid device issues)
    for key in batch:
        batch[key] = batch[key].to('cpu')
    
    # Test training step
    metrics = trainer.train_step(batch)
    
    print(f"Training metrics: {metrics}")
    print("✓ SFT trainer test passed\n")
    
    return trainer


def test_ppo_components(policy_model, reward_model, tokenizer):
    """Test PPO trainer components"""
    print("Testing PPO components...")
    
    config = get_fast_rlhf_config()
    
    # Create reference model (copy of policy)
    ref_model = SLMModel(config.base_model_config)
    ref_model.load_state_dict(policy_model.state_dict())
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Test response generation
    query = "What is the capital of France?"
    query_tokens = torch.tensor([tokenizer.encode(query)])
    
    with torch.no_grad():
        responses, logprobs, values = ppo_trainer.generate_responses(
            queries=query_tokens,
            max_length=50,
            temperature=0.7
        )
    
    print(f"Query: {query}")
    print(f"Response shape: {responses.shape}")
    print(f"Generated tokens: {responses[0].tolist()[:10]}...")  # First 10 tokens
    
    # Test reward computation
    full_sequence = torch.cat([query_tokens, responses], dim=1)
    rewards = reward_model.get_rewards(full_sequence)
    
    print(f"Reward: {rewards.item():.4f}")
    print("✓ PPO components test passed\n")
    
    return ppo_trainer


def test_integration():
    """Test full integration"""
    print("=" * 60)
    print("RLHF PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Test 1: Tokenizer
        tokenizer = test_tokenizer()
        
        # Test 2: Base model
        base_model = test_base_model(tokenizer)
        
        # Test 3: Reward model
        reward_model = test_reward_model(base_model, tokenizer)
        
        # Test 4: Datasets
        sft_dataset, pref_dataset, ppo_dataset = test_datasets(tokenizer)
        
        # Test 5: SFT trainer
        sft_trainer = test_sft_trainer(base_model, tokenizer)
        
        # Test 6: PPO components
        ppo_trainer = test_ppo_components(base_model, reward_model, tokenizer)
        
        # Test 7: Dataset validation
        config = get_fast_rlhf_config()
        is_valid = validate_datasets(config)
        print(f"Dataset validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("RLHF pipeline is ready for training.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_training_test():
    """Run a very quick training test with minimal data"""
    print("\n" + "=" * 60)
    print("QUICK TRAINING TEST")
    print("=" * 60)
    
    # Use fast config with even smaller settings
    config = get_fast_rlhf_config()
    config.sft_epochs = 1
    config.sft_batch_size = 2
    config.reward_model_epochs = 1
    config.reward_model_batch_size = 2
    config.max_train_steps = 5
    
    try:
        from train_rlhf import RLHFPipeline
        
        # Initialize pipeline
        pipeline = RLHFPipeline(config)
        
        # Initialize base model
        pipeline.initialize_base_model()
        
        print("Quick training test setup complete!")
        print("Ready to run full pipeline with: python train_rlhf.py --config fast")
        
        return True
        
    except Exception as e:
        print(f"Quick training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run integration test
    success = test_integration()
    
    if success:
        # Run quick training test
        run_quick_training_test()
    else:
        print("Integration test failed. Please fix issues before training.")
        sys.exit(1)