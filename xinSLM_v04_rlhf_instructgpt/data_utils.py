"""
Data preprocessing utilities for RLHF datasets
Handles SFT, preference, and PPO prompt datasets
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import sys
import logging
from dataclasses import dataclass

# Add parent directory to path to import SLM components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xinSLM_v03_slm_llama_architecture'))
from tokenizer import SLMTokenizer


@dataclass
class PreferenceExample:
    """Single preference comparison example"""
    prompt: str
    chosen: str
    rejected: str
    score_chosen: Optional[float] = None
    score_rejected: Optional[float] = None


class PreferenceDataset(Dataset):
    """Dataset for reward model training with preference comparisons"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: SLMTokenizer,
        max_length: int = 512,
        instruction_template: str = "Human: ",
        response_template: str = "\n\nAssistant: "
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        self.response_template = response_template
        
        # Load preference data
        self.examples = self._load_preference_data(data_path)
        
    def _load_preference_data(self, data_path: str) -> List[PreferenceExample]:
        """Load preference comparison data"""
        
        if not os.path.exists(data_path):
            # Create sample preference data
            print(f"Preference data file {data_path} not found. Creating sample data...")
            sample_data = [
                {
                    "prompt": "What is the capital of France?",
                    "chosen": "The capital of France is Paris. It's a beautiful city known for its art, culture, and landmarks like the Eiffel Tower.",
                    "rejected": "Paris. It's in France."
                },
                {
                    "prompt": "Explain machine learning in simple terms.",
                    "chosen": "Machine learning is a branch of artificial intelligence where computers learn to make predictions or decisions by analyzing patterns in data, without being explicitly programmed for each specific task. It's like teaching a computer to recognize patterns the same way humans do.",
                    "rejected": "Machine learning is when computers learn stuff from data."
                },
                {
                    "prompt": "How do you make a good cup of coffee?",
                    "chosen": "To make a good cup of coffee: 1) Use fresh, quality coffee beans, 2) Grind them just before brewing, 3) Use the right water temperature (195-205°F), 4) Maintain proper coffee-to-water ratio (1:15-1:17), 5) Brew for the appropriate time based on your method. The key is consistency and using quality ingredients.",
                    "rejected": "Put coffee in hot water and drink it."
                },
                {
                    "prompt": "What are the benefits of exercise?",
                    "chosen": "Regular exercise provides numerous benefits including: improved cardiovascular health, stronger muscles and bones, better mental health and mood, increased energy levels, improved sleep quality, enhanced immune function, and reduced risk of chronic diseases like diabetes and heart disease. It also helps with weight management and cognitive function.",
                    "rejected": "Exercise is good for you. It makes you healthy."
                },
                {
                    "prompt": "Write a short poem about nature.",
                    "chosen": "In the forest deep and green,\nWhere sunlight filters through,\nThe gentle breeze whispers softly,\nOf morning fresh with dew.\n\nBirds sing their sweet melodies,\nFlowers bloom in vibrant hues,\nNature's symphony surrounds us,\nIn this peaceful, sacred muse.",
                    "rejected": "Trees are green\nSky is blue\nNature is nice\nI like it too"
                },
                {
                    "prompt": "How does photosynthesis work?",
                    "chosen": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. It occurs mainly in the leaves, where chlorophyll captures light energy. The process has two main stages: light-dependent reactions (which produce ATP and NADPH) and light-independent reactions (Calvin cycle) where CO2 is fixed into glucose. This process is essential for life on Earth as it produces oxygen and serves as the base of most food chains.",
                    "rejected": "Plants use sunlight to make food and oxygen."
                }
            ]
            
            # Create directory and save sample data
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w') as f:
                for item in sample_data:
                    f.write(json.dumps(item) + '\n')
        
        # Load data
        examples = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    examples.append(PreferenceExample(
                        prompt=data['prompt'],
                        chosen=data['chosen'],
                        rejected=data['rejected'],
                        score_chosen=data.get('score_chosen'),
                        score_rejected=data.get('score_rejected')
                    ))
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single preference comparison example"""
        example = self.examples[idx]
        
        # Format chosen and rejected responses
        chosen_text = (
            self.instruction_template + example.prompt +
            self.response_template + example.chosen
        )
        rejected_text = (
            self.instruction_template + example.prompt +
            self.response_template + example.rejected
        )
        
        # Tokenize
        chosen_tokens = self.tokenizer.encode(chosen_text)
        rejected_tokens = self.tokenizer.encode(rejected_text)
        
        # Truncate if too long
        if len(chosen_tokens) > self.max_length:
            chosen_tokens = chosen_tokens[:self.max_length]
        if len(rejected_tokens) > self.max_length:
            rejected_tokens = rejected_tokens[:self.max_length]
        
        # Convert to tensors
        chosen_input_ids = torch.tensor(chosen_tokens, dtype=torch.long)
        rejected_input_ids = torch.tensor(rejected_tokens, dtype=torch.long)
        
        # Create attention masks
        chosen_attention_mask = torch.ones_like(chosen_input_ids)
        rejected_attention_mask = torch.ones_like(rejected_input_ids)
        
        return {
            'chosen_input_ids': chosen_input_ids,
            'rejected_input_ids': rejected_input_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_attention_mask': rejected_attention_mask
        }
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching preference examples"""
        
        # Separate chosen and rejected
        chosen_input_ids = [item['chosen_input_ids'] for item in batch]
        rejected_input_ids = [item['rejected_input_ids'] for item in batch]
        chosen_attention_mask = [item['chosen_attention_mask'] for item in batch]
        rejected_attention_mask = [item['rejected_attention_mask'] for item in batch]
        
        # Pad sequences
        def pad_sequences(sequences, pad_value=0):
            max_len = max(seq.size(0) for seq in sequences)
            padded = []
            for seq in sequences:
                pad_len = max_len - seq.size(0)
                if pad_len > 0:
                    padded_seq = F.pad(seq, (0, pad_len), value=pad_value)
                else:
                    padded_seq = seq
                padded.append(padded_seq)
            return torch.stack(padded)
        
        return {
            'chosen_input_ids': pad_sequences(chosen_input_ids, pad_value=0),
            'rejected_input_ids': pad_sequences(rejected_input_ids, pad_value=0),
            'chosen_attention_mask': pad_sequences(chosen_attention_mask, pad_value=0),
            'rejected_attention_mask': pad_sequences(rejected_attention_mask, pad_value=0)
        }


class PPOPromptDataset(Dataset):
    """Dataset of prompts for PPO training"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: SLMTokenizer,
        max_length: int = 256,
        instruction_template: str = "Human: ",
        response_template: str = "\n\nAssistant: "
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        self.response_template = response_template
        
        # Load prompts
        self.prompts = self._load_prompts(data_path)
        
    def _load_prompts(self, data_path: str) -> List[str]:
        """Load prompts for PPO training"""
        
        if not os.path.exists(data_path):
            # Create sample prompts
            print(f"PPO prompts file {data_path} not found. Creating sample prompts...")
            sample_prompts = [
                "Explain the importance of renewable energy.",
                "What are the key principles of good communication?",
                "Describe how to maintain a healthy lifestyle.",
                "What is the scientific method?",
                "How do you solve conflicts peacefully?",
                "Explain the water cycle in detail.",
                "What are the benefits of reading books?",
                "How does the internet work?",
                "What makes a good leader?",
                "Describe the process of photosynthesis.",
                "What are the main causes of climate change?",
                "How do you build confidence?",
                "Explain the importance of biodiversity.",
                "What are effective study techniques?",
                "How do you manage stress effectively?",
                "What is artificial intelligence?",
                "Describe the structure of an atom.",
                "How do you write a compelling story?",
                "What are the benefits of teamwork?",
                "Explain how vaccines work.",
                "What makes cities sustainable?",
                "How do you develop critical thinking skills?",
                "What is the role of art in society?",
                "Describe the water purification process.",
                "How do you maintain work-life balance?",
                "What are the principles of good design?",
                "Explain the concept of democracy.",
                "How do you learn a new language effectively?",
                "What are the effects of deforestation?",
                "Describe how memory works in the brain."
            ]
            
            # Save sample prompts
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w') as f:
                for prompt in sample_prompts:
                    f.write(json.dumps({"prompt": prompt}) + '\n')
        
        # Load prompts
        prompts = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    prompts.append(data['prompt'])
        
        return prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single prompt for generation"""
        prompt = self.prompts[idx]
        
        # Format prompt
        formatted_prompt = self.instruction_template + prompt + self.response_template
        
        # Tokenize
        tokens = self.tokenizer.encode(formatted_prompt)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt_text': prompt
        }


def create_preference_dataset(
    data_path: str,
    tokenizer: SLMTokenizer,
    config,
    validation_split: float = 0.1
) -> PreferenceDataset:
    """Create preference dataset for reward model training"""
    
    dataset = PreferenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config.reward_model_max_length,
        instruction_template=config.instruction_template,
        response_template=config.response_template
    )
    
    return dataset


def create_ppo_dataset(
    data_path: str,
    tokenizer: SLMTokenizer,
    config
) -> PPOPromptDataset:
    """Create PPO prompt dataset"""
    
    dataset = PPOPromptDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config.ppo_max_length,
        instruction_template=config.instruction_template,
        response_template=config.response_template
    )
    
    return dataset


class HumanFeedbackSimulator:
    """Simulate human feedback for generating preference pairs"""
    
    def __init__(self, tokenizer: SLMTokenizer):
        self.tokenizer = tokenizer
        
        # Simple heuristics for response quality
        self.quality_indicators = {
            'length_bonus': 0.1,  # Longer responses get small bonus
            'detail_bonus': 0.2,   # Responses with more detail
            'politeness_bonus': 0.1,  # Polite responses
            'accuracy_bonus': 0.3,    # Factually accurate (simulated)
        }
    
    def score_response(self, prompt: str, response: str) -> float:
        """Score a response based on simple heuristics"""
        score = 0.5  # Base score
        
        # Length bonus (moderate length is good)
        response_length = len(response.split())
        if 10 <= response_length <= 100:
            score += self.quality_indicators['length_bonus']
        
        # Detail indicators
        detail_words = ['because', 'therefore', 'however', 'specifically', 'for example', 'such as']
        detail_count = sum(1 for word in detail_words if word in response.lower())
        score += detail_count * self.quality_indicators['detail_bonus'] / 10
        
        # Politeness indicators
        polite_words = ['please', 'thank you', 'you\'re welcome', 'certainly', 'of course']
        politeness_count = sum(1 for word in polite_words if word in response.lower())
        score += politeness_count * self.quality_indicators['politeness_bonus'] / 5
        
        # Penalize very short or very long responses
        if response_length < 5:
            score -= 0.3
        elif response_length > 200:
            score -= 0.2
        
        # Penalize repetitive responses
        words = response.lower().split()
        if len(set(words)) < len(words) * 0.5:  # Less than 50% unique words
            score -= 0.2
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def create_preference_pair(
        self,
        prompt: str,
        response1: str,
        response2: str
    ) -> Tuple[str, str, float, float]:
        """Create a preference pair with scores"""
        
        score1 = self.score_response(prompt, response1)
        score2 = self.score_response(prompt, response2)
        
        # Return in chosen/rejected format
        if score1 >= score2:
            return response1, response2, score1, score2
        else:
            return response2, response1, score2, score1


def augment_preference_data(
    base_data_path: str,
    model_responses_path: str,
    output_path: str,
    tokenizer: SLMTokenizer
):
    """Augment preference data using model-generated responses"""
    
    simulator = HumanFeedbackSimulator(tokenizer)
    
    # Load base preference data
    base_examples = []
    if os.path.exists(base_data_path):
        with open(base_data_path, 'r') as f:
            for line in f:
                if line.strip():
                    base_examples.append(json.loads(line.strip()))
    
    # Load model responses if available
    model_responses = {}
    if os.path.exists(model_responses_path):
        with open(model_responses_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    prompt = data['prompt']
                    if prompt not in model_responses:
                        model_responses[prompt] = []
                    model_responses[prompt].append(data['response'])
    
    # Create augmented dataset
    augmented_examples = []
    
    # Add base examples
    for example in base_examples:
        augmented_examples.append(example)
    
    # Create new preference pairs from model responses
    for prompt, responses in model_responses.items():
        if len(responses) >= 2:
            # Create pairs from different responses
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    chosen, rejected, score_chosen, score_rejected = simulator.create_preference_pair(
                        prompt, responses[i], responses[j]
                    )
                    
                    augmented_examples.append({
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': rejected,
                        'score_chosen': score_chosen,
                        'score_rejected': score_rejected
                    })
    
    # Save augmented dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for example in augmented_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created augmented preference dataset with {len(augmented_examples)} examples")


def validate_datasets(config):
    """Validate that all required datasets exist and are well-formed"""
    
    logger = logging.getLogger(__name__)
    
    # Check SFT data
    if not os.path.exists(config.sft_data_path):
        logger.warning(f"SFT data not found at {config.sft_data_path}")
        return False
    
    # Check reward data
    if not os.path.exists(config.reward_data_path):
        logger.warning(f"Reward data not found at {config.reward_data_path}")
        return False
    
    # Check PPO prompts
    if not os.path.exists(config.ppo_prompts_path):
        logger.warning(f"PPO prompts not found at {config.ppo_prompts_path}")
        return False
    
    # Validate data format
    try:
        # Test SFT data
        with open(config.sft_data_path, 'r') as f:
            sft_sample = json.loads(f.readline().strip())
            assert 'instruction' in sft_sample and 'response' in sft_sample
        
        # Test reward data
        with open(config.reward_data_path, 'r') as f:
            reward_sample = json.loads(f.readline().strip())
            assert 'prompt' in reward_sample and 'chosen' in reward_sample and 'rejected' in reward_sample
        
        # Test PPO prompts
        with open(config.ppo_prompts_path, 'r') as f:
            ppo_sample = json.loads(f.readline().strip())
            assert 'prompt' in ppo_sample
        
        logger.info("All datasets validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test data utilities
    from rlhf_config import get_fast_rlhf_config
    
    config = get_fast_rlhf_config()
    
    # Initialize tokenizer
    tokenizer = SLMTokenizer()
    
    # Test preference dataset
    print("Testing preference dataset...")
    pref_dataset = create_preference_dataset(
        data_path=config.reward_data_path,
        tokenizer=tokenizer,
        config=config
    )
    print(f"Preference dataset size: {len(pref_dataset)}")
    
    # Test PPO dataset
    print("Testing PPO dataset...")
    ppo_dataset = create_ppo_dataset(
        data_path=config.ppo_prompts_path,
        tokenizer=tokenizer,
        config=config
    )
    print(f"PPO dataset size: {len(ppo_dataset)}")
    
    # Validate datasets
    print("Validating datasets...")
    is_valid = validate_datasets(config)
    print(f"Datasets valid: {is_valid}")
    
    print("Data utilities test completed!")