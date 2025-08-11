"""
Multi-Dataset Loader for GPT-OSS MoE Training
Supports combining multiple datasets including BookCorpus for improved training
"""

import logging
from datasets import load_dataset, concatenate_datasets, Dataset
from typing import List, Dict, Any, Optional, Union
import warnings

logger = logging.getLogger(__name__)

class MultiDatasetLoader:
    """Loader for combining multiple datasets for language model training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for dataset processing"""
        self.tokenizer = tokenizer
    
    def load_single_dataset(self, dataset_config: Dict[str, Any]) -> Dataset:
        """Load a single dataset based on configuration"""
        dataset_name = dataset_config.get('name')
        dataset_path = dataset_config.get('path')
        subset = dataset_config.get('subset')
        split = dataset_config.get('split', 'train')
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Handle different dataset sources
            if dataset_name == 'wikitext-2':
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            elif dataset_name == 'wikitext-103':
                dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
            elif dataset_name == 'bookcorpus':
                logger.error(f"BookCorpus is not available due to Hugging Face policy changes")
                raise ValueError("BookCorpus dataset is deprecated. Use 'wikitext-103' or other alternatives.")
            elif dataset_name == 'openwebtext':
                dataset = load_dataset('openwebtext', split=split)
            elif dataset_name == 'c4':
                dataset = load_dataset('c4', 'en', split=split, streaming=True)
            elif dataset_name == 'pile':
                dataset = load_dataset('monology/pile-uncopyrighted', split=split)
            elif dataset_path:
                # Load from local path
                dataset = load_dataset(dataset_path, split=split)
            elif subset:
                # Load with subset
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                # Load standard dataset
                dataset = load_dataset(dataset_name, split=split)
            
            # Apply dataset-specific preprocessing if configured
            dataset = self._preprocess_dataset(dataset, dataset_config)
            
            # Limit dataset size if specified
            max_samples = dataset_config.get('max_samples')
            if max_samples and len(dataset) > max_samples:
                logger.info(f"Limiting {dataset_name} to {max_samples} samples")
                dataset = dataset.select(range(max_samples))
            
            # Add dataset source information
            def add_source(example):
                example['dataset_source'] = dataset_name
                return example
            
            dataset = dataset.map(add_source)
            
            logger.info(f"Loaded {dataset_name}: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def _preprocess_dataset(self, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        """Apply dataset-specific preprocessing"""
        dataset_name = config.get('name')
        
        if dataset_name == 'bookcorpus':
            # BookCorpus specific preprocessing
            def process_bookcorpus(example):
                # Clean text and handle encoding issues
                text = example['text']
                if text:
                    # Remove excessive whitespace
                    text = ' '.join(text.split())
                    # Ensure minimum length for meaningful training
                    if len(text.split()) < 10:
                        text = ''
                example['text'] = text
                return example
            
            dataset = dataset.map(process_bookcorpus)
            # Filter out empty texts
            dataset = dataset.filter(lambda x: len(x['text']) > 0)
            
        elif dataset_name in ['wikitext-2', 'wikitext-103']:
            # WikiText specific preprocessing
            def process_wikitext(example):
                # Remove lines with only '=' characters (section headers)
                text = example['text']
                if text and not text.strip().startswith('='):
                    # Remove excessive whitespace
                    text = ' '.join(text.split())
                else:
                    text = ''
                example['text'] = text
                return example
            
            dataset = dataset.map(process_wikitext)
            dataset = dataset.filter(lambda x: len(x['text']) > 0)
        
        return dataset
    
    def load_combined_dataset(self, datasets_config: List[Dict[str, Any]], split: str = 'train') -> Dataset:
        """Load and combine multiple datasets"""
        logger.info(f"Loading {len(datasets_config)} datasets for {split} split")
        
        loaded_datasets = []
        total_samples = 0
        
        for dataset_config in datasets_config:
            # Update split for this dataset
            dataset_config = dict(dataset_config)  # Copy to avoid modifying original
            dataset_config['split'] = split
            
            try:
                dataset = self.load_single_dataset(dataset_config)
                loaded_datasets.append(dataset)
                total_samples += len(dataset)
                
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_config.get('name')}: {e}")
                continue
        
        if not loaded_datasets:
            raise ValueError("No datasets were successfully loaded")
        
        # Combine datasets
        if len(loaded_datasets) == 1:
            combined_dataset = loaded_datasets[0]
        else:
            logger.info("Combining datasets...")
            combined_dataset = concatenate_datasets(loaded_datasets)
        
        # Shuffle combined dataset
        combined_dataset = combined_dataset.shuffle(seed=42)
        
        logger.info(f"Combined dataset: {len(combined_dataset)} total samples from {len(loaded_datasets)} sources")
        
        return combined_dataset
    
    def create_dataloaders(self, train_datasets: List[Dict], eval_datasets: List[Dict], 
                          tokenizer, max_seq_length: int = 1024):
        """Create train and eval datasets with tokenization"""
        self.set_tokenizer(tokenizer)
        
        # Load datasets
        train_dataset = self.load_combined_dataset(train_datasets, split='train')
        
        # Handle validation datasets
        eval_dataset = None
        if eval_datasets:
            try:
                eval_dataset = self.load_combined_dataset(eval_datasets, split='validation')
            except:
                # If validation split doesn't exist, create from train
                logger.info("Creating validation split from training data")
                train_val_split = train_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = train_val_split['train']
                eval_dataset = train_val_split['test']
        
        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_attention_mask=True,
            )
        
        # Remove text column and tokenize
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing training data"
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['text'],
                desc="Tokenizing evaluation data"
            )
        
        # Group texts for language modeling - Fixed version
        def group_texts(examples):
            # Only operate on token fields
            valid_keys = [k for k in ('input_ids', 'attention_mask') if k in examples]
            if not valid_keys:
                return {"input_ids": [], "attention_mask": [], "labels": []}
            
            # Concatenate all token sequences
            concatenated = {k: sum(examples[k], []) for k in valid_keys}
            
            total_length = len(concatenated[valid_keys[0]])
            result = {k: [] for k in valid_keys}
            
            if total_length >= max_seq_length:
                # Create complete fixed-size blocks
                num_blocks = total_length // max_seq_length
                for k, t in concatenated.items():
                    for i in range(num_blocks):
                        start_idx = i * max_seq_length
                        end_idx = start_idx + max_seq_length
                        result[k].append(t[start_idx:end_idx])
            else:
                # Handle short sequences by padding to fixed size
                if total_length > 0:
                    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
                    if pad_token_id is None:
                        pad_token_id = tokenizer.eos_token_id
                    
                    for k, t in concatenated.items():
                        if k == 'input_ids':
                            padded = t + [pad_token_id] * (max_seq_length - len(t))
                        else:  # attention_mask
                            padded = t + [0] * (max_seq_length - len(t))
                        result[k].append(padded)
            
            # Create labels mirroring input_ids
            if result.get("input_ids"):
                result["labels"] = [seq.copy() for seq in result["input_ids"]]
            else:
                result["labels"] = []
            
            return result
        
        # Important: remove original columns (like dataset_source) because grouping changes row count
        train_dataset = train_dataset.map(
            group_texts,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Grouping training texts"
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                group_texts,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Grouping evaluation texts"
            )
        
        return train_dataset, eval_dataset

# Dataset configuration examples
PRESET_DATASETS = {
    'bookcorpus': {
        'name': 'bookcorpus',
        'max_samples': None,  # Use all data
        'weight': 1.0
    },
    'wikitext-2': {
        'name': 'wikitext-2',
        'max_samples': None,
        'weight': 1.0
    },
    'wikitext-103': {
        'name': 'wikitext-103', 
        'max_samples': 50000,  # Limit for memory
        'weight': 1.0
    },
    'openwebtext': {
        'name': 'openwebtext',
        'max_samples': 100000,  # Limit for memory
        'weight': 1.0
    }
}

def create_dataset_config(dataset_names: List[str]) -> List[Dict[str, Any]]:
    """Create dataset configuration from dataset names"""
    configs = []
    for name in dataset_names:
        if name in PRESET_DATASETS:
            configs.append(PRESET_DATASETS[name])
        else:
            # Default configuration for unknown datasets
            configs.append({
                'name': name,
                'max_samples': None,
                'weight': 1.0
            })
    return configs