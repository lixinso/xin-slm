# Datasets for Small Language Model Training

This directory contains datasets and documentation for training the SLM implementations in this repository.

## Currently Available

### WikiText-2
- **Location**: `dataset/WikiText-2/`
- **Size**: ~4MB
- **Description**: Collection of good and featured Wikipedia articles
- **Use Case**: Quick testing and initial experiments
- **Languages**: English
- **Format**: Plain text

## Recommended Datasets for SLM Training

### =€ Quick Start (Testing)

#### 1. WikiText-103
- **Size**: ~500MB
- **Download**: `load_dataset('wikitext', 'wikitext-103-raw-v1')`
- **Description**: Larger version of WikiText with 100x more data
- **Best for**: Initial SLM training experiments
- **Tokens**: ~100M tokens

#### 2. OpenWebText
- **Size**: ~38GB
- **Download**: `load_dataset('openwebtext')`
- **Description**: Open-source recreation of GPT-2's WebText dataset
- **Best for**: General language modeling
- **Tokens**: ~8B tokens

### =Ú Text-Only Datasets

#### 3. The Pile (Subset)
- **Size**: Variable (recommend 1-10% subset)
- **Download**: `load_dataset('EleutherAI/pile', split='train[:1%]')`
- **Description**: 800GB of diverse, high-quality text
- **Best for**: Diverse domain training
- **Domains**: Books, academic papers, code, web text, etc.

#### 4. C4 (Colossal Clean Crawled Corpus)
- **Size**: ~750GB (use subset)
- **Download**: `load_dataset('c4', 'en', split='train[:1%]')`
- **Description**: Cleaned Common Crawl data used by T5
- **Best for**: Web-scale language modeling

### < Multilingual Datasets

#### 5. mC4 (Multilingual C4)
- **Size**: Variable by language
- **Download**: `load_dataset('mc4', 'en')` (replace 'en' with language code)
- **Description**: Multilingual version of C4
- **Languages**: 100+ languages
- **Best for**: Multilingual SLM training

#### 6. OSCAR
- **Size**: Variable by language
- **Download**: `load_dataset('oscar', 'unshuffled_deduplicated_en')`
- **Description**: Multilingual corpus from Common Crawl
- **Languages**: 160+ languages
- **Best for**: High-quality multilingual text

### =» Code + Text Datasets

#### 7. The Stack
- **Size**: ~3TB (use subset)
- **Download**: `load_dataset('bigcode/the-stack', split='train[:0.1%]')`
- **Description**: Large collection of source code
- **Languages**: 30+ programming languages
- **Best for**: Code-capable SLM training

#### 8. CodeParrot
- **Size**: ~50GB
- **Download**: `load_dataset('codeparrot/codeparrot-clean')`
- **Description**: Python code dataset
- **Best for**: Python-focused code training

#### 9. RedPajama
- **Size**: ~1.2TB (use subset)
- **Download**: `load_dataset('togethercomputer/RedPajama-Data-1T-Sample')`
- **Description**: Open reproduction of LLaMA training data
- **Domains**: CommonCrawl, Wikipedia, books, academic papers, code
- **Best for**: Reproducing large model training

### =Ö Books and Literature

#### 10. BookCorpus
- **Size**: ~1GB
- **Download**: Available through various sources
- **Description**: Collection of over 11,000 books
- **Best for**: Long-form text understanding

#### 11. Project Gutenberg
- **Size**: ~3GB
- **Download**: `load_dataset('sedthh/gutenberg_english')`
- **Description**: Public domain books and literature
- **Best for**: Literary and classical text training

## Dataset Usage Examples

### Basic Setup

```python
from datasets import load_dataset
from v03_slm_llama_architecture.train import TextDataset
from v03_slm_llama_architecture.tokenizer import SLMTokenizer

# Initialize tokenizer
tokenizer = SLMTokenizer(vocab_size=128256)
```

### Example 1: WikiText-103 (Recommended Start)

```python
# Load dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

# Filter out empty/short texts
train_texts = [text for text in dataset['train']['text'] 
               if len(text.strip()) > 100]
eval_texts = [text for text in dataset['validation']['text'] 
              if len(text.strip()) > 100]

# Create datasets
train_dataset = TextDataset(train_texts, tokenizer, max_length=2048)
eval_dataset = TextDataset(eval_texts, tokenizer, max_length=2048)

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")
```

### Example 2: The Pile (Subset)

```python
# Load 1% of The Pile for manageable training
dataset = load_dataset('EleutherAI/pile', split='train[:1%]')

# Extract texts
train_texts = dataset['text'][:80000]  # 80% for training
eval_texts = dataset['text'][80000:100000]  # 20% for evaluation

# Create datasets
train_dataset = TextDataset(train_texts, tokenizer, max_length=2048)
eval_dataset = TextDataset(eval_texts, tokenizer, max_length=2048)
```

### Example 3: Code + Text Mix

```python
# Load code dataset
code_dataset = load_dataset('bigcode/the-stack', 
                           data_files='data/python/*.jsonl.gz',
                           split='train[:1000]')

# Load text dataset  
text_dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

# Combine datasets
code_texts = [item['content'] for item in code_dataset 
              if len(item['content'].strip()) > 200]
wiki_texts = [text for text in text_dataset['train']['text'][:5000]
              if len(text.strip()) > 100]

# Mix code and text
mixed_texts = code_texts + wiki_texts
random.shuffle(mixed_texts)

# Create training dataset
train_dataset = TextDataset(mixed_texts, tokenizer, max_length=2048)
```

### Example 4: Multilingual Training

```python
# Load multiple languages
languages = ['en', 'es', 'fr', 'de']
all_texts = []

for lang in languages:
    dataset = load_dataset('mc4', lang, split='train[:0.1%]')
    texts = [text for text in dataset['text'] 
             if len(text.strip()) > 100][:10000]  # 10k per language
    all_texts.extend(texts)

# Shuffle to mix languages
random.shuffle(all_texts)

# Create multilingual dataset
train_dataset = TextDataset(all_texts, tokenizer, max_length=2048)
```

## Memory and Compute Recommendations

### Mac Mini M4 Guidelines

| Dataset Size | RAM Usage | Training Time | Recommendation |
|--------------|-----------|---------------|----------------|
| WikiText-2 | ~2GB | 30 minutes | Testing only |
| WikiText-103 | ~4GB | 2-4 hours | Good start |
| OpenWebText (1%) | ~8GB | 6-12 hours | Serious training |
| The Pile (1%) | ~16GB | 12-24 hours | Advanced training |

### Training Configuration by Dataset

#### Small Datasets (WikiText-103)
```python
config = SLMConfig(
    batch_size=8,
    sequence_length=1024,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    num_epochs=10
)
```

#### Medium Datasets (OpenWebText subset)
```python
config = SLMConfig(
    batch_size=4,
    sequence_length=2048,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    num_epochs=3
)
```

#### Large Datasets (The Pile subset)
```python
config = SLMConfig(
    batch_size=2,
    sequence_length=4096,
    gradient_accumulation_steps=16,
    learning_rate=3e-4,
    num_epochs=1
)
```

## Download and Preparation Scripts

### Automatic Dataset Downloader

```bash
cd dataset/
python -c "
from datasets import load_dataset
import os

# Download WikiText-103
print('Downloading WikiText-103...')
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
dataset.save_to_disk('WikiText-103')

# Download OpenWebText subset
print('Downloading OpenWebText subset...')
dataset = load_dataset('openwebtext', split='train[:1%]')
dataset.save_to_disk('OpenWebText-1pct')

print('Datasets downloaded to dataset/ directory')
"
```

### Custom Dataset Preparation

```python
# For custom text files
def prepare_custom_dataset(file_paths, output_dir):
    all_texts = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split into chunks (e.g., by paragraphs)
            chunks = text.split('\n\n')
            chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
            all_texts.extend(chunks)
    
    # Save as dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({'text': all_texts})
    dataset.save_to_disk(output_dir)
    
    return dataset

# Usage
# dataset = prepare_custom_dataset(['my_text1.txt', 'my_text2.txt'], 'custom_dataset')
```

## Quality and Filtering

### Text Quality Filters

```python
def filter_high_quality_text(texts):
    """Filter texts for quality based on various criteria"""
    filtered = []
    
    for text in texts:
        # Length filter
        if len(text.split()) < 50:
            continue
            
        # Language detection (optional)
        # if detect_language(text) != 'en':
        #     continue
            
        # Remove texts with too many special characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.6:
            continue
            
        # Remove very repetitive texts
        words = text.split()
        unique_words = len(set(words))
        if unique_words / len(words) < 0.5:
            continue
            
        filtered.append(text)
    
    return filtered
```

## Storage Structure

```
dataset/
   README.md                 # This file
   WikiText-2/              # Small test dataset (included)
   WikiText-103/            # Downloaded WikiText-103
   OpenWebText-1pct/        # 1% subset of OpenWebText
   ThePile-subset/          # Custom subset of The Pile
   custom/                  # Your custom datasets
   scripts/                 # Dataset preparation scripts
       download_datasets.py
       prepare_custom.py
       quality_filter.py
```

## Best Practices

1. **Start Small**: Begin with WikiText-103 to test your training pipeline
2. **Quality over Quantity**: Better to train on 100MB of high-quality text than 1GB of poor quality
3. **Monitor Memory**: Use `htop` to monitor RAM usage during training
4. **Save Checkpoints**: Large datasets require checkpointing every few hours
5. **Validate Early**: Run evaluation frequently to catch overfitting
6. **Mix Domains**: Combine different types of text for better generalization

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or sequence_length
2. **Slow Download**: Use HuggingFace cache directory: `export HF_HOME=/path/to/cache`
3. **Dataset Too Large**: Use dataset streaming: `load_dataset(..., streaming=True)`
4. **Tokenization Errors**: Check for encoding issues in custom datasets

### Performance Tips

1. **Use SSD Storage**: Store datasets on fast storage for better I/O
2. **Preprocess Once**: Save tokenized datasets to avoid repeated preprocessing
3. **Parallel Loading**: Use multiple workers in DataLoader: `num_workers=4`
4. **Memory Mapping**: Use memory-mapped datasets for large files

## Contributing

To add a new dataset:

1. Add description and usage example to this README
2. Test with the SLM training pipeline
3. Document memory requirements and training time
4. Add download/preparation script if needed

## License Notes

Always check dataset licenses before use:
- **WikiText**: Creative Commons
- **OpenWebText**: Various (check individual sources)  
- **The Pile**: Various (check components)
- **Code datasets**: Various open source licenses

Make sure your use case complies with the respective licenses.