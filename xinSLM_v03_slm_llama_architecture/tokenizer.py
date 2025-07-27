"""
Tokenizer implementation for SLM
Based on Llama 3.2's tiktoken-style tokenizer with extended vocabulary
"""
import torch
import json
import regex as re
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path


class SLMTokenizer:
    """
    Byte-level BPE tokenizer similar to Llama 3.2's tiktoken-based tokenizer
    Features a large vocabulary (~128K tokens) for multilingual and code support
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        vocab_size: int = 128256,
        pad_token: str = "<pad>",
        eos_token: str = "<|endoftext|>",
        bos_token: str = "<|begin_of_text|>",
        unk_token: str = "<unk>",
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        
        # Special tokens
        self.special_tokens = {
            pad_token: 0,
            unk_token: 1,
            bos_token: 2,
            eos_token: 3,
        }
        
        # Initialize with basic vocabulary if files not provided
        if vocab_file is None or merges_file is None:
            self._build_default_vocab()
        else:
            self._load_vocab(vocab_file, merges_file)
        
        # Regex pattern for tokenization (similar to GPT-4)
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    def _build_default_vocab(self):
        """Build a default vocabulary for demonstration"""
        print("Building default vocabulary...")
        
        # Start with special tokens
        self.encoder = self.special_tokens.copy()
        
        # Add byte-level tokens (0-255)
        for i in range(256):
            if i not in self.encoder.values():
                self.encoder[chr(i)] = len(self.encoder)
        
        # Add common subwords and multilingual tokens
        common_tokens = [
            # English common words
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they",
            "I", "at", "be", "this", "have", "from", "or", "one", "had",
            "by", "word", "but", "not", "what", "all", "were", "we", "when",
            
            # Code tokens
            "def", "class", "import", "from", "if", "else", "elif", "for",
            "while", "try", "except", "return", "print", "True", "False",
            "None", "self", "in", "and", "or", "not", "lambda", "with",
            "as", "pass", "break", "continue", "yield", "async", "await",
            
            # Programming symbols
            "==", "!=", "<=", ">=", "->", "=>", "&&", "||", "++", "--",
            "+=", "-=", "*=", "/=", "%=", "**", "//", "<<", ">>",
            
            # Common punctuation combinations
            ". ", ", ", "! ", "? ", ": ", "; ", "\" ", "' ", "( ", ") ",
            "[ ", "] ", "{ ", "} ", "< ", "> ", "/ ", "\\ ", "| ",
            
            # Multilingual common words (basic set)
            # Spanish
            "el", "la", "de", "que", "y", "en", "un", "es", "se", "no",
            "te", "lo", "le", "da", "su", "por", "son", "con", "para",
            
            # French  
            "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir",
            "que", "pour", "dans", "ce", "son", "une", "sur", "avec", "ne",
            
            # German
            "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
            "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine",
            
            # Common prefixes/suffixes
            "un", "re", "in", "dis", "en", "non", "over", "mis", "sub", "pre",
            "ing", "ed", "er", "est", "ly", "tion", "ness", "ment", "ful",
        ]
        
        # Add common tokens
        for token in common_tokens:
            if token not in self.encoder:
                self.encoder[token] = len(self.encoder)
        
        # Fill remaining vocabulary with placeholder tokens
        while len(self.encoder) < self.vocab_size:
            placeholder = f"<extra_token_{len(self.encoder)}>"
            self.encoder[placeholder] = len(self.encoder)
        
        # Create decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Initialize empty BPE merges for default vocab
        self.bpe_merges = {}
        self.bpe_ranks = {}
    
    def _load_vocab(self, vocab_file: str, merges_file: str):
        """Load vocabulary and BPE merges from files"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Load BPE merges
        with open(merges_file, 'r', encoding='utf-8') as f:
            bpe_data = f.read().split('\n')
        
        bpe_merges = [tuple(merge.split()) for merge in bpe_data[1:-1]]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.bpe_merges = {merge: i for i, merge in enumerate(bpe_merges)}
    
    def get_pairs(self, word: Tuple[str, ...]) -> set:
        """Get all symbol pairs in a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def bpe(self, token: str) -> str:
        """Apply Byte-Pair Encoding to a token"""
        if not self.bpe_ranks:
            return token
        
        word = tuple(token)
        pairs = self.get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        
        return ' '.join(word)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        if add_special_tokens:
            text = self.bos_token + text
        
        # Split text using regex pattern
        tokens = re.findall(self.pat, text)
        
        token_ids = []
        for token in tokens:
            # Apply BPE if available
            if self.bpe_ranks:
                token = self.bpe(token)
                subtokens = token.split(' ')
            else:
                subtokens = [token]
            
            for subtoken in subtokens:
                if subtoken in self.encoder:
                    token_ids.append(self.encoder[subtoken])
                else:
                    # Fall back to byte-level encoding
                    for byte in subtoken.encode('utf-8'):
                        if chr(byte) in self.encoder:
                            token_ids.append(self.encoder[chr(byte)])
                        else:
                            token_ids.append(self.encoder[self.unk_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                token = self.decoder[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.unk_token)
        
        return ''.join(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to string tokens"""
        token_ids = self.encode(text, add_special_tokens=False)
        return [self.decoder[token_id] for token_id in token_ids]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert string tokens to IDs"""
        return [self.encoder.get(token, self.encoder[self.unk_token]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to string tokens"""
        return [self.decoder.get(id, self.unk_token) for id in ids]
    
    def batch_encode(self, texts: List[str], padding: bool = True, max_length: Optional[int] = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """Batch encode multiple texts"""
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        
        if max_length is None:
            max_length = max(len(seq) for seq in encoded) if encoded else 0
        
        # Truncate sequences that are too long
        encoded = [seq[:max_length] for seq in encoded]
        
        if padding:
            # Pad sequences to max_length
            padded = []
            attention_masks = []
            
            for seq in encoded:
                pad_length = max_length - len(seq)
                padded_seq = seq + [self.encoder[self.pad_token]] * pad_length
                attention_mask = [1] * len(seq) + [0] * pad_length
                
                padded.append(padded_seq)
                attention_masks.append(attention_mask)
            
            return {
                'input_ids': torch.tensor(padded, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        else:
            return {'input_ids': [torch.tensor(seq, dtype=torch.long) for seq in encoded]}
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        vocab_file = save_path / "vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)
        
        # Save merges if available
        if self.bpe_ranks:
            merges_file = save_path / "merges.txt"
            with open(merges_file, 'w', encoding='utf-8') as f:
                f.write("#version: 0.2\n")
                for merge in sorted(self.bpe_ranks.keys(), key=lambda x: self.bpe_ranks[x]):
                    f.write(f"{merge[0]} {merge[1]}\n")
        
        # Save tokenizer config
        config = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "unk_token": self.unk_token,
        }
        config_file = save_path / "tokenizer_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load tokenizer from directory"""
        model_path = Path(model_path)
        
        vocab_file = model_path / "vocab.json"
        merges_file = model_path / "merges.txt"
        config_file = model_path / "tokenizer_config.json"
        
        # Load config
        config = {}
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # Create tokenizer
        if vocab_file.exists() and merges_file.exists():
            return cls(vocab_file=str(vocab_file), merges_file=str(merges_file), **config)
        else:
            return cls(**config)
    
    def __len__(self):
        return len(self.encoder)
    
    @property
    def pad_token_id(self):
        return self.encoder[self.pad_token]
    
    @property
    def eos_token_id(self):
        return self.encoder[self.eos_token]
    
    @property
    def bos_token_id(self):
        return self.encoder[self.bos_token]
    
    @property
    def unk_token_id(self):
        return self.encoder[self.unk_token]


def create_slm_tokenizer(vocab_size: int = 128256) -> SLMTokenizer:
    """Create an SLM tokenizer with specified vocabulary size"""
    return SLMTokenizer(vocab_size=vocab_size)