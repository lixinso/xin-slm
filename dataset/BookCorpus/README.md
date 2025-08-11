# BookCorpus Dataset

## Overview
BookCorpus is a large collection of free novel books written by unpublished authors, originally used to train foundational language models including OpenAI's GPT and Google's BERT.

## Dataset Statistics
- **Total Books**: 11,038 books (approximately 7,185 unique due to duplicates)
- **Content Volume**: ~74M sentences, ~1B words
- **File Size**: ~1GB of plain text
- **Genres**: 16 sub-genres including Romance, Historical, Adventure, Fantasy, etc.
- **Source**: Originally collected from Smashwords.com (free books platform)

## Historical Significance
This dataset has been instrumental in training several influential large language models:
- **OpenAI's Original GPT Model** - Primary training corpus
- **Google's BERT** - Key training component  
- **Amazon's Bort** - Training data
- **Various Early LLMs** - Foundation dataset for language model research

## Download Sources

### Primary/Recommended
1. **Hugging Face** (Most Reliable)
   - URL: https://huggingface.co/datasets/bookcorpus/bookcorpus
   - Well-maintained and documented
   - Provides preview and download access

2. **Papers with Code**
   - URL: https://paperswithcode.com/dataset/bookcorpus
   - Academic documentation and research context

### Alternative Sources
3. **GitHub Community Links**
   - URL: https://github.com/soskek/bookcorpus/issues/27
   - Contains 18k plain text files (as of 2020)
   - Unofficial but widely referenced

## Training Benefits for Language Models
- **Long-form Context**: Books provide extended, coherent narratives for learning long-range dependencies
- **High-quality Text**: Published books typically have better grammar and structure than web text
- **Narrative Coherence**: Teaches models story progression and logical flow
- **Literary Diversity**: Exposure to various writing styles, genres, and vocabulary
- **Sequential Structure**: Natural chapter/paragraph organization for hierarchical learning

## File Format
- **Format**: Plain text files (.txt)
- **Encoding**: UTF-8
- **Structure**: One book per file, with natural paragraph breaks
- **Preprocessing**: Minimal - maintains original book structure

## Usage Recommendations
- **Best For**: Training language models on long-form text understanding
- **Ideal Applications**: 
  - Story generation and continuation
  - Long-context language modeling
  - Literary style transfer
  - Narrative coherence training
  - Creative writing assistance models

## Important Legal and Ethical Considerations

### ⚠️ Copyright and Consent Issues
- **Author Consent**: Books were collected **without authors' explicit permission**
- **Copyright Status**: Many books contain redistribution restrictions
- **Ethical Concerns**: Authors did not consent to AI training usage
- **Legal Status**: Use for research may fall under fair use, but commercial use is questionable

### Responsible Usage Guidelines
- Consider ethical implications before using this dataset
- Acknowledge the lack of author consent in research publications
- Explore alternative datasets with proper licensing when possible
- Be transparent about dataset limitations in model documentation

## Alternative Datasets (Ethically Sourced)
For training without copyright concerns, consider:
- **Project Gutenberg** - Public domain books
- **OpenBookCorpus** - Ethically sourced alternative
- **CommonCrawl Books** - Web-scraped book content with filtering

## Dataset Quality Notes
- **Duplicates Present**: ~3,853 duplicate books in the collection
- **Genre Distribution**: Uneven distribution across sub-genres
- **Quality Variation**: Mix of professional and amateur writing quality
- **Language**: Primarily English language content
- **Time Period**: Books collected circa 2015-2016

## Technical Specifications
- **Total Files**: ~18,000 plain text files
- **Average Book Length**: ~67KB per book
- **Character Encoding**: UTF-8
- **Line Endings**: Unix-style (LF)
- **Compression**: Available in compressed formats (.tar, .zip)

## Research Applications
This dataset has been used in research for:
- Language model pre-training
- Long-context understanding evaluation
- Narrative generation benchmarks
- Literary style analysis
- Reading comprehension tasks


ERROR - Failed to load dataset bookcorpus: BookCorpus dataset is deprecated. Use 'wikitext-103' or other alternatives.