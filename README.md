# Novel Style Training Experiment

A Python application that applies GPT-OSS RL training principles to study dataset size implications by training separate models on individual novels. Each model learns the unique style of a single classic book, then generates new stories in that style.

## Overview

Based on the GPT-OSS reinforcement learning infrastructure from the original code, this project creates:

- **Single Novel Training**: Each model trains on one book to test data efficiency
- **Style Learning**: Models learn specific author styles and writing patterns
- **Novel Generation**: Interactive chat interface for creating new stories
- **Comparative Analysis**: Study how dataset size affects model performance

## Features

- ✅ **Individual Novel Processing**: Train separate models on different classic books
- ✅ **Style Analysis**: Extract and match writing style characteristics
- ✅ **RL Training**: Uses GRPO with novel-specific reward functions
- ✅ **Interactive Chat**: Generate stories with trained models
- ✅ **Experiment Tracking**: Compare results across different novels

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install unsloth torch transformers datasets

# Clone or download this project
# Ensure you have CUDA-capable GPU for training
```

### 2. Process Existing Novels

The novels are already available in the `novels__uncleaned` folder and need to be cleaned and prepared:

```bash
# Interactive setup (recommended)
python setup_novels_fixed.py

# OR process directly:
python simple_novel_processor.py --batch 5    # Quick start
python simple_novel_processor.py --batch 20   # Larger batch
python simple_novel_processor.py --list       # List processed
```

The system automatically cleans and prepares the existing novels:
- Text cleaning and normalization
- Remove headers, footers, and formatting artifacts
- Genre and style detection
- Validation and quality checks
- Batch processing to manage token limits efficiently

### 3. Train Models

```bash
# Run experiments on all novels
python single_novel_trainer.py

# This will:
# - Analyze each novel's style
# - Train separate models using RL
# - Save results and trained models
# - Generate comparative analysis
```

### 4. Chat with Trained Models

```bash
# Interactive chat interface
python novel_chat.py

# Or specify a particular experiment
python novel_chat.py --experiment alice_in_wonderland_experiment
```

## Project Structure (Clean)

```
├── single_novel_trainer.py    # Main training script
├── novel_chat.py             # Interactive chat interface
├── simple_novel_processor.py  # Novel processor
├── setup_novels_fixed.py     # Interactive setup
├── novels__uncleaned/        # Raw novel files (provided)
├── novels/                   # Organized, cleaned novels
│   ├── alice_in_wonderland/  # Each novel in own directory
│   │   ├── alice_in_wonderland.txt
│   │   ├── analysis.json
│   │   └── README.md
│   ├── alchemist/
│   └── ...
├── experiments/              # Training results
├── archive/                  # Reference files
├── simple_processing_status.json
└── README.md
```

## How It Works

### 1. Novel Processing
- **Automated Download**: Fetches classic novels from Project Gutenberg
- **Text Cleaning**: Removes headers, footers, and formatting artifacts
- **Style Analysis**: Extracts writing characteristics (sentence length, vocabulary, dialogue patterns)
- **Validation**: Ensures novels meet quality standards for training
- **Chunking**: Splits into training segments while preserving narrative flow

### 2. Style-Specific Training
Each model is trained using:
- **Style Fidelity Reward**: Matches the original author's writing patterns
- **Narrative Quality Reward**: Ensures coherent storytelling
- **Creativity Reward**: Encourages original content within the style

### 3. Reinforcement Learning
- Uses GRPO (Group Relative Policy Optimization)
- Novel-specific reward functions prevent "reward hacking"
- Trains LoRA adapters for efficient fine-tuning

### 4. Generation and Analysis
- Interactive chat for story generation
- Comparative analysis across different novels
- Study of dataset size impact on style learning

## Example Usage

### Training on Alice in Wonderland
```python
# The system will automatically:
# 1. Analyze Carroll's whimsical, nonsensical style
# 2. Train model to generate similar fantastical narratives
# 3. Learn patterns like dialogue-heavy scenes and imaginative descriptions
```

### Generating New Stories
```bash
> Write a story about a mysterious garden
Generating...

In the heart of the forgotten estate lay a garden most peculiar, where roses bloomed in hues that had no earthly names and hedgerows whispered secrets to the wind. Alice—for indeed it seemed every tale must begin with an Alice—discovered this place quite by accident, as the best discoveries often are...
```

## Experiment Goals

This project investigates:

1. **Data Efficiency**: How much text is needed to learn an author's style?
2. **Style Transfer**: Can models capture subtle writing characteristics?
3. **Creative Generation**: Do trained models generate original yet style-consistent content?
4. **Comparative Analysis**: How do different novels affect training outcomes?

## Technical Details

### Model Configuration
- Base Model: Llama 3.2 3B (4-bit quantized)
- LoRA Rank: 8 (for quick experiments)
- Max Sequence Length: 1024 tokens
- Training Steps: 50 (adjustable)

### Reward Functions
- **Style Fidelity**: Compares generated text style metrics to target novel
- **Narrative Quality**: Evaluates story coherence and literary elements
- **Creativity**: Rewards original content while maintaining style consistency

### Hardware Requirements
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- 16GB+ system RAM
- Works on Google Colab free tier

## Chat Interface Commands

```bash
/help     - Show available commands
/info     - Display model and novel information
/clear    - Clear conversation history
/save     - Save conversation to file
/temp X   - Set generation temperature (0.1-2.0)
/tokens X - Set max generation length (50-800)
/quit     - Exit chat
```

## Example Experiments

### Comparing Victorian vs Modern Styles
```bash
# Train on different eras
novels/pride_and_prejudice.txt    # Jane Austen - formal, structured
novels/alice_in_wonderland.txt    # Lewis Carroll - whimsical, playful

# Compare generated outputs for style differences
```

### Dataset Size Impact
```bash
# The system tracks:
# - Novel word count vs training time
# - Style complexity vs learning efficiency
# - Generation quality vs training data size
```

## Contributing

This is an experimental framework for studying language model training with limited data. Feel free to:

- Add new reward functions
- Experiment with different novels
- Modify training parameters
- Analyze generated content

## Acknowledgments

Based on GPT-OSS RL training principles from the Unsloth framework. The original code demonstrates sophisticated reward engineering and anti-reward-hacking techniques that inspired this novel-focused adaptation.

## License

MIT License - See original GPT-OSS and Unsloth licenses for model usage terms.