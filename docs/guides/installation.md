# Installation Guide

This guide provides detailed instructions for installing and setting up the Model Tea Iterative Novel Training System.

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 10 GB free space
- GPU: Optional (CPU training supported)

**Recommended:**
- CPU: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- RAM: 16-32 GB
- Storage: 50+ GB SSD
- GPU: NVIDIA GPU with 6+ GB VRAM (RTX 3060 or better)

### Software Requirements

**Operating System:**
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

**Python:**
- Python 3.8 or higher
- Python 3.9+ recommended for best performance

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository

```bash
git clone <repository-url>
cd model-trainer
```

#### Step 2: Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Verify Installation

```bash
# Test basic functionality
python -c "from iterative_novel_trainer import IterativeTrainer; print('Installation successful!')"

# Test tagging system
python -c "from corpus_tagger import SimpleNovelTagger; print('Tagging system ready!')"

# Test memory integration
python -c "from episodic_memory_system import EpisodicMemorySystem; print('Memory system ready!')"

# Run test suite
python -m pytest tests/ -v
```

### Method 2: Development Installation

For contributors and developers:

```bash
# Clone repository
git clone <repository-url>
cd model-trainer

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Method 3: Docker Installation

```bash
# Pull Docker image
docker pull model-tea:latest

# Or build from source
docker build -t model-tea .

# Run container
docker run -it --gpus all -v $(pwd)/novels:/app/novels model-tea
```

## Dependency Installation

### Core Dependencies

```bash
# PyTorch (CPU version)
pip install torch torchvision torchaudio

# Transformers library
pip install transformers

# Data processing
pip install numpy pandas datasets

# Evaluation and metrics
pip install scikit-learn

# Utilities
pip install tqdm pathlib dataclasses
```

### GPU Support

#### NVIDIA CUDA

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### AMD ROCm (Linux only)

```bash
# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

### Optional Dependencies

```bash
# Advanced evaluation metrics
pip install scipy matplotlib seaborn

# Web interface (if using)
pip install flask fastapi uvicorn

# Development tools
pip install pytest black flake8 mypy

# Notebook support
pip install jupyter ipykernel

# Progress visualization
pip install wandb tensorboard
```

## Configuration Setup

### Environment Variables

Create a `.env` file in the project root:

```env
# Model cache directory
TRANSFORMERS_CACHE=/path/to/cache

# CUDA settings
CUDA_VISIBLE_DEVICES=0

# Logging level
LOG_LEVEL=INFO

# Memory settings
MAX_MEMORY_MB=16384
```

### Configuration Files

#### Default Configuration

The system creates default configuration files on first run:

```
model-trainer/
├── config/
│   ├── training_config.json
│   ├── memory_config.json
│   └── serving_config.json
└── logs/
    └── training.log
```

#### Custom Configuration

Create custom configuration files:

```json
// config/custom_training.json
{
    "base_model": "gpt2-medium",
    "iterations_per_novel": 15,
    "max_steps_per_iteration": 10,
    "learning_rate_start": 3e-5,
    "learning_rate_end": 1e-5,
    "chunk_size": 250,
    "validation_split": 0.15
}
```

## Novel Setup

### Novel Directory Structure

```
novels/
├── alice_in_wonderland.txt
├── call_of_cthulhu.txt
├── frankenstein.txt
├── pride_and_prejudice.txt
└── subdirectories/
    ├── science_fiction/
    │   ├── foundation.txt
    │   └── dune.txt
    └── fantasy/
        ├── lotr_fellowship.txt
        └── chronicles_narnia.txt
```

### Text Format Requirements

**File Format:**
- Plain text files (.txt)
- UTF-8 encoding
- Unix or Windows line endings

**Content Guidelines:**
- Minimum 1,000 words
- Maximum 500,000 words (for performance)
- Clean text without excessive markup
- Consistent paragraph formatting

**Example Preparation:**

```python
# Script to prepare novel text
def prepare_novel_text(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean text
    text = text.replace('\r\n', '\n')  # Normalize line endings
    text = ' '.join(text.split())      # Normalize whitespace

    # Remove headers/footers if needed
    # text = remove_gutenberg_headers(text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

# Usage
prepare_novel_text('raw/alice.txt', 'novels/alice_in_wonderland.txt')
```

## Performance Optimization

### GPU Setup

#### CUDA Configuration

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
```

#### Memory Management

```python
# config/gpu_config.json
{
    "gpu_memory_fraction": 0.8,
    "mixed_precision": true,
    "gradient_checkpointing": true,
    "max_batch_size": 4
}
```

### CPU Optimization

```python
# config/cpu_config.json
{
    "num_workers": 4,
    "pin_memory": false,
    "use_amp": false,
    "batch_size": 1,
    "gradient_accumulation_steps": 8
}
```

## Verification and Testing

### Quick Verification

```bash
# Test basic import
python -c "from iterative_novel_trainer import IterativeTrainer; print('[OK] Core module imported')"

# Test memory system
python -c "from episodic_memory_system import EpisodicMemorySystem; print('[OK] Memory system imported')"

# Test tagging system
python -c "from corpus_tagger import SimpleNovelTagger; print('[OK] Tagging system imported')"

# Test batch processing
python -c "from batch_tag_novels import batch_tag_novels; print('[OK] Batch processing imported')"

# List available novels
python iterative_novel_trainer.py --list
```

### Comprehensive Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_iterative_trainer.py -v
python -m pytest tests/test_episodic_memory.py -v
python -m pytest tests/test_pipeline.py -v

# Run integration tests
python -m pytest tests/test_integration.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
```

### Performance Testing

```bash
# Quick training test
python iterative_novel_trainer.py --novel alice_in_wonderland --iterations 2 --steps 1

# Tagging system test
python batch_tag_novels.py specific alice_in_wonderland call_of_cthulhu

# Memory system test
python -c "
from episodic_memory_system import EpisodicMemorySystem
from pathlib import Path
memory_system = EpisodicMemorySystem()
result = memory_system.build_memory_for_model('test', Path('novels/alice_in_wonderland.txt'))
print(f'Created {len(result[\"memories\"])} memories')
"

# Pipeline test
python -c "
from model_tea.core.pipeline import MLPipeline, PipelineConfig
from model_tea.core.pipeline import DataPreprocessingStage
pipeline = MLPipeline(PipelineConfig(pipeline_name='test'))
pipeline.add_stage(DataPreprocessingStage({}))
result = pipeline.execute({'raw_data': [1, 2, 3, 4, 5]})
print('Pipeline test successful')
"
```

## Troubleshooting

### Common Installation Issues

#### Issue: ImportError for torch

```bash
# Solution: Install PyTorch explicitly
pip install torch torchvision torchaudio
```

#### Issue: CUDA not available

```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: Out of memory during training

```python
# Solution: Reduce batch size and sequence length
config = IterativeConfig(
    batch_size=1,
    max_seq_length=512,
    gradient_accumulation_steps=8
)
```

#### Issue: Module not found errors

```bash
# Solution: Install in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/model-trainer"
```

### Performance Issues

#### Slow Training

**CPU-bound solutions:**
```python
config = IterativeConfig(
    batch_size=1,
    gradient_accumulation_steps=4,
    max_seq_length=512
)
```

**Memory-bound solutions:**
```python
config = IterativeConfig(
    chunk_size=150,        # Smaller chunks
    validation_split=0.1,  # Less validation data
    max_steps_per_iteration=4  # Fewer steps
)
```

#### Memory Issues

```python
# Reduce memory usage
config = IterativeConfig(
    base_model="gpt2",  # Smaller model
    max_seq_length=256,       # Shorter sequences
    batch_size=1,             # Smaller batches
    gradient_accumulation_steps=8  # Maintain effective batch size
)
```

### Logging and Debugging

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in configuration
LOG_LEVEL=DEBUG python iterative_novel_trainer.py
```

#### Check System Resources

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor CPU and memory
htop

# Check disk space
df -h
```

## Next Steps

After successful installation:

1. **Read the [Training Guide](training.md)** for detailed usage instructions
2. **Review [Feature Documentation](../features/README.md)** for advanced capabilities
3. **Explore [Examples](../examples/)** for common use cases
4. **Check [API Reference](../api/README.md)** for detailed API documentation

## Getting Help

If you encounter issues:

1. **Check the logs** in `logs/training.log`
2. **Review [troubleshooting section](#troubleshooting)** above
3. **Run the test suite** to identify specific problems
4. **Open an issue** on GitHub with:
   - System information
   - Error messages
   - Steps to reproduce
   - Log files

## Updates and Maintenance

### Updating the System

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run tests after update
python -m pytest tests/ -v
```

### Cleaning Up

```bash
# Clear model cache
rm -rf ~/.cache/huggingface/

# Clear training outputs
rm -rf iterative_models/

# Clear memory files
rm -rf memories/

# Reset configuration to defaults
rm -rf config/
```

This completes the installation guide. You should now have a fully functional Model Tea Iterative Novel Training System ready for use.