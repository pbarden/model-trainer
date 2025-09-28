# Training Guide

This comprehensive guide covers how to effectively use the Model Tea Iterative Novel Training System for training language models on literary works.

## Quick Start

### Basic Training Command

```bash
# Train on a novel with default settings
python iterative_novel_trainer.py --novel alice_in_wonderland

# List available novels
python iterative_novel_trainer.py --list

# Generate sample from trained model
python iterative_novel_trainer.py --novel alice_in_wonderland --generate
```

### First Training Session

```python
from iterative_novel_trainer import IterativeTrainer, IterativeConfig

# Create basic configuration
config = IterativeConfig(
    base_model="gpt2",
    iterations_per_novel=8,
    max_steps_per_iteration=5
)

# Initialize trainer
trainer = IterativeTrainer(config)

# Train on a novel
results = trainer.train_novel("alice_in_wonderland")

print(f"Training completed in {results['total_time']:.1f} seconds")
print(f"Final quality score: {results['final_quality']:.3f}")
```

## Training Configuration

### Basic Configuration Parameters

```python
config = IterativeConfig(
    # Model settings
    base_model="gpt2",                    # Base model to fine-tune
    max_seq_length=1024,                  # Maximum sequence length

    # Training progression
    iterations_per_novel=12,              # Number of training iterations
    max_steps_per_iteration=8,            # Training steps per iteration

    # Learning rate schedule
    learning_rate_start=5e-5,             # Initial learning rate
    learning_rate_end=2e-5,               # Final learning rate

    # Data processing
    chunk_size=200,                       # Starting chunk size (words)
    chunk_overlap=50,                     # Overlap between chunks
    validation_split=0.2,                 # Validation data ratio

    # Training parameters
    batch_size=1,                         # Training batch size
    gradient_accumulation_steps=4,        # Gradient accumulation
    warmup_ratio=0.1,                     # Learning rate warmup

    # Quality control
    perplexity_threshold=50.0,            # Perplexity warning threshold
    max_repetition_penalty=1.2,           # Repetition penalty
    temperature_range=(0.7, 1.0),         # Temperature for generation

    # Directories
    novels_dir="novels",                  # Novel directory
    output_dir="iterative_models"         # Output directory
)
```

### Configuration for Different Use Cases

#### Quick Testing
```python
config = IterativeConfig(
    iterations_per_novel=3,
    max_steps_per_iteration=2,
    chunk_size=150,
    validation_split=0.3
)
```

#### High-Quality Training
```python
config = IterativeConfig(
    base_model="gpt2-medium",
    iterations_per_novel=20,
    max_steps_per_iteration=15,
    learning_rate_start=3e-5,
    learning_rate_end=1e-5,
    chunk_size=300,
    validation_split=0.15
)
```

#### Memory-Constrained Training
```python
config = IterativeConfig(
    base_model="distilgpt2",
    max_seq_length=512,
    batch_size=1,
    gradient_accumulation_steps=8,
    chunk_size=150
)
```

## Training Process

### Understanding Iterative Training

The system trains in progressive iterations, each with increasing complexity:

1. **Iteration 1**: Small chunks (200 words), high learning rate
2. **Iteration 6**: Medium chunks (400 words), moderate learning rate
3. **Iteration 12**: Large chunks (640 words), low learning rate

Each iteration includes:
- Text chunking and preprocessing
- Model training
- Quality assessment
- Memory integration
- Progress evaluation

### Monitoring Training Progress

```python
# Training with progress monitoring
trainer = IterativeTrainer(config)

# Set up logging to monitor progress
import logging
logging.basicConfig(level=logging.INFO)

# Train with progress callbacks
results = trainer.train_novel("alice_in_wonderland")

# Review training statistics
print("Iteration Details:")
for i, iteration_data in enumerate(results["iteration_details"], 1):
    print(f"  Iteration {i}:")
    print(f"    Time: {iteration_data['time']:.1f}s")
    print(f"    Perplexity: {iteration_data['perplexity']:.2f}")
    print(f"    Quality: {iteration_data['quality']:.3f}")
    print(f"    Chunks: {iteration_data['chunk_count']}")
```

### Training Output Structure

After training, you'll find:

```
iterative_models/
└── alice_in_wonderland/
    ├── iteration_1/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── training_state.json
    ├── iteration_2/
    │   └── ...
    ├── final/
    │   ├── pytorch_model.bin
    │   ├── tokenizer.json
    │   ├── config.json
    │   └── model_metrics.json
    └── training_log.txt
```

## Advanced Training Techniques

### Custom Training Loops

```python
class CustomIterativeTrainer(IterativeTrainer):
    def custom_train_novel(self, novel_name: str) -> Dict[str, Any]:
        novel_path = self.novels_dir / novel_name
        novel_text = self.load_novel(novel_path)

        results = {"iterations": []}

        for iteration in range(1, self.config.iterations_per_novel + 1):
            # Custom chunking strategy
            chunks = self.create_custom_chunks(novel_text, iteration)

            # Custom learning rate calculation
            learning_rate = self.calculate_adaptive_learning_rate(iteration, results)

            # Train iteration with custom parameters
            iteration_result = self.train_iteration_custom(chunks, learning_rate)

            results["iterations"].append(iteration_result)

            # Custom stopping criteria
            if self.should_stop_early(results):
                break

        return results

    def create_custom_chunks(self, text: str, iteration: int) -> List[str]:
        # Implement custom chunking logic
        pass

    def calculate_adaptive_learning_rate(self, iteration: int, results: Dict) -> float:
        # Implement adaptive learning rate
        pass
```

### Multi-Novel Training

```python
def train_multiple_novels(novel_list: List[str], base_config: IterativeConfig):
    results = {}

    for novel in novel_list:
        print(f"Training on {novel}...")

        # Customize config per novel
        novel_config = customize_config_for_novel(base_config, novel)

        trainer = IterativeTrainer(novel_config)
        novel_results = trainer.train_novel(novel)

        results[novel] = novel_results

        # Optional: Use previous novel's model as starting point
        if len(results) > 1:
            base_config.base_model = f"iterative_models/{novel}/final"

    return results

def customize_config_for_novel(base_config: IterativeConfig, novel: str) -> IterativeConfig:
    # Adjust configuration based on novel characteristics
    if "short" in novel.lower():
        base_config.iterations_per_novel = 8
        base_config.chunk_size = 150
    elif "long" in novel.lower():
        base_config.iterations_per_novel = 20
        base_config.chunk_size = 300

    return base_config
```

### Transfer Learning

```python
def fine_tune_from_previous_model(source_novel: str, target_novel: str):
    # Start with pre-trained model from source novel
    config = IterativeConfig(
        base_model=f"iterative_models/{source_novel}/final",
        iterations_per_novel=8,  # Fewer iterations for fine-tuning
        learning_rate_start=1e-5,  # Lower learning rate
        learning_rate_end=5e-6
    )

    trainer = IterativeTrainer(config)
    results = trainer.train_novel(target_novel)

    return results
```

## Memory Integration

### Episodic Memory Configuration

```python
from episodic_memory_system import EpisodicMemorySystem, MemoryConfig

# Configure memory system
memory_config = MemoryConfig(
    memory_chunk_size=35,
    max_memories_per_novel=300,
    max_retrieved_memories=7,
    relevance_threshold=0.08
)

# Initialize memory system
memory_system = EpisodicMemorySystem(memory_config)

# Build memories during training
memory_result = memory_system.build_memory_for_model(
    model_name="alice_model",
    novel_path=Path("novels/alice_in_wonderland.txt")
)

print(f"Created {len(memory_result['memories'])} memories")
```

### Memory-Augmented Training

```python
def train_with_memory_integration(novel_name: str):
    # Standard training configuration
    training_config = IterativeConfig()
    trainer = IterativeTrainer(training_config)

    # Memory system configuration
    memory_config = MemoryConfig()
    memory_system = EpisodicMemorySystem(memory_config)

    # Train model
    training_results = trainer.train_novel(novel_name)

    # Build episodic memories
    novel_path = Path(f"novels/{novel_name}.txt")
    memory_results = memory_system.build_memory_for_model(novel_name, novel_path)

    # Combine results
    return {
        "training": training_results,
        "memory": memory_results,
        "model_path": f"iterative_models/{novel_name}/final"
    }
```

### Memory-Enhanced Generation

```python
def generate_with_memory(model_name: str, prompt: str, memory_system: EpisodicMemorySystem):
    # Activate relevant memories
    memories, stats = memory_system.activate_memories(model_name, prompt)

    # Format memory context
    memory_context = "\n".join([
        f"Memory: {memory.content}" for memory in memories[:3]
    ])

    # Enhanced prompt with memory context
    enhanced_prompt = f"Context:\n{memory_context}\n\nPrompt: {prompt}\nResponse:"

    # Load and use model for generation
    # (Implementation depends on your model loading approach)
    return enhanced_prompt
```

## Quality Assessment and Monitoring

### Understanding Quality Metrics

The system tracks multiple quality indicators:

1. **Perplexity**: Model confidence (lower is better)
2. **Quality Score**: Overall generation quality (0-1, higher is better)
3. **Repetition**: Detection of repetitive patterns
4. **Coherence**: Logical flow and consistency

### Setting Quality Thresholds

```python
config = IterativeConfig(
    perplexity_threshold=30.0,      # Stop if perplexity exceeds this
    min_quality_score=0.7,          # Minimum acceptable quality
    max_repetition_ratio=0.15       # Maximum allowable repetition
)
```

### Custom Quality Assessment

```python
from quality_validator import QualityValidator, ValidationConfig

# Configure quality validator
validation_config = ValidationConfig(
    check_repetition=True,
    check_coherence=True,
    check_style_consistency=True,
    repetition_threshold=0.2,
    coherence_threshold=0.6
)

validator = QualityValidator(validation_config)

# Use in training loop
def assess_iteration_quality(model, tokenizer, validation_data):
    # Generate sample text
    sample = generate_sample(model, tokenizer, "The ancient castle")

    # Assess quality
    quality_results = validator.evaluate_sample(sample)

    return {
        "overall_score": quality_results.overall_score,
        "repetition_score": quality_results.repetition_score,
        "coherence_score": quality_results.coherence_score,
        "style_score": quality_results.style_score
    }
```

## Optimization Strategies

### Performance Optimization

#### GPU Optimization

```python
# GPU-optimized configuration
gpu_config = IterativeConfig(
    batch_size=4,                    # Larger batch size
    gradient_accumulation_steps=2,   # Fewer accumulation steps
    max_seq_length=1024,             # Full sequence length
    use_mixed_precision=True,        # Enable mixed precision
    dataloader_num_workers=4         # Parallel data loading
)
```

#### CPU Optimization

```python
# CPU-optimized configuration
cpu_config = IterativeConfig(
    batch_size=1,                    # Minimal batch size
    gradient_accumulation_steps=8,   # More accumulation
    max_seq_length=512,              # Shorter sequences
    use_mixed_precision=False,       # Disable mixed precision
    dataloader_num_workers=2         # Fewer workers
)
```

### Memory Optimization

```python
# Memory-efficient training
memory_config = IterativeConfig(
    chunk_size=150,                  # Smaller chunks
    validation_split=0.1,            # Less validation data
    gradient_checkpointing=True,     # Enable checkpointing
    clear_cache_every_n_steps=10     # Regular cache clearing
)
```

### Time Optimization

```python
# Fast training configuration
fast_config = IterativeConfig(
    iterations_per_novel=6,          # Fewer iterations
    max_steps_per_iteration=4,       # Fewer steps
    chunk_size=250,                  # Larger chunks
    validation_split=0.05,           # Minimal validation
    early_stopping_patience=2        # Early stopping
)
```

## Troubleshooting Training Issues

### Common Problems and Solutions

#### High Perplexity

**Problem**: Perplexity increases during training
```python
# Solution: Reduce learning rate
config.learning_rate_start = 2e-5
config.learning_rate_end = 5e-6

# Or increase validation split
config.validation_split = 0.25

# Or reduce batch size
config.batch_size = 1
config.gradient_accumulation_steps = 8
```

#### Poor Generation Quality

**Problem**: Generated text is incoherent or repetitive
```python
# Solution: Adjust generation parameters
config.temperature_range = (0.6, 0.8)  # Lower temperature
config.max_repetition_penalty = 1.5    # Higher repetition penalty

# Or train for more iterations
config.iterations_per_novel = 20
config.max_steps_per_iteration = 12
```

#### Out of Memory

**Problem**: CUDA out of memory errors
```python
# Solution: Reduce memory usage
config.max_seq_length = 256
config.batch_size = 1
config.gradient_accumulation_steps = 16
config.gradient_checkpointing = True

# Clear cache regularly
import torch
torch.cuda.empty_cache()
```

#### Training Too Slow

**Problem**: Training takes too long
```python
# Solution: Optimize for speed
config.iterations_per_novel = 8
config.max_steps_per_iteration = 4
config.chunk_size = 300
config.validation_split = 0.05
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use smaller configuration for debugging
debug_config = IterativeConfig(
    iterations_per_novel=2,
    max_steps_per_iteration=1,
    chunk_size=100,
    validation_split=0.5
)

trainer = IterativeTrainer(debug_config)
```

## Best Practices

### Novel Preparation

1. **Text Quality**: Ensure clean, well-formatted text
2. **Length**: Optimal range is 10,000-100,000 words
3. **Encoding**: Use UTF-8 encoding
4. **Structure**: Remove headers, footers, and excessive formatting

### Training Strategy

1. **Start Small**: Begin with short iterations for testing
2. **Monitor Progress**: Watch perplexity and quality scores
3. **Be Patient**: Good training takes time
4. **Save Checkpoints**: Enable checkpointing for long training sessions

### Configuration Guidelines

1. **Learning Rate**: Start conservative (3e-5 to 5e-5)
2. **Iterations**: 12-15 for most novels
3. **Chunk Size**: 200-300 words initially
4. **Validation**: 15-20% for good evaluation

### Resource Management

1. **GPU Memory**: Monitor VRAM usage
2. **System Memory**: Watch RAM consumption
3. **Storage**: Ensure adequate disk space
4. **Time**: Allow 2-6 hours for full training

## Example Training Sessions

### Training Session 1: Classic Literature

```python
# Training on Pride and Prejudice
config = IterativeConfig(
    base_model="gpt2",
    iterations_per_novel=15,
    learning_rate_start=4e-5,
    chunk_size=250,
    temperature_range=(0.7, 0.9)
)

trainer = IterativeTrainer(config)
results = trainer.train_novel("pride_and_prejudice")

# Results analysis
print(f"Training time: {results['total_time']/60:.1f} minutes")
print(f"Final perplexity: {results['final_perplexity']:.2f}")
print(f"Memory count: {results['memory_count']}")
```

### Training Session 2: Science Fiction

```python
# Training on Foundation
config = IterativeConfig(
    base_model="gpt2-medium",
    iterations_per_novel=18,
    learning_rate_start=3e-5,
    chunk_size=300,
    max_seq_length=1024,
    temperature_range=(0.8, 1.0)
)

trainer = IterativeTrainer(config)
results = trainer.train_novel("foundation")

# Generate sample to test sci-fi style
sample = trainer.generate_sample("foundation", "The galactic empire")
print(f"Generated sample: {sample}")
```

### Training Session 3: Multiple Novels

```python
# Training on multiple related novels
novels = ["frankenstein", "dr_jekyll_mr_hyde", "dracula"]
base_config = IterativeConfig(
    iterations_per_novel=12,
    learning_rate_start=3e-5
)

results = {}
for novel in novels:
    print(f"Training on {novel}...")
    trainer = IterativeTrainer(base_config)
    results[novel] = trainer.train_novel(novel)

    # Use previous model as base for next training
    base_config.base_model = f"iterative_models/{novel}/final"

# Compare results
for novel, result in results.items():
    print(f"{novel}: Quality {result['final_quality']:.3f}, "
          f"Time {result['total_time']/60:.1f}min")
```

## Next Steps

After completing training:

1. **Test Generation**: Generate samples to evaluate quality
2. **Deploy Model**: Use Model Tea serving for production deployment
3. **Analyze Memories**: Explore the episodic memory system
4. **Fine-tune**: Adjust configuration based on results
5. **Scale Up**: Train on larger or multiple novels

For more advanced usage, see:
- [Feature Documentation](../features/README.md)
- [API Reference](../api/README.md)
- [Module Documentation](../modules/README.md)

## Getting Help

If you encounter issues:
1. Check the logs in `logs/training.log`
2. Review this troubleshooting section
3. Test with debug configuration
4. Open an issue on GitHub with detailed information