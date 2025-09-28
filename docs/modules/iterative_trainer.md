# Iterative Novel Trainer Module

The `iterative_novel_trainer.py` module is the core orchestrator of the training system, implementing progressive difficulty training for language models on literary works.

## Overview

The Iterative Novel Trainer uses a unique approach where:
1. Text is progressively chunked into larger pieces across iterations
2. Learning rate decreases over iterations for fine-tuning
3. Quality is monitored using perplexity and generation metrics
4. Episodic memory is built and integrated throughout training

## Classes

### `IterativeConfig`

Configuration dataclass for training parameters.

#### Parameters

```python
@dataclass
class IterativeConfig:
    base_model: str = "gpt2"                    # Base model to fine-tune
    max_seq_length: int = 1024                  # Maximum sequence length
    iterations_per_novel: int = 12              # Number of training iterations
    max_steps_per_iteration: int = 8            # Max steps per iteration
    learning_rate_start: float = 5e-5          # Starting learning rate
    learning_rate_end: float = 2e-5            # Ending learning rate
    chunk_size: int = 200                       # Initial chunk size (words)
    chunk_overlap: int = 50                     # Overlap between chunks
    validation_split: float = 0.2              # Validation data ratio
    batch_size: int = 1                        # Training batch size
    gradient_accumulation_steps: int = 4       # Gradient accumulation
    warmup_ratio: float = 0.1                  # Learning rate warmup
    max_repetition_penalty: float = 1.2        # Repetition penalty
    temperature_range: tuple = (0.7, 1.0)      # Temperature range
    perplexity_threshold: float = 50.0         # Perplexity warning threshold
    novels_dir: str = "novels"                 # Novel directory
    output_dir: str = "iterative_models"       # Output directory
```

### `IterativeTrainer`

Main training orchestrator class.

#### Constructor

```python
def __init__(self, config: IterativeConfig)
```

#### Key Methods

##### `train_novel(novel_name: str) -> Dict[str, Any]`

Trains a model on the specified novel using iterative progression.

**Parameters:**
- `novel_name`: Name of the novel file (without extension)

**Returns:**
- Dictionary containing training results, metrics, and model paths

**Example:**
```python
config = IterativeConfig(iterations_per_novel=10)
trainer = IterativeTrainer(config)
results = trainer.train_novel("alice_in_wonderland")
```

##### `list_available_novels() -> List[str]`

Returns a list of available novels for training.

**Returns:**
- List of novel names found in the novels directory

##### `generate_sample(novel_name: str, prompt: str = None) -> str`

Generates a text sample from a trained model.

**Parameters:**
- `novel_name`: Name of the trained model
- `prompt`: Optional prompt text

**Returns:**
- Generated text sample

##### `calculate_perplexity(model, tokenizer, val_texts: List[str]) -> float`

Calculates perplexity on validation texts.

**Parameters:**
- `model`: Trained model
- `tokenizer`: Model tokenizer
- `val_texts`: List of validation texts

**Returns:**
- Perplexity score

##### `test_generation_quality(model, tokenizer, prompt: str) -> Dict[str, Any]`

Tests generation quality with metrics.

**Parameters:**
- `model`: Trained model
- `tokenizer`: Model tokenizer
- `prompt`: Test prompt

**Returns:**
- Dictionary containing quality metrics

## Training Process

### Iteration Flow

1. **Text Chunking**: Progressive chunk size increase (200 → 640 words)
2. **Data Preparation**: Train/validation split with tokenization
3. **Model Training**: Fine-tuning with decreasing learning rate
4. **Quality Assessment**: Perplexity and generation quality evaluation
5. **Memory Building**: Episodic memory creation and integration
6. **Progress Monitoring**: Continuous quality and convergence tracking

### Progressive Difficulty

The system implements progressive difficulty through:

- **Increasing Chunk Size**:
  - Iteration 1: 200 words
  - Iteration 6: 400 words
  - Iteration 12: 640 words

- **Decreasing Learning Rate**:
  - Linear decay from `learning_rate_start` to `learning_rate_end`
  - Adaptive based on iteration progress

- **Validation Monitoring**:
  - Perplexity tracking
  - Generation quality assessment
  - Early stopping if quality degrades

## Quality Metrics

### Perplexity Monitoring

- Calculated on validation set each iteration
- Warning threshold configurable (default: 50.0)
- Used for training stability assessment

### Generation Quality

- Repetition detection
- Coherence scoring
- Style consistency measurement
- Vocabulary diversity analysis

## Integration Points

### Episodic Memory System

The trainer integrates with the episodic memory system:

```python
# Memory building during training
memory_system = EpisodicMemorySystem()
result = memory_system.build_memory_for_model(model_name, novel_path)
```

### Quality Validator

Quality assessment throughout training:

```python
# Quality validation
validator = QualityValidator()
quality_score = validator.evaluate_sample(generated_text)
```

## Configuration Examples

### Basic Training

```python
config = IterativeConfig(
    base_model="gpt2",
    iterations_per_novel=8,
    max_steps_per_iteration=5
)
```

### Advanced Training

```python
config = IterativeConfig(
    base_model="gpt2-medium",
    iterations_per_novel=15,
    max_steps_per_iteration=12,
    learning_rate_start=1e-4,
    learning_rate_end=1e-5,
    chunk_size=300,
    validation_split=0.15
)
```

### Memory-Optimized Training

```python
config = IterativeConfig(
    base_model="distilgpt2",
    batch_size=1,
    gradient_accumulation_steps=8,
    max_seq_length=512
)
```

## Output Structure

### Model Artifacts

```
iterative_models/
└── {novel_name}/
    ├── iteration_{n}/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── training_state.json
    └── final/
        ├── pytorch_model.bin
        ├── tokenizer.json
        └── model_metrics.json
```

### Training Logs

- Iteration-by-iteration metrics
- Perplexity progression
- Quality scores
- Memory statistics
- Generation samples

## Error Handling

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **High Perplexity**: Check learning rate or data quality
3. **Poor Generation**: Adjust temperature or repetition penalty
4. **Missing Novels**: Verify novels directory and file names

### Recovery Mechanisms

- Automatic checkpointing
- Training state preservation
- Progressive fallback strategies
- Memory cleanup on errors

## Performance Optimization

### GPU Utilization

- Automatic device detection
- Mixed precision training support
- Dynamic batching for memory efficiency

### Memory Management

- Progressive chunk loading
- Tokenizer caching
- Model state cleanup

### Monitoring

- Real-time progress tracking
- Resource usage monitoring
- Quality degradation detection

## Extension Points

### Custom Models

```python
config = IterativeConfig(base_model="path/to/custom/model")
```

### Custom Quality Metrics

```python
# Override quality assessment
trainer.quality_validator = CustomQualityValidator()
```

### Custom Memory Integration

```python
# Custom memory system
trainer.memory_system = CustomMemorySystem()
```

## See Also

- [Episodic Memory System](episodic_memory.md)
- [Quality Validator](quality_validator.md)
- [Model Tea Core Pipeline](model_tea_core.md)
- [Training Guide](../guides/training.md)