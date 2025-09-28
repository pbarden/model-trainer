# Iterative Training Feature

The Iterative Training feature is the core methodology of the Model Tea system, implementing a progressive difficulty approach to training language models on literary works.

## Overview

Traditional language model training processes entire texts at once with fixed parameters. Iterative Training instead:

1. **Progressively increases text complexity** across training iterations
2. **Adapts learning rates dynamically** based on iteration progress
3. **Monitors quality continuously** to prevent overfitting
4. **Builds episodic memories incrementally** throughout the process
5. **Provides fine-grained control** over training progression

## Core Methodology

### Progressive Text Chunking

The system starts with small, manageable text chunks and gradually increases their size:

```
Iteration 1:  200-word chunks (High granularity, easy learning)
Iteration 6:  400-word chunks (Medium complexity)
Iteration 12: 640-word chunks (Full complexity, coherent passages)
```

**Benefits:**
- **Easier convergence** on initial iterations
- **Gradual complexity introduction** prevents overwhelming the model
- **Better long-range dependency learning** in later iterations
- **Reduced catastrophic forgetting** through progressive adaptation

### Dynamic Learning Rate Scheduling

Learning rates follow a linear decay schedule across iterations:

```python
def calculate_learning_rate(iteration: int) -> float:
    progress = iteration / total_iterations
    return learning_rate_start * (1 - progress) + learning_rate_end * progress
```

**Example Schedule:**
```
Iteration 1:  5e-5 (Aggressive learning)
Iteration 6:  3.5e-5 (Moderate learning)
Iteration 12: 2e-5 (Fine-tuning)
```

### Quality-Driven Progression

Each iteration includes quality assessment to guide training:

- **Perplexity Monitoring**: Measures model confidence and fluency
- **Generation Quality**: Evaluates coherence, repetition, and style
- **Convergence Detection**: Identifies optimal stopping points
- **Degradation Prevention**: Halts training if quality decreases

## Implementation Details

### Iteration Workflow

```python
for iteration in range(1, iterations_per_novel + 1):
    # 1. Progressive chunking
    chunk_size = calculate_chunk_size(iteration)
    chunks = create_chunks(novel_text, chunk_size)

    # 2. Dynamic learning rate
    learning_rate = calculate_learning_rate(iteration)

    # 3. Training with monitoring
    model = train_iteration(chunks, learning_rate)

    # 4. Quality assessment
    perplexity = calculate_perplexity(model, validation_data)
    quality_score = assess_generation_quality(model)

    # 5. Memory integration
    build_episodic_memories(novel_text, iteration)

    # 6. Progress evaluation
    should_continue = evaluate_progress(perplexity, quality_score)
```

### Chunk Size Calculation

The chunk size follows a progressive schedule:

```python
def calculate_chunk_size(iteration: int, total_iterations: int,
                        start_size: int = 200, end_size: int = 640) -> int:
    progress = (iteration - 1) / (total_iterations - 1)
    return int(start_size + (end_size - start_size) * progress)
```

### Overlap Management

Chunks include controlled overlap to maintain context:

```python
chunk_overlap = min(chunk_size // 4, 50)  # 25% or 50 words max
```

## Configuration Options

### Basic Configuration

```python
config = IterativeConfig(
    iterations_per_novel=12,        # Number of training iterations
    max_steps_per_iteration=8,      # Training steps per iteration
    learning_rate_start=5e-5,       # Initial learning rate
    learning_rate_end=2e-5,         # Final learning rate
    chunk_size=200,                 # Starting chunk size
    validation_split=0.2            # Validation data percentage
)
```

### Advanced Configuration

```python
config = IterativeConfig(
    # Progressive difficulty
    iterations_per_novel=20,
    chunk_size=150,                 # Smaller starting chunks
    chunk_overlap=75,               # More overlap for context

    # Learning dynamics
    learning_rate_start=1e-4,       # Higher initial learning
    learning_rate_end=1e-5,         # Lower final learning
    warmup_ratio=0.15,              # Longer warmup period

    # Quality control
    perplexity_threshold=30.0,      # Stricter perplexity limit
    max_repetition_penalty=1.3,     # Higher repetition penalty

    # Memory integration
    validation_split=0.15           # More training data
)
```

## Training Progression Examples

### Short Story Training (5,000 words)

```
Iteration 1:  25 chunks × 200 words = 5,000 words total
Iteration 6:  13 chunks × 400 words = 5,200 words total
Iteration 10: 8 chunks × 625 words = 5,000 words total
```

**Benefits for short stories:**
- Captures narrative structure quickly
- Maintains coherence across brief plot arcs
- Preserves style and voice consistency

### Novel Training (80,000 words)

```
Iteration 1:  400 chunks × 200 words = 80,000 words
Iteration 6:  200 chunks × 400 words = 80,000 words
Iteration 12: 125 chunks × 640 words = 80,000 words
```

**Benefits for novels:**
- Gradual adaptation to complex narrative structures
- Better handling of character development arcs
- Improved long-range dependency modeling

## Quality Metrics Integration

### Perplexity Tracking

Perplexity measures model confidence and is tracked across iterations:

```python
# Calculate perplexity on validation set
perplexity = calculate_perplexity(model, validation_chunks)

# Status determination
if perplexity < 10:
    status = "excellent"
elif perplexity < 25:
    status = "good"
elif perplexity < 50:
    status = "acceptable"
else:
    status = "concerning"
```

### Generation Quality Assessment

Each iteration includes sample generation and quality evaluation:

```python
# Generate sample text
sample = generate_sample(model, prompt="The old house")

# Assess quality
quality_metrics = {
    "coherence": assess_coherence(sample),
    "repetition": detect_repetition(sample),
    "style_consistency": measure_style_consistency(sample, reference_text),
    "vocabulary_diversity": calculate_vocabulary_diversity(sample)
}
```

## Memory Integration

### Progressive Memory Building

Episodic memories are built incrementally during training:

```python
# Memory building per iteration
if iteration % 3 == 0:  # Every 3rd iteration
    memory_system.build_memories_incremental(
        text_chunk=current_chunks,
        iteration=iteration,
        model_state=model.state_dict()
    )
```

### Memory-Informed Training

Later iterations can leverage accumulated memories:

```python
# Retrieve relevant memories for training context
relevant_memories = memory_system.activate_memories(
    prompt=training_prompt,
    max_memories=5
)

# Enhance training with memory context
enhanced_prompt = f"{training_prompt}\nContext: {format_memories(relevant_memories)}"
```

## Adaptive Strategies

### Dynamic Iteration Adjustment

The system can adapt iteration counts based on progress:

```python
def should_continue_training(current_iteration: int, quality_history: List[float]) -> bool:
    # Check for convergence
    if len(quality_history) >= 3:
        recent_improvement = quality_history[-1] - quality_history[-3]
        if recent_improvement < 0.01:  # Minimal improvement
            return False

    # Check for quality degradation
    if len(quality_history) >= 2:
        if quality_history[-1] < quality_history[-2] * 0.95:  # 5% degradation
            return False

    return current_iteration < max_iterations
```

### Learning Rate Adaptation

Learning rates can be adjusted based on performance:

```python
def adaptive_learning_rate(base_lr: float, iteration: int,
                          quality_score: float, perplexity: float) -> float:
    # Reduce learning rate if perplexity is high
    if perplexity > 50:
        return base_lr * 0.5

    # Increase learning rate if quality is improving rapidly
    if quality_score > 0.95:
        return base_lr * 1.2

    return base_lr
```

## Performance Optimization

### Efficient Chunking

```python
def create_progressive_chunks(text: str, iteration: int) -> List[str]:
    chunk_size = calculate_chunk_size(iteration)
    overlap = min(chunk_size // 4, 50)

    # Use sliding window for efficiency
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap

    return chunks
```

### Memory-Efficient Training

```python
def train_iteration_efficient(chunks: List[str], model, tokenizer):
    # Process chunks in batches to manage memory
    batch_size = 4
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]

        # Tokenize batch
        batch_tokens = tokenizer(batch_chunks, truncation=True,
                               padding=True, return_tensors="pt")

        # Train on batch
        train_on_batch(model, batch_tokens)

        # Clear memory
        torch.cuda.empty_cache()
```

## Use Cases and Applications

### Literary Style Transfer

```python
# Train on source style (e.g., Victorian literature)
source_config = IterativeConfig(
    iterations_per_novel=15,
    learning_rate_start=3e-5,
    chunk_size=300
)

# Fine-tune for target style with fewer iterations
target_config = IterativeConfig(
    iterations_per_novel=8,
    learning_rate_start=1e-5,
    chunk_size=400
)
```

### Genre-Specific Training

```python
# Horror novels - emphasis on atmosphere
horror_config = IterativeConfig(
    chunk_size=250,                 # Longer chunks for atmosphere
    learning_rate_start=4e-5,       # Moderate learning rate
    perplexity_threshold=35.0,      # Allow some uncertainty
    temperature_range=(0.8, 1.1)   # Higher creativity
)

# Technical writing - emphasis on clarity
technical_config = IterativeConfig(
    chunk_size=150,                 # Shorter, focused chunks
    learning_rate_start=6e-5,       # Faster learning
    perplexity_threshold=20.0,      # Require high confidence
    temperature_range=(0.6, 0.8)   # Lower creativity, more precision
)
```

### Multi-Language Support

```python
# Language-specific configurations
configs = {
    "english": IterativeConfig(chunk_size=200, iterations_per_novel=12),
    "french": IterativeConfig(chunk_size=180, iterations_per_novel=14),  # More iterations for complexity
    "german": IterativeConfig(chunk_size=150, iterations_per_novel=16),  # Compound words need smaller chunks
}
```

## Monitoring and Debugging

### Training Progress Visualization

```python
def plot_training_progress(trainer_results: Dict[str, Any]):
    iterations = range(1, len(trainer_results["perplexity_history"]) + 1)

    plt.figure(figsize=(12, 8))

    # Perplexity progression
    plt.subplot(2, 2, 1)
    plt.plot(iterations, trainer_results["perplexity_history"])
    plt.title("Perplexity Progression")
    plt.xlabel("Iteration")
    plt.ylabel("Perplexity")

    # Quality score progression
    plt.subplot(2, 2, 2)
    plt.plot(iterations, trainer_results["quality_history"])
    plt.title("Quality Score Progression")
    plt.xlabel("Iteration")
    plt.ylabel("Quality Score")

    # Learning rate schedule
    plt.subplot(2, 2, 3)
    plt.plot(iterations, trainer_results["learning_rate_history"])
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")

    # Chunk size progression
    plt.subplot(2, 2, 4)
    plt.plot(iterations, trainer_results["chunk_size_history"])
    plt.title("Chunk Size Progression")
    plt.xlabel("Iteration")
    plt.ylabel("Chunk Size (words)")

    plt.tight_layout()
    plt.show()
```

### Early Warning Systems

```python
def check_training_health(iteration: int, perplexity: float,
                         quality_score: float, learning_rate: float) -> List[str]:
    warnings = []

    if perplexity > 100:
        warnings.append(f"Very high perplexity ({perplexity:.1f}) - model may be struggling")

    if quality_score < 0.5:
        warnings.append(f"Low quality score ({quality_score:.3f}) - consider adjusting parameters")

    if iteration > 5 and learning_rate > 3e-5:
        warnings.append("Learning rate may be too high for later iterations")

    return warnings
```

## Best Practices

### Choosing Iteration Count

- **Short texts (< 10K words)**: 8-10 iterations
- **Medium texts (10K-50K words)**: 12-15 iterations
- **Long texts (> 50K words)**: 15-20 iterations
- **Complex literary works**: Add 2-4 extra iterations

### Learning Rate Guidelines

- **Start conservatively**: 3e-5 to 5e-5 for most cases
- **End lower**: 1e-5 to 2e-5 for fine-tuning
- **Adjust for model size**: Larger models need lower rates
- **Monitor perplexity**: Reduce if training becomes unstable

### Chunk Size Strategy

- **Start small**: 150-200 words for complex texts
- **End appropriately**: 500-800 words maximum
- **Consider content**: Dialogue-heavy texts need smaller chunks
- **Overlap wisely**: 20-25% overlap maintains context

## Troubleshooting

### High Perplexity Issues

**Problem**: Perplexity increases during training
**Solutions**:
- Reduce learning rate by 50%
- Increase chunk overlap
- Check for data quality issues
- Reduce batch size

### Quality Degradation

**Problem**: Generation quality decreases in later iterations
**Solutions**:
- Implement early stopping
- Reduce final learning rate
- Increase validation split
- Add regularization

### Memory Issues

**Problem**: Out of memory during training
**Solutions**:
- Reduce batch size
- Implement gradient accumulation
- Use smaller chunk sizes
- Enable gradient checkpointing

## Future Enhancements

### Planned Features

1. **Adaptive Chunking**: Dynamic chunk size based on content complexity
2. **Multi-Objective Optimization**: Balance perplexity, quality, and memory efficiency
3. **Transfer Learning**: Pre-trained iteration schedules for different genres
4. **Distributed Training**: Multi-GPU support for large-scale training

### Research Directions

1. **Curriculum Learning**: Intelligent ordering of training examples
2. **Meta-Learning**: Learning optimal iteration strategies
3. **Neural Architecture Search**: Automatic model architecture adaptation
4. **Continual Learning**: Training on multiple novels without forgetting

## See Also

- [Episodic Memory Integration](memory_integration.md)
- [Quality Assessment](quality_assessment.md)
- [Configuration Management](configuration_management.md)
- [Training Guide](../guides/training.md)
- [Iterative Trainer Module](../modules/iterative_trainer.md)