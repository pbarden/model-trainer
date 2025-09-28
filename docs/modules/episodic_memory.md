# Episodic Memory System Module

The `episodic_memory_system.py` module implements a human-like memory system for AI models, enabling contextual memory storage, retrieval, and activation during training and inference.

## Overview

The Episodic Memory System mimics human memory patterns by:
1. Extracting meaningful memories from text
2. Categorizing memories by type (character, location, dialogue, etc.)
3. Storing memories with emotional context and importance scores
4. Retrieving relevant memories based on query similarity
5. Providing memory activation for enhanced model context

## Classes

### `MemoryConfig`

Configuration for the episodic memory system.

#### Parameters

```python
@dataclass
class MemoryConfig:
    # Memory extraction settings
    memory_chunk_size: int = 35                 # Words per memory chunk
    overlap_ratio: float = 0.3                  # Overlap between chunks
    max_memories_per_novel: int = 250           # Maximum memories to store

    # Memory types to extract
    extract_characters: bool = True             # Extract character memories
    extract_locations: bool = True              # Extract location memories
    extract_emotions: bool = True               # Extract emotional memories
    extract_themes: bool = True                 # Extract thematic memories
    extract_dialogue: bool = True               # Extract dialogue memories
    extract_descriptions: bool = True           # Extract descriptive memories

    # Retrieval settings
    max_retrieved_memories: int = 5             # Memories to activate
    relevance_threshold: float = 0.1            # Minimum similarity threshold
    randomness_factor: float = 0.15             # Randomness in retrieval

    # CPU optimization
    use_simple_similarity: bool = True          # Use CPU-friendly similarity
    memory_cache_size: int = 100                # Cache size for retrieval
```

### `Memory`

Individual memory entry dataclass.

#### Structure

```python
@dataclass
class Memory:
    content: str                    # Memory text content
    memory_type: str               # Type: character, location, dialogue, etc.
    keywords: List[str]            # Extracted keywords
    context: str                   # Contextual information
    emotional_tone: str            # Emotional classification
    importance_score: float        # Importance rating (0-1)
    chapter_position: float        # Position in text (0-1)
```

### `MemoryExtractor`

Extracts different types of memories from novel text.

#### Key Methods

##### `extract_memories(novel_text: str, novel_title: str) -> List[Memory]`

Main extraction method that processes text and creates memories.

**Parameters:**
- `novel_text`: Full text of the novel
- `novel_title`: Title for context

**Returns:**
- List of extracted Memory objects

##### `_create_chunks(text: str) -> List[str]`

Chunks text into overlapping segments for memory extraction.

**Parameters:**
- `text`: Input text to chunk

**Returns:**
- List of text chunks

##### `_extract_character_memories(chunk: str, position: float) -> List[Memory]`

Extracts character-related memories from text chunks.

##### `_extract_location_memories(chunk: str, position: float) -> List[Memory]`

Extracts location and setting memories.

##### `_extract_dialogue_memories(chunk: str, position: float) -> List[Memory]`

Extracts dialogue and conversation memories.

##### `_detect_emotional_tone(text: str) -> str`

Analyzes emotional tone of text.

**Returns:**
- Emotional classification: "positive", "negative", "neutral", etc.

##### `_calculate_importance(text: str) -> float`

Calculates importance score based on content analysis.

**Returns:**
- Importance score between 0.0 and 1.0

### `MemoryRetriever`

Handles memory retrieval and similarity calculation.

#### Key Methods

##### `retrieve_memories(prompt: str, memories: List[Memory]) -> List[Memory]`

Retrieves relevant memories based on a prompt.

**Parameters:**
- `prompt`: Query text for retrieval
- `memories`: Available memory pool

**Returns:**
- List of relevant memories, ranked by relevance

##### `_calculate_relevance(prompt_words: set, memory: Memory) -> float`

Calculates relevance score between prompt and memory.

**Returns:**
- Relevance score between 0.0 and 1.0

### `EpisodicMemorySystem`

Main orchestrator for the memory system.

#### Constructor

```python
def __init__(self, config: MemoryConfig = None)
```

#### Key Methods

##### `build_memory_for_model(model_name: str, novel_path: Path) -> Dict[str, Any]`

Builds complete memory system for a model from a novel.

**Parameters:**
- `model_name`: Identifier for the model
- `novel_path`: Path to the novel file

**Returns:**
- Dictionary containing memories and analysis

**Example:**
```python
memory_system = EpisodicMemorySystem()
result = memory_system.build_memory_for_model("my_model", Path("novels/alice.txt"))
memories = result["memories"]
analysis = result["analysis"]
```

##### `load_memory_for_model(model_name: str) -> bool`

Loads existing memory system for a model.

**Parameters:**
- `model_name`: Model identifier

**Returns:**
- True if memories were loaded successfully

##### `activate_memories(model_name: str, prompt: str) -> Tuple[List[Memory], Dict[str, Any]]`

Activates relevant memories for a given prompt.

**Parameters:**
- `model_name`: Model identifier
- `prompt`: Input prompt for memory activation

**Returns:**
- Tuple of (relevant_memories, activation_stats)

**Example:**
```python
memories, stats = memory_system.activate_memories("my_model", "Tell me about the castle")
```

## Memory Types

### Character Memories

Extracted from text containing:
- Character names and descriptions
- Character actions and behaviors
- Character relationships
- Character development moments

**Example:**
```python
Memory(
    content="Alice was beginning to get very tired of sitting by her sister",
    memory_type="character",
    keywords=["Alice", "tired", "sister"],
    emotional_tone="restless",
    importance_score=0.6
)
```

### Location Memories

Extracted from text describing:
- Physical settings and environments
- Geographic locations
- Architectural descriptions
- Atmospheric details

**Example:**
```python
Memory(
    content="The rabbit-hole went straight on like a tunnel for some way",
    memory_type="location",
    keywords=["rabbit-hole", "tunnel", "straight"],
    emotional_tone="mysterious",
    importance_score=0.8
)
```

### Dialogue Memories

Extracted from:
- Direct speech and conversations
- Important declarations
- Character interactions
- Memorable quotes

### Emotional Memories

Captured from:
- Emotionally charged scenes
- Mood and atmosphere descriptions
- Character emotional states
- Dramatic moments

### Thematic Memories

Identified from:
- Recurring themes and motifs
- Symbolic content
- Philosophical insights
- Story morals and lessons

## Memory Storage

### File Structure

```
memories/
└── {model_name}/
    ├── memories.json           # Serialized memories
    ├── memory_index.json       # Search index
    └── memory_stats.json       # Analysis statistics
```

### Memory Serialization

```python
# Memory objects are serialized to JSON
{
    "content": "Memory text content",
    "memory_type": "character",
    "keywords": ["keyword1", "keyword2"],
    "context": "Contextual information",
    "emotional_tone": "positive",
    "importance_score": 0.75,
    "chapter_position": 0.35
}
```

## Retrieval Mechanisms

### Similarity Calculation

The system uses multiple similarity measures:

1. **Keyword Overlap**: Direct keyword matching
2. **Semantic Similarity**: Contextual word relationships
3. **Emotional Resonance**: Emotional tone matching
4. **Positional Relevance**: Chapter position consideration

### Ranking Algorithm

```python
relevance_score = (
    keyword_similarity * 0.4 +
    semantic_similarity * 0.3 +
    importance_score * 0.2 +
    emotional_match * 0.1
)
```

### Memory Activation

Memory activation follows human-like patterns:
- **Recency Effect**: Recent memories have higher activation
- **Importance Boost**: High-importance memories activate more easily
- **Associative Chains**: Related memories activate together
- **Randomness Factor**: Introduces human-like memory imperfection

## Integration Examples

### Training Integration

```python
# During training iterations
memory_system = EpisodicMemorySystem()
novel_path = Path("novels/alice_in_wonderland.txt")

# Build memories
result = memory_system.build_memory_for_model("alice_model", novel_path)
print(f"Created {len(result['memories'])} memories")

# Use memories during training
for prompt in training_prompts:
    memories, stats = memory_system.activate_memories("alice_model", prompt)
    enhanced_prompt = f"{prompt}\n\nRelevant memories:\n"
    for memory in memories:
        enhanced_prompt += f"- {memory.content}\n"
```

### Inference Integration

```python
# During model inference
def generate_with_memory(model, tokenizer, prompt, memory_system, model_name):
    # Activate relevant memories
    memories, stats = memory_system.activate_memories(model_name, prompt)

    # Enhance prompt with memory context
    memory_context = "\n".join([f"Memory: {m.content}" for m in memories])
    enhanced_prompt = f"Context:\n{memory_context}\n\nPrompt: {prompt}"

    # Generate with enhanced context
    inputs = tokenizer(enhanced_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Configuration Examples

### High-Quality Memory Extraction

```python
config = MemoryConfig(
    memory_chunk_size=50,           # Larger chunks for detail
    max_memories_per_novel=500,     # More memories
    relevance_threshold=0.05,       # Lower threshold
    max_retrieved_memories=10       # More context
)
```

### Fast, Lightweight Configuration

```python
config = MemoryConfig(
    memory_chunk_size=25,           # Smaller chunks
    max_memories_per_novel=100,     # Fewer memories
    relevance_threshold=0.2,        # Higher threshold
    use_simple_similarity=True      # CPU-optimized
)
```

### Character-Focused Configuration

```python
config = MemoryConfig(
    extract_characters=True,
    extract_locations=False,
    extract_emotions=True,
    extract_themes=False,
    extract_dialogue=True,
    extract_descriptions=False
)
```

## Performance Optimization

### Memory Efficiency

- **Chunk-based Processing**: Processes text in manageable chunks
- **Selective Extraction**: Configurable memory types
- **Importance Filtering**: Keeps only significant memories
- **Cache Management**: LRU cache for retrieval operations

### CPU Optimization

- **Simple Similarity**: Fast keyword-based matching
- **Batch Processing**: Vectorized operations where possible
- **Index Caching**: Pre-computed search indices
- **Memory Pooling**: Reuses memory objects

### Scalability

- **Incremental Building**: Add memories without rebuilding
- **Distributed Storage**: File-based memory persistence
- **Configurable Limits**: Adjustable memory constraints
- **Lazy Loading**: Load memories on-demand

## Analysis and Statistics

### Memory Analysis

The system provides detailed analysis:

```python
{
    "total_memories": 245,
    "memory_distribution": {
        "character": 45,
        "location": 38,
        "dialogue": 52,
        "emotion": 41,
        "theme": 35,
        "description": 34
    },
    "average_importance": 0.67,
    "emotional_distribution": {
        "positive": 78,
        "negative": 45,
        "neutral": 122
    },
    "memory_density": 2.45  # memories per 1000 words
}
```

### Retrieval Statistics

```python
{
    "memories_activated": 5,
    "average_relevance": 0.73,
    "activation_time_ms": 12.5,
    "memory_types_used": ["character", "location", "dialogue"],
    "emotional_coherence": 0.82
}
```

## Error Handling

### Common Issues

1. **Empty Text**: Graceful handling of empty or minimal text
2. **Memory Overflow**: Automatic pruning when limits exceeded
3. **Encoding Issues**: Robust text processing and cleaning
4. **Missing Files**: Fallback mechanisms for missing memory files

### Recovery Strategies

- **Fallback Extraction**: Alternative methods if primary extraction fails
- **Memory Reconstruction**: Rebuild from source if corruption detected
- **Partial Loading**: Load available memories even if some are corrupted
- **Default Memories**: Provide basic memories if extraction completely fails

## Extension Points

### Custom Memory Types

```python
class CustomMemoryExtractor(MemoryExtractor):
    def _extract_custom_memories(self, chunk: str, position: float) -> List[Memory]:
        # Custom extraction logic
        pass
```

### Custom Similarity Metrics

```python
class CustomMemoryRetriever(MemoryRetriever):
    def _calculate_relevance(self, prompt_words: set, memory: Memory) -> float:
        # Custom relevance calculation
        pass
```

### External Memory Storage

```python
class DatabaseMemorySystem(EpisodicMemorySystem):
    def save_memories(self, memories: List[Memory]):
        # Save to database instead of files
        pass
```

## See Also

- [Iterative Novel Trainer](iterative_trainer.md)
- [Quality Validator](quality_validator.md)
- [Feature Documentation - Memory Integration](../features/memory_integration.md)
- [API Reference - Memory System](../api/memory_system.md)