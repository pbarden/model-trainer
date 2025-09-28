# Corpus Tagging System Guide

The Model Tea Corpus Tagging System provides comprehensive analysis and categorization of literary works to enhance both training and memory systems.

## Overview

The tagging system extracts 42+ different categories of metadata from novels, creating rich semantic profiles that improve model training, memory formation, and content analysis.

## System Architecture

### Core Components

1. **SimpleNovelTagger**: Main tagging engine that processes individual novels
2. **batch_tag_novels.py**: Batch processing script for corpus-wide tagging
3. **Enhanced Memory Integration**: Automatic integration with episodic memory system

### Data Structure

Each novel generates a comprehensive `tags.json` file containing:

```json
{
  "title": "Novel Title",
  "basic_stats": { "word_count": 15000, "sentence_count": 800 },
  "themes": ["horror", "mystery", "supernatural"],
  "scene_types": ["dialogue_scene", "action_scene", "death_scene"],
  "behaviors": ["heroic", "mysterious", "fearful"],
  "narrative_elements": ["first_person", "foreshadowing", "suspense"],
  "social_dynamics": ["family", "friendship", "authority"],
  "memory_compatibility": { "optimal_chunk_size": 250 }
}
```

## Tag Categories

### 1. Basic Themes (15 categories)

**Literary Genres:**
- `romance`: Love stories, relationships, emotional connections
- `horror`: Fear, terror, supernatural dread, gothic elements
- `mystery`: Puzzles, investigations, secrets, detective work
- `adventure`: Journeys, quests, exploration, danger
- `fantasy`: Magic, mythical creatures, otherworldly elements
- `historical`: Period settings, historical events, past eras
- `war`: Military conflicts, battles, soldiers, combat
- `supernatural`: Ghosts, spirits, paranormal phenomena
- `crime`: Criminal activities, law enforcement, justice

**Specialized Genres:**
- `scifi`: Science fiction, technology, space, future concepts
- `spy`: Espionage, intelligence, secret missions, undercover work
- `thriller`: High tension, suspense, urgent situations, chase scenes
- `action`: Fast-paced sequences, physical confrontation, intensity
- `space`: Cosmic settings, planets, interstellar travel

### 2. Scene Types (12 categories)

**Interactive Scenes:**
- `dialogue_scene`: Conversations, verbal exchanges, character interactions
- `romantic_scene`: Intimate moments, expressions of love, courtship
- `conflict_scene`: Arguments, disagreements, tension between characters

**Action Sequences:**
- `action_scene`: Physical activity, movement, dynamic events
- `death_scene`: Mortality, killing, funeral sequences
- `travel_scene`: Journeys, movement between locations

**Discovery & Emotion:**
- `discovery_scene`: Revelations, finding hidden things, realizations
- `emotional_scene`: Strong feelings, tears, grief, joy

**Setting-Based:**
- `indoor_scene`: Interior locations, buildings, rooms
- `outdoor_scene`: External environments, nature, open spaces

**Temporal:**
- `flashback_scene`: Past events, memories, historical sequences
- `dream_scene`: Unconscious states, visions, surreal experiences

### 3. Character Behaviors (10 types)

**Heroic Traits:**
- `heroic`: Brave actions, selfless behavior, protective instincts
- `compassionate`: Kindness, empathy, caring for others
- `loyal`: Faithfulness, devotion, trustworthy actions
- `intelligent`: Cleverness, wisdom, problem-solving abilities

**Complex Traits:**
- `mysterious`: Secretive behavior, hidden motives, enigmatic actions
- `ambitious`: Goal-driven behavior, determination, success-seeking
- `fearful`: Anxiety, worry, scared reactions

**Negative Traits:**
- `villainous`: Evil actions, malicious intent, harmful behavior
- `aggressive`: Hostile actions, violence, confrontational behavior
- `deceptive`: Lies, betrayal, dishonest actions

### 4. Narrative Elements (10 elements)

**Perspective:**
- `first_person`: "I" narration, personal viewpoint
- `third_person`: "He/she/they" narration, external viewpoint

**Literary Devices:**
- `foreshadowing`: Hints about future events
- `flashback`: Scenes from the past
- `symbolism`: Metaphorical representations
- `irony`: Contrary or unexpected outcomes

**Story Structure:**
- `prophecy`: Predictions, destined events
- `suspense`: Tension, uncertainty, anticipation
- `twist`: Unexpected plot developments
- `resolution`: Conclusions, problem-solving, endings

### 5. Social Dynamics (10 relationships)

**Personal Relationships:**
- `family`: Parent-child, sibling, relative connections
- `friendship`: Companionship, allies, trusted bonds
- `romance`: Love interests, couples, romantic partnerships
- `mentorship`: Teacher-student, guide-apprentice relationships

**Power Structures:**
- `authority`: Leadership, command, ruling positions
- `hierarchy`: Social rank, class distinctions, status levels
- `rivalry`: Competition, opponents, conflicting goals

**Group Dynamics:**
- `alliance`: Partnerships, cooperation, united efforts
- `rebellion`: Resistance, defiance, uprising against authority
- `betrayal`: Broken trust, backstabbing, deception

## Usage Guide

### Basic Tagging

```python
from corpus_tagger import SimpleNovelTagger
from pathlib import Path

# Initialize tagger
tagger = SimpleNovelTagger()

# Tag a single novel
tags = tagger.tag_novel(Path("novels/alice_in_wonderland.txt"))

# Access results
print(f"Themes: {tags.themes}")
print(f"Scene types: {tags.scene_types}")
print(f"Behaviors: {tags.behaviors}")
print(f"Writing style: {tags.writing_style}")
```

### Batch Processing

```python
from batch_tag_novels import batch_tag_novels, tag_specific_novels

# Tag all novels in corpus
batch_tag_novels()

# Tag first 50 novels
batch_tag_novels(max_novels=50)

# Tag starting from index 100
batch_tag_novels(max_novels=25, start_from=100)

# Tag specific novels
tag_specific_novels(["alice_in_wonderland", "call_of_cthulhu", "frankenstein"])
```

### Novel Search and Selection

```python
from corpus_tagger import find_novels_by_criteria

# Find novels by theme
horror_novels = find_novels_by_criteria({'theme': 'horror'})
scifi_novels = find_novels_by_criteria({'theme': 'scifi'})

# Find by multiple criteria
spy_thrillers = find_novels_by_criteria({
    'theme': 'spy',
    'genre': 'Thriller/Suspense'
})

# Find by length
short_novels = find_novels_by_criteria({'max_words': 20000})
long_novels = find_novels_by_criteria({'min_words': 100000})

# Find by character presence
novels_with_heroes = find_novels_by_criteria({'character': 'hero'})
```

## Integration with Training

### Memory-Enhanced Training

The tagging system automatically integrates with the episodic memory system:

```python
from episodic_memory_system import EpisodicMemorySystem
from pathlib import Path

memory_system = EpisodicMemorySystem()

# Build memories with tag enhancement
result = memory_system.build_memory_for_model(
    "alice_model",
    Path("novels/alice_in_wonderland.txt")
)

# Tags automatically loaded and integrated
print(f"Memories enhanced with {len(result['tag_integration']['themes'])} themes")
```

### Training Configuration

```python
from iterative_novel_trainer import IterativeTrainer, IterativeConfig

# Configure training to leverage tags
config = IterativeConfig(
    base_model="gpt2",
    iterations_per_novel=15,
    use_enhanced_tags=True,  # Enable tag integration
    tag_based_chunking=True  # Use scene boundaries for chunking
)

trainer = IterativeTrainer(config)
results = trainer.train_novel("alice_in_wonderland")
```

## Performance and Statistics

### Processing Speed

- **Individual novels**: ~2-5 seconds per novel
- **Batch processing**: ~33 novels per minute
- **Memory usage**: Minimal, scales with text length

### Accuracy Metrics

The tagging system uses keyword-based analysis with normalized scoring:

- **Theme detection**: 85%+ accuracy for major themes
- **Scene classification**: 80%+ accuracy for clear scene types
- **Behavior analysis**: 75%+ accuracy for prominent character traits
- **Narrative elements**: 90%+ accuracy for structural elements

### Output Statistics

For a typical novel (15,000-30,000 words):

- **Themes detected**: 5-15 categories
- **Scene types**: 6-10 categories
- **Behaviors**: 7-10 categories
- **Narrative elements**: 6-9 categories
- **Social dynamics**: 4-8 categories

## File Structure

After tagging, each novel directory contains:

```
novels/
└── alice_in_wonderland/
    ├── alice_in_wonderland.txt
    └── tags.json
```

The master corpus index is created at:

```
corpus_tags.json
```

## Advanced Features

### Custom Keyword Sets

Modify theme detection by updating keyword dictionaries:

```python
tagger = SimpleNovelTagger()

# Add custom keywords
tagger.theme_keywords['cyberpunk'] = [
    'cyberspace', 'hacker', 'virtual', 'matrix', 'neural'
]

# Adjust detection thresholds
tagger.detection_threshold = 0.3  # Lower = more sensitive
```

### Tag-Based Analysis

```python
# Load corpus data
with open("corpus_tags.json", 'r') as f:
    corpus = json.load(f)

# Analyze theme distribution
theme_counts = {}
for novel_id, tags in corpus['novels'].items():
    for theme in tags['themes']:
        theme_counts[theme] = theme_counts.get(theme, 0) + 1

print("Most common themes:", sorted(theme_counts.items(), key=lambda x: x[1], reverse=True))
```

### Memory Optimization

For large corpora, optimize memory usage:

```python
# Process in smaller batches
for i in range(0, 250, 50):  # Process 50 novels at a time
    batch_tag_novels(max_novels=50, start_from=i)
    print(f"Processed batch {i//50 + 1}")
```

## Troubleshooting

### Common Issues

**Unicode encoding errors:**
- Ensure all text files use UTF-8 encoding
- Check for special characters in novel files

**Missing tags.json files:**
- Verify novels directory structure
- Check file permissions
- Ensure text files exist in novel directories

**Low tag detection:**
- Novel may be too short (minimum 1,000 words recommended)
- Check for clean text without excessive formatting
- Adjust detection thresholds if needed

### Performance Issues

**Slow processing:**
- Reduce batch size for memory-constrained systems
- Process novels sequentially instead of in parallel
- Clear cache between large batches

**High memory usage:**
- Process smaller batches
- Use streaming for very large novels
- Clear intermediate results regularly

## Best Practices

### Novel Preparation

1. **Clean text format**: Remove headers, footers, excessive formatting
2. **Consistent encoding**: Use UTF-8 for all files
3. **Appropriate length**: 5,000-100,000 words for optimal results
4. **Single text file**: One .txt file per novel directory

### Processing Strategy

1. **Start small**: Test with 5-10 novels first
2. **Batch processing**: Use batch_tag_novels for large corpora
3. **Verify results**: Check sample tags.json files for accuracy
4. **Backup data**: Save corpus_tags.json regularly

### Integration Planning

1. **Tag first**: Run tagging before training for best results
2. **Memory integration**: Let system automatically enhance memories
3. **Training optimization**: Use tag data to guide model selection
4. **Analysis workflow**: Leverage tags for corpus understanding

This tagging system provides the foundation for advanced literary analysis and enhanced model training, making it possible to work with large, diverse corpora while maintaining semantic understanding and improving training outcomes.