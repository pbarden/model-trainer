# Model Tea - Developer Usage Guide

## Quick Start

### Training Combined Models

Train a combined model from multiple novels:

```bash
# Train specific combined model
python combined_model_trainer.py --model vs_mintchip

# Train all combined models
python combined_model_trainer.py --all-models

# List available models
python combined_model_trainer.py --list-models

# Force retrain existing model
python combined_model_trainer.py --model vs_mintchip --force
```

### Interactive Chat

Chat with trained models:

```bash
# Basic usage (use defaults)
python interactive_chat.py

# Custom generation settings
python interactive_chat.py --temperature 0.7 --max-length 300

# All available parameters
python interactive_chat.py \
  --models-dir iterative_models \
  --max-length 250 \
  --temperature 0.5 \
  --top-p 0.85 \
  --top-k 30 \
  --repetition-penalty 1.4
```

---

## Combined Model Trainer

### Configuration

Edit `combined_model_trainer.py` or `CombinedModelConfig` dataclass:

```python
@dataclass
class CombinedModelConfig:
    combine_novels_method: str = "concatenate"  # How to combine novels
    novel_separator: str = "\n\n=== NEW NOVEL ===\n\n"
    min_novels_required: int = 2
    max_combined_size: int = 2000000  # 2M words max

    max_iterations: int = 12  # Training iterations
    learning_rate_start: float = 5e-5
    learning_rate_end: float = 5e-6

    combined_memories_count: int = 350
    cross_novel_memories: bool = True
```

### Model Mapping File

Create `model_mapping.json` to define combined models:

```json
{
  "models": {
    "vs_mintchip": {
      "description": "Very short novels combined model",
      "novels": [
        {
          "directory_name": "novel1",
          "original_name": "Novel One"
        },
        {
          "directory_name": "novel2",
          "original_name": "Novel Two"
        }
      ]
    },
    "horror_collection": {
      "description": "Horror novels collection",
      "novels": [
        {"directory_name": "lovecraft1", "original_name": "Call of Cthulhu"},
        {"directory_name": "poe1", "original_name": "The Raven"}
      ]
    }
  }
}
```

### Training Parameters

Parameters passed to `IterativeConfig`:

- **iterations_per_novel**: Number of training iterations (default: 12)
- **learning_rate_start**: Initial learning rate (default: 5e-5)
- **learning_rate_end**: Final learning rate (default: 5e-6)

These are configurable in `CombinedModelConfig` as:
- `max_iterations`
- `learning_rate_start`
- `learning_rate_end`

### Output Structure

```
iterative_models/
├── vs_mintchip/
│   ├── iteration_0/
│   ├── iteration_1/
│   ├── ...
│   ├── final/              # Use this for chat
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files
│   └── training_results.json
```

---

## Interactive Chat

### Generation Parameters

All parameters can be passed via command-line or modified in-session:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `max_length` | 250 | 50-500 | Response length in tokens |
| `temperature` | 0.5 | 0.1-2.0 | Randomness (lower=focused, higher=creative) |
| `top_p` | 0.85 | 0.1-1.0 | Nucleus sampling (lower=focused, higher=diverse) |
| `top_k` | 30 | 10-200 | Vocabulary limit per step (lower=focused) |
| `repetition_penalty` | 1.4 | 1.0-2.0 | Discourages repetition (higher=less repetitive) |

### In-Chat Commands

```
/help, /h           - Show help menu
/commands           - List all commands
/status             - Show current session status

MODEL MANAGEMENT:
/models             - List available models
/models all         - List all models including checkpoints
/switch             - Switch to different model
/info               - Show current model information

GENERATION SETTINGS:
/settings, /set     - Adjust generation parameters
                      Includes presets: coherent, balanced, creative

CONVERSATION:
/clear              - Clear conversation history
/history            - Show conversation history
/export             - Export conversation to file

EXIT:
/quit, /exit, /q    - Exit chat
```

### Presets

Use `/settings` then option `6` to load presets:

**Coherent** (focused, readable):
```
max_length: 100
temperature: 0.5
top_p: 0.85
top_k: 30
repetition_penalty: 1.4
```

**Balanced** (mix of coherence and creativity):
```
max_length: 150
temperature: 0.7
top_p: 0.9
top_k: 50
repetition_penalty: 1.2
```

**Creative** (varied but less predictable):
```
max_length: 200
temperature: 0.9
top_p: 0.95
top_k: 80
repetition_penalty: 1.15
```

### Recommended Settings by Use Case

**Short, focused responses:**
```bash
python interactive_chat.py --max-length 100 --temperature 0.5 --top-k 30
```

**Story continuation:**
```bash
python interactive_chat.py --max-length 300 --temperature 0.7 --top-p 0.9 --top-k 50
```

**Creative writing:**
```bash
python interactive_chat.py --max-length 250 --temperature 0.8 --top-p 0.92 --repetition-penalty 1.2
```

**Prevent rambling/repetition:**
```bash
python interactive_chat.py --max-length 150 --temperature 0.6 --top-k 40 --repetition-penalty 1.5
```

---

## Model Directory Structure

Expected structure for models:

```
iterative_models/
├── model_name/
│   └── final/              # Final trained model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── vocab.json
│       └── merges.txt
```

Novels should be in:

```
novels/
├── novel_name/
│   ├── novel_name.txt      # Novel content
│   └── analysis.json       # Optional metadata
```

---

## Troubleshooting

### Combined Model Trainer

**Error: "model_mapping.json not found"**
- Create `model_mapping.json` in project root
- See Model Mapping File section above

**Error: "Model has only X novels. Minimum required: 2"**
- Add more novels to the model definition in `model_mapping.json`

**Error: "Novel directory not found"**
- Ensure `directory_name` in mapping matches folder in `novels/`

### Interactive Chat

**Error: "No trained models found"**
- Train models first using `combined_model_trainer.py` or `iterative_novel_trainer.py`
- Check `--models-dir` points to correct location

**Model generates repetitive text**
- Increase `--repetition-penalty` (try 1.5-1.8)
- Decrease `--temperature` (try 0.5-0.6)
- Decrease `--top-k` (try 20-30)

**Model output is incoherent**
- Decrease `--temperature` (try 0.5-0.7)
- Decrease `--top-p` (try 0.8-0.85)
- Decrease `--top-k` (try 30-40)
- Use the "coherent" preset in `/settings`

**Model output too short**
- Increase `--max-length` (try 200-400)
- Check if model is hitting EOS token early

---

## Examples

### Example 1: Train and Chat

```bash
# 1. Train combined model
python combined_model_trainer.py --model horror_anthology

# 2. Start chat with trained model
python interactive_chat.py

# 3. Select the horror_anthology/final model
# 4. Chat with it
```

### Example 2: Custom Settings

```bash
# Start chat with creative settings
python interactive_chat.py \
  --temperature 0.8 \
  --max-length 300 \
  --top-k 60 \
  --repetition-penalty 1.2
```

### Example 3: Switch Models in Session

```bash
python interactive_chat.py

# In chat:
You: /switch
# Select different model from list
# Continue chatting with new model
```

### Example 4: Export Conversation

```bash
python interactive_chat.py

# In chat:
You: Tell me a story about the ocean
Assistant: [generates response]
You: /export
# Saves to: conversation_model_name_20231215_143022.txt
```

---

## API Usage (Python)

### Combined Model Training

```python
from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig

# Create custom config
config = CombinedModelConfig()
config.max_iterations = 15
config.learning_rate_start = 3e-5

# Initialize trainer
trainer = CombinedModelTrainer(config)

# Train specific model
results = trainer.train_combined_model("vs_mintchip")

# Train all models
all_results = trainer.train_all_combined_models()
```

### Interactive Chat

```python
from interactive_chat import InteractiveChat

# Create chat instance with custom settings
chat = InteractiveChat(
    models_dir="iterative_models",
    max_length=300,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2
)

# Run interactive session
chat.run()
```

### Programmatic Generation

```python
from interactive_chat import InteractiveChat

chat = InteractiveChat()
chat.load_model("vs_mintchip/final")

# Generate response
response = chat.generate_response(
    prompt="Once upon a time",
    max_length=200,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2
)

print(response)
```

---

## Performance Notes

- **CPU Training**: All training is CPU-optimized, ~2GB RAM usage
- **Generation**: CPU generation is slower than GPU but functional
- **Model Size**: Each final model is ~300-500MB depending on base model
- **Training Time**: 12 iterations on combined model takes 20-60 minutes depending on total word count

---

## Best Practices

1. **Model Naming**: Use descriptive names in `model_mapping.json` (e.g., `horror_short`, `scifi_collection`)

2. **Novel Selection**: Combine novels with similar:
   - Genre/style
   - Length (within 2x of each other)
   - Writing quality

3. **Generation Settings**:
   - Start with defaults
   - Adjust one parameter at a time
   - Use presets as starting points

4. **Conversation History**:
   - Use `/clear` if model gets confused
   - Export important conversations with `/export`
   - History is limited to last 3-4 exchanges for context management

5. **Training**:
   - Ensure individual novels are preprocessed/cleaned
   - Check `training_results.json` for quality metrics
   - Use `--force` to retrain if needed

---

## Configuration Reference

### CombinedModelConfig

```python
combine_novels_method: str      # "concatenate" or "interleave"
novel_separator: str            # Separator between novels
min_novels_required: int        # Minimum novels per combined model
max_combined_size: int          # Maximum total words

max_iterations: int             # Training iterations
learning_rate_start: float      # Initial LR
learning_rate_end: float        # Final LR

combined_memories_count: int    # Episodic memories to create
cross_novel_memories: bool      # Enable cross-novel memory links
```

### InteractiveChat Parameters

```python
models_dir: str                 # Directory with trained models
max_length: int                 # Max generation tokens
temperature: float              # Sampling temperature
top_p: float                    # Nucleus sampling threshold
top_k: int                      # Top-k sampling
repetition_penalty: float       # Repetition penalty
```

---

## Support Files

- `model_mapping.json` - Combined model definitions
- `training_results.json` - Training metrics and analysis (per model)
- `conversation_*.txt` - Exported chat logs
- `USAGE.md` - This file

For more information, see source code documentation in:
- `combined_model_trainer.py`
- `interactive_chat.py`
- `iterative_novel_trainer.py`
