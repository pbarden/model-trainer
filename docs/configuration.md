# Configuration

Model Tea uses a centralized configuration system with environment variable support.

## Environment Variables

All configuration values can be set via environment variables with the `MODEL_TEA_` prefix.

### Quick Start

```bash
# Use custom base model
MODEL_TEA_BASE_MODEL=gpt2 model-tea train direct frankenstein

# Change API port
MODEL_TEA_API_PORT=9000 python -m model_tea.api.server

# Adjust training parameters
MODEL_TEA_BATCH_SIZE=16 MODEL_TEA_MAX_ITERATIONS=20 model-tea train train-model horror
```

### Using .env File

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
MODEL_TEA_BASE_MODEL=distilgpt2
MODEL_TEA_BATCH_SIZE=8
MODEL_TEA_API_PORT=8000
MODEL_TEA_DEFAULT_TEMPERATURE=0.7
```

## Configuration Categories

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_BASE_MODEL` | `distilgpt2` | Base model for training |
| `MODEL_TEA_ITERATIVE_BASE_MODEL` | `distilgpt2` | Base model for iterative training |
| `MODEL_TEA_MAX_SEQ_LENGTH` | `1024` | Maximum sequence length |
| `MODEL_TEA_ITERATIVE_MAX_SEQ_LENGTH` | `512` | Max sequence length (iterative) |

### Training Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_LEARNING_RATE_START` | `5e-5` | Starting learning rate |
| `MODEL_TEA_LEARNING_RATE_END` | `5e-6` | Ending learning rate |
| `MODEL_TEA_ITERATIVE_LEARNING_RATE_START` | `2e-5` | Starting LR (iterative) |
| `MODEL_TEA_ITERATIVE_LEARNING_RATE_END` | `5e-6` | Ending LR (iterative) |
| `MODEL_TEA_ITERATIONS_PER_NOVEL` | `12` | Iterations per novel |
| `MODEL_TEA_MAX_STEPS_PER_ITERATION` | `20` | Max steps per iteration |
| `MODEL_TEA_ITERATIVE_MAX_STEPS` | `125` | Max steps (iterative) |
| `MODEL_TEA_WARMUP_STEPS` | `5` | Warmup steps |
| `MODEL_TEA_WARMUP_RATIO` | `0.15` | Warmup ratio (iterative) |
| `MODEL_TEA_BATCH_SIZE` | `8` | Training batch size |
| `MODEL_TEA_GRADIENT_ACCUMULATION_STEPS` | `2` | Gradient accumulation |
| `MODEL_TEA_CHUNK_SIZE` | `200` | Text chunk size |
| `MODEL_TEA_ITERATIVE_CHUNK_SIZE` | `150` | Chunk size (iterative) |
| `MODEL_TEA_CHUNK_OVERLAP` | `30` | Chunk overlap |
| `MODEL_TEA_VALIDATION_SPLIT` | `0.2` | Validation split ratio |
| `MODEL_TEA_ITERATIVE_VALIDATION_SPLIT` | `0.15` | Validation split (iterative) |

### LoRA Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_USE_LORA` | `true` | Enable LoRA |
| `MODEL_TEA_LORA_R` | `8` | LoRA rank |
| `MODEL_TEA_LORA_ALPHA` | `16` | LoRA alpha |
| `MODEL_TEA_LORA_DROPOUT` | `0.05` | LoRA dropout |
| `MODEL_TEA_LORA_TARGET_MODULES` | `c_attn,c_proj` | Target modules (comma-separated) |

### Quality Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_PERPLEXITY_THRESHOLD` | `50.0` | Perplexity threshold |
| `MODEL_TEA_ITERATIVE_PERPLEXITY_THRESHOLD` | `20.0` | Perplexity (iterative) |
| `MODEL_TEA_TARGET_PERPLEXITY` | `15.0` | Target perplexity |
| `MODEL_TEA_QUALITY_THRESHOLD` | `0.8` | Quality threshold |
| `MODEL_TEA_PERPLEXITY_IMPROVEMENT_THRESHOLD` | `2.0` | Min improvement |
| `MODEL_TEA_QUALITY_DEGRADATION_THRESHOLD` | `0.01` | Max degradation |

### Adaptive Training

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_ADAPTIVE_TRAINING` | `true` | Enable adaptive training |
| `MODEL_TEA_EARLY_STOPPING_PATIENCE` | `3` | Early stopping patience |
| `MODEL_TEA_OVERFITTING_DETECTION_WINDOW` | `3` | Overfitting window |
| `MODEL_TEA_VALIDATION_LOSS_PATIENCE` | `3` | Validation patience |
| `MODEL_TEA_MIN_ITERATIONS` | `5` | Minimum iterations |
| `MODEL_TEA_MAX_ITERATIONS` | `50` | Maximum iterations |

### Generation Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_DEFAULT_MAX_LENGTH` | `250` | Default max generation length |
| `MODEL_TEA_DEFAULT_TEMPERATURE` | `0.5` | Default temperature |
| `MODEL_TEA_VALIDATION_TEMPERATURE` | `0.8` | Validation temperature |
| `MODEL_TEA_MAX_REPETITION_PENALTY` | `1.1` | Max repetition penalty |
| `MODEL_TEA_TEMPERATURE_RANGE_MIN` | `0.7` | Min temperature range |
| `MODEL_TEA_TEMPERATURE_RANGE_MAX` | `0.9` | Max temperature range |

### Path Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_NOVELS_DIR` | `novels` | Novels directory |
| `MODEL_TEA_MODELS_FILE` | `models.json` | Models metadata file |
| `MODEL_TEA_NOVELS_FILE` | `novels.json` | Novels metadata file |
| `MODEL_TEA_OUTPUT_DIR` | `iterative_models` | Output directory |
| `MODEL_TEA_SAVE_CHECKPOINTS` | `true` | Save checkpoints |

### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_API_HOST` | `0.0.0.0` | API server host |
| `MODEL_TEA_API_PORT` | `8000` | API server port |
| `MODEL_TEA_API_CORS_ORIGINS` | `*` | CORS origins (comma-separated) |

**Production Security:** Always specify exact CORS origins in production:
```bash
MODEL_TEA_API_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### Logging Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TEA_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |

## Python API

Access settings programmatically:

```python
from model_tea.config.settings import settings

print(f"Base model: {settings.base_model}")
print(f"API port: {settings.api_port}")
print(f"Batch size: {settings.batch_size}")

# Path objects
novels_path = settings.novels_path
output_path = settings.output_path
```

## Legacy Config Classes

For backward compatibility, old config classes still work:

```python
from model_tea.config import ModelTeaConfig
from model_tea.trainers.iterative.config import IterativeConfig

# These now load values from centralized settings
basic_config = ModelTeaConfig()
iterative_config = IterativeConfig()
```

## Examples

### High-Performance Training

```bash
MODEL_TEA_BATCH_SIZE=16 \
MODEL_TEA_LEARNING_RATE_START=3e-5 \
MODEL_TEA_MAX_ITERATIONS=100 \
model-tea train train-model large_model
```

### Low-Resource Training

```bash
MODEL_TEA_BATCH_SIZE=4 \
MODEL_TEA_GRADIENT_ACCUMULATION_STEPS=4 \
MODEL_TEA_ITERATIVE_MAX_SEQ_LENGTH=256 \
model-tea train direct small_novel
```

### Production API

```bash
MODEL_TEA_API_HOST=127.0.0.1 \
MODEL_TEA_API_PORT=8080 \
MODEL_TEA_API_CORS_ORIGINS=https://myapp.com \
MODEL_TEA_LOG_LEVEL=WARNING \
python -m model_tea.api.server
```

### Custom Model

```bash
MODEL_TEA_BASE_MODEL=gpt2-medium \
MODEL_TEA_MAX_SEQ_LENGTH=2048 \
model-tea train train-model custom
```
