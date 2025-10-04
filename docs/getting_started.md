# Getting Started

## Installation

```bash
cd model-trainer
pip install -e .
```

## Quick Start

### 1. Generate Model Mapping

Create model_mapping.json from your novels:

```bash
python src/model_tea/utils/generate_mapping.py
```

This scans `novels/` and creates one model per novel.

### 2. Train a Model

```bash
model-tea train train-model frankenstein
```

### 3. Chat with Model

```bash
model-tea chat interactive frankenstein/final
```

## Directory Structure

```
model-trainer/
├── novels/              # Place .txt files here
├── iterative_models/    # Trained models output here
├── model_mapping.json   # Model definitions
└── src/model_tea/       # Source code
```

## First Training Run

1. Add a novel to `novels/frankenstein/frankenstein.txt`
2. Generate mapping: `python src/model_tea/utils/generate_mapping.py`
3. Train: `model-tea train train-model frankenstein`
4. Chat: `model-tea chat interactive frankenstein/final`

## Development Mode

For testing single novels without model_mapping.json:

```bash
model-tea train direct frankenstein --max-iterations 5
```

## Next Steps

- [CLI Commands](cli/commands.md) - Full command reference
- [Model Mapping Schema](schema/model_mapping.md) - Creating custom model combinations
- [Configuration](configuration.md) - Customizing training parameters
- [API Endpoints](api/endpoints.md) - REST API usage
- [Architecture](architecture/overview.md) - Understanding the system
