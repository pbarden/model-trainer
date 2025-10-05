# Model Tea

Professional CPU-optimized language model training system with progressive difficulty training.

## Features

- **Iterative Training:** Progressive difficulty training for optimal learning
- **LoRA Support:** Efficient training with Low-Rank Adaptation
- **Adaptive Training:** Automatic early stopping and overfitting detection
- **Centralized Configuration:** Environment variable support for all settings
- **REST API:** FastAPI-based API for programmatic access
- **CLI Interface:** Rich terminal interface for training and inference

## Installation

```bash
pip install -e .
```

## Quick Start

### Train a Model

```bash
model-tea train train-model frankenstein
model-tea train direct call_of_cthulhu
```

### Chat with Model

```bash
model-tea chat interactive frankenstein/final
```

### Start API Server

```bash
python -m model_tea.api.server
```

## Configuration

Model Tea uses environment variables for configuration. All settings are optional with sensible defaults.

### Quick Configuration

```bash
MODEL_TEA_BASE_MODEL=gpt2 model-tea train train-model horror
MODEL_TEA_API_PORT=9000 python -m model_tea.api.server
```

### Using .env File

```bash
cp .env.example .env
```

Edit `.env`:

```bash
MODEL_TEA_BASE_MODEL=distilgpt2
MODEL_TEA_BATCH_SIZE=8
MODEL_TEA_DEFAULT_TEMPERATURE=0.7
```

See [Configuration Guide](docs/configuration.md) for all 50+ configurable parameters.

## CLI Usage

### Training

```bash
model-tea train train-model <model_key>
model-tea train direct <novel_name>
model-tea train list
```

### Chat

```bash
model-tea chat interactive <model_name>
model-tea chat generate <model_name> --prompt "Once upon a time"
```

### Models

```bash
model-tea models list
model-tea models info <model_name>
model-tea models delete <model_name>
```

### System

```bash
model-tea status
model-tea list
```

See [CLI Commands](docs/cli/commands.md) for complete reference.

## API Usage

### Start Server

```bash
python -m model_tea.api.server
```

Or with custom configuration:

```bash
MODEL_TEA_API_HOST=0.0.0.0 MODEL_TEA_API_PORT=8000 python -m model_tea.api.server
```

### Endpoints

```
POST   /api/training/model
POST   /api/training/direct
GET    /api/training/status/{job_id}
GET    /api/training/novels
POST   /api/chat/generate
GET    /api/models
GET    /api/models/{model_name}
GET    /
GET    /health
```

### Example Request

```python
import requests

response = requests.post("http://localhost:8000/api/chat/generate", json={
    "model_name": "frankenstein/final",
    "prompt": "It was a dark and stormy night",
    "max_length": 200,
    "temperature": 0.7
})

print(response.json()["response"])
```

See [API Documentation](docs/api/endpoints.md) for details.

## Python API

```python
from model_tea.services import TrainingService, ChatService
from model_tea.config.settings import settings

training = TrainingService()
results = training.train_model("frankenstein")

chat = ChatService()
chat.load_model("frankenstein/final")
response = chat.generate("Once upon a time", temperature=0.7)
print(response)
```

## Project Structure

```
model-trainer/
├── src/model_tea/
│   ├── trainers/           Iterative and model-based training
│   ├── core/               Core validation logic
│   ├── services/           Business logic layer
│   ├── api/                FastAPI REST API
│   ├── cli/                Click CLI interface
│   ├── config/             Centralized configuration
│   └── utils/              Utility functions
├── novels/                 Training corpus
├── iterative_models/       Trained models output
├── models.json             Model definitions
├── .env.example            Configuration template
└── docs/                   Documentation
```

## Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Configuration](docs/configuration.md)
- [CLI Commands](docs/cli/commands.md)
- [API Endpoints](docs/api/endpoints.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Model Mapping Schema](docs/schema/model_mapping.md)

## Development

### Run Tests

```bash
pytest tests/
```

### Format Code

```bash
black src/
isort src/
```

### Type Check

```bash
mypy src/
```

## License

MIT

## Copyright

ChaiQ LLC
