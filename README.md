# Model Tea

Professional CPU-optimized language model training system with progressive difficulty training.

## Installation

```bash
pip install -e .
```

## Documentation

### Getting Started
- [Getting Started Guide](docs/getting_started.md) - Quick start and first training run

### Core Documentation
- [CLI Commands](docs/cli/commands.md) - Complete CLI reference
- [API Endpoints](docs/api/endpoints.md) - REST API documentation
- [Configuration](docs/configuration.md) - Configuration options
- [Architecture Overview](docs/architecture/overview.md) - System design

### Schemas
- [Model Mapping Schema](docs/schema/model_mapping.md) - model_mapping.json format

## CLI Usage

### Training

#### Train from model_mapping.json (Primary Method)
```bash
# Train any model (single or multi-novel) from model_mapping.json
model-tea train train-model <model_key>
model-tea train train-model bc_sprinkles
model-tea train train-model xt_orange_soda --max-iterations 10
```

#### Direct Training (Development/Testing)
```bash
# Train a novel directly from novels/ directory
model-tea train direct <novel_name>
model-tea train direct call_of_cthulhu --max-iterations 10 --batch-size 8
```

#### List Available Resources
```bash
model-tea train list
```

### Chat

```bash
model-tea chat interactive <model_name>
model-tea chat interactive frankenstein/final

model-tea chat generate <model_name> --prompt "Once upon a time"
```

### Models

```bash
model-tea models list
model-tea models list --all

model-tea models info frankenstein/final
model-tea models delete old_model --force
```

## API Usage

### Start Server

```bash
uvicorn model_tea.api.server:app --reload
```

Or use the CLI:

```bash
python -m model_tea.api.server
```

### API Endpoints

#### Training

```
POST   /api/training/model     - Train model from model_mapping.json
POST   /api/training/direct    - Train novel directly (dev/testing)
GET    /api/training/status/{job_id}
GET    /api/training/novels
```

#### Chat

```
POST   /api/chat/generate
```

#### Models

```
GET    /api/models
GET    /api/models/{model_name}
```

#### Health

```
GET    /
GET    /health
```

### Example API Request

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

## Python API

```python
from model_tea import TrainingService, ChatService, ModelConfig

# Train model from model_mapping.json
training = TrainingService()
results = training.train_model("xt_orange_soda")

# Or train directly for development
from model_tea.trainers.iterative import IterativeConfig
config = IterativeConfig(max_iterations=10)
results = training.train_direct("frankenstein", config)

# Generate text
chat = ChatService()
chat.load_model("frankenstein/final")
response = chat.generate("Once upon a time")
print(response)
```

## Project Structure

```
model-trainer/
├── src/model_tea/
│   ├── trainers/
│   │   ├── iterative/     - Direct novel training (dev/testing)
│   │   └── model/         - Model training from model_mapping.json
│   ├── core/              - Core validation logic
│   ├── services/          - Business logic layer
│   ├── api/               - FastAPI REST API
│   ├── cli/               - Click CLI interface
│   ├── config/            - Configuration management
│   └── utils/             - Utility functions
├── novels/                - Training corpus
├── iterative_models/      - Trained models output
├── model_mapping.json     - Model definitions
└── tests/                 - Test suite
```

## Configuration

Training configurations can be customized:

```python
# For model_mapping.json-based training
from model_tea.trainers.model import ModelConfig

config = ModelConfig(
    max_iterations=15,
    combine_novels_method="concatenate",
    min_novels_required=1
)

# For direct training (dev/testing)
from model_tea.trainers.iterative import IterativeConfig

config = IterativeConfig(
    max_iterations=15,
    batch_size=8,
    learning_rate_start=2e-5,
    use_lora=True
)
```

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
