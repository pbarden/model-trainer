# Model Tea

Professional CPU-optimized language model training system with progressive difficulty training.

## Installation

```bash
cd model-tea-pro
pip install -e .
```

## CLI Usage

### Training

```bash
model-tea train novel <novel_name>
model-tea train novel call_of_cthulhu --max-iterations 10

model-tea train combined <model_name>
model-tea train combined bc_sprinkles

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
POST   /api/training/novel
POST   /api/training/combined
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
from model_tea import TrainingService, ChatService, IterativeConfig

training = TrainingService()
results = training.train_novel("frankenstein")

chat = ChatService()
chat.load_model("frankenstein/final")
response = chat.generate("Once upon a time")
print(response)
```

## Project Structure

```
model-tea-pro/
├── src/model_tea/
│   ├── trainers/
│   │   ├── iterative/     - Iterative novel training
│   │   └── combined/      - Combined model training
│   ├── core/              - Core validation logic
│   ├── services/          - Business logic layer
│   ├── api/               - FastAPI REST API
│   ├── cli/               - Click CLI interface
│   ├── config/            - Configuration management
│   └── utils/             - Utility functions
├── data/
│   ├── novels/            - Training corpus
│   ├── models/            - Trained models
│   └── config/            - Configuration files
└── tests/                 - Test suite
```

## Configuration

Training configurations can be customized:

```python
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
