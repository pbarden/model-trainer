# API Endpoints

## Starting the Server

```bash
uvicorn model_tea.api.server:app --reload
```

API available at: http://localhost:8000
Documentation: http://localhost:8000/docs

## Training Endpoints

### POST /api/training/model
Train a model from model_mapping.json

Request:
```json
{
  "model_key": "frankenstein",
  "max_iterations": 15
}
```

Response:
```json
{
  "success": true,
  "message": "Training completed successfully",
  "results": {...}
}
```

### POST /api/training/direct
Train a novel directly (development)

Request:
```json
{
  "novel_name": "frankenstein",
  "max_iterations": 10,
  "batch_size": 8,
  "learning_rate": 0.00002
}
```

Response:
```json
{
  "success": true,
  "message": "Training completed successfully",
  "training_time": 3600.5,
  "iterations": 10,
  "final_quality": 0.85,
  "results": {...}
}
```

### GET /api/training/status/{job_id}
Get training status

Response:
```json
{
  "job_id": "12345",
  "status": "running",
  "progress": 0.5,
  "message": "Training in progress"
}
```

### GET /api/training/novels
List available novels

Response:
```json
{
  "novels": ["frankenstein", "dracula", ...],
  "total": 345
}
```

## Chat Endpoints

### POST /api/chat/generate
Generate text from model

Request:
```json
{
  "model_name": "frankenstein/final",
  "prompt": "It was a dark and stormy night",
  "max_length": 200,
  "temperature": 0.7,
  "top_p": 0.85,
  "top_k": 30,
  "repetition_penalty": 1.4
}
```

Response:
```json
{
  "response": "Generated text here...",
  "model_name": "frankenstein/final",
  "prompt": "It was a dark and stormy night"
}
```

## Model Endpoints

### GET /api/models
List all models

Response:
```json
{
  "models": ["frankenstein", "dracula", ...],
  "total": 102
}
```

### GET /api/models/{model_name}
Get model information

Response:
```json
{
  "name": "frankenstein/final",
  "path": "/path/to/model",
  "exists": true,
  "size_mb": 512.5,
  "model_type": "causal_lm",
  "vocab_size": 50257
}
```

## Health Endpoints

### GET /
API root

Response:
```json
{
  "message": "Model Tea API",
  "version": "2.0.0"
}
```

### GET /health
Health check

Response:
```json
{
  "status": "healthy"
}
```
