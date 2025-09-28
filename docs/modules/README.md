# Module Documentation

This directory contains detailed documentation for each module in the Model Tea Iterative Novel Training System.

## Core Modules

### [Iterative Novel Trainer](iterative_trainer.md)
The main training orchestrator that handles progressive difficulty training across multiple iterations.

### [Episodic Memory System](episodic_memory.md)
Human-like memory storage and retrieval system for enhanced contextual understanding.

### [Quality Validator](quality_validator.md)
Real-time generation quality assessment and perplexity monitoring system.

## Model Tea Framework

### [Core Pipeline](model_tea_core.md)
ML pipeline management and orchestration framework.

### [Evaluation System](model_tea_evaluation.md)
Comprehensive model evaluation and metrics calculation.

### [Deployment & Serving](model_tea_serving.md)
Production-ready model deployment and serving infrastructure.

## Utility Modules

### [Model Tea Utils](model_tea_utils.md)
Shared utilities and helper functions for the training system.

## Module Relationships

```
iterative_novel_trainer.py
├── episodic_memory_system.py
├── quality_validator.py
├── model_tea_utils.py
└── model_tea/
    ├── core/
    │   ├── pipeline.py
    │   └── evaluation.py
    └── deployment/
        └── serving.py
```

## Quick Navigation

- [API Reference](../api/README.md)
- [Feature Documentation](../features/README.md)
- [Installation Guide](../guides/installation.md)
- [Training Guide](../guides/training.md)