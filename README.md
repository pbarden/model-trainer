# Model Tea - Iterative Novel Training System

A comprehensive AI model training framework designed for iterative novel training with episodic memory integration. This system trains language models on literary works using progressive difficulty and memory-based learning techniques.

## 🚀 Features

- **Iterative Training**: Progressive difficulty training across multiple iterations
- **Episodic Memory System**: Human-like memory storage and retrieval for enhanced context
- **Quality Validation**: Real-time generation quality assessment and perplexity monitoring
- **Model Serving**: Production-ready model deployment with health monitoring
- **ML Pipeline Framework**: Extensible pipeline system for model development workflows
- **Comprehensive Evaluation**: Built-in metrics and cross-validation capabilities

## 📋 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers

### Installation

```bash
git clone <repository-url>
cd model-trainer
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train a model on a novel
python iterative_novel_trainer.py --novel call_of_cthulhu

# List available novels
python iterative_novel_trainer.py --list

# Generate samples from a trained model
python iterative_novel_trainer.py --novel frankenstein --generate
```

## 🏗️ Architecture

The system consists of several key components:

- **Core Training System** (`iterative_novel_trainer.py`) - Main training orchestrator
- **Episodic Memory** (`episodic_memory_system.py`) - Memory storage and retrieval
- **Model Tea Framework** (`model_tea/`) - ML pipeline and serving infrastructure
- **Quality Validation** (`quality_validator.py`) - Generation quality assessment

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Installation Guide](docs/guides/installation.md)
- [Training Guide](docs/guides/training.md)
- [API Reference](docs/api/README.md)
- [Module Documentation](docs/modules/README.md)
- [Feature Documentation](docs/features/README.md)

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_iterative_trainer.py -v
python -m pytest tests/test_episodic_memory.py -v
python -m pytest tests/test_pipeline.py -v
```

Current test coverage: **137 tests** with **91% pass rate**

## 📊 Performance

- Supports novels from 1,000 to 100,000+ words
- GPU acceleration available for training
- Memory-efficient chunking for large texts
- Configurable batch sizes and learning rates

## 🛠️ Configuration

Key configuration options:

```python
from iterative_novel_trainer import IterativeConfig

config = IterativeConfig(
    base_model="gpt2",
    iterations_per_novel=12,
    max_steps_per_iteration=8,
    learning_rate_start=5e-5,
    chunk_size=200
)
```

## 📖 Example Workflows

### Train on a Classic Novel

```python
from iterative_novel_trainer import IterativeTrainer, IterativeConfig

config = IterativeConfig(iterations_per_novel=10)
trainer = IterativeTrainer(config)
results = trainer.train_novel("alice_in_wonderland")
```

### Generate with Memory Context

```python
from episodic_memory_system import EpisodicMemorySystem

memory_system = EpisodicMemorySystem()
memories, stats = memory_system.activate_memories("model_name", "prompt")
```

### Deploy Model for Serving

```python
from model_tea.deployment.serving import ModelServer, ServingConfig

config = ServingConfig(model_name="trained_model", port=8080)
server = ModelServer(config)
server.load_model(model)
server.start()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Model Tea Framework

This project uses the Model Tea framework for ML pipeline management, model serving, and evaluation. Model Tea provides:

- Extensible pipeline stages
- Model deployment infrastructure
- Comprehensive evaluation metrics
- Production monitoring capabilities

## 💡 Research & Development

This system is designed for:
- Literary AI research
- Style transfer experiments
- Memory-augmented language modeling
- Progressive training methodologies

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review existing test cases for usage examples

---

**Model Tea - Iterative Novel Training System**
*Advanced AI model training with episodic memory integration*