# Configuration

## IterativeConfig

Used for direct training and internally by ModelTrainer.

```python
from model_tea.trainers.iterative import IterativeConfig

config = IterativeConfig(
    base_model="distilgpt2",
    max_seq_length=512,
    batch_size=4,
    max_iterations=50,
    min_iterations=5,
    max_steps_per_iteration=200,
    learning_rate_start=2e-5,
    learning_rate_end=5e-6,
    chunk_size=150,
    chunk_overlap=20,
    validation_split=0.1,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
```

### Parameters

#### Model Settings
- `base_model` - Base model to fine-tune (default: "distilgpt2")
- `max_seq_length` - Maximum sequence length (default: 512)

#### Training Settings
- `batch_size` - Training batch size (default: 4)
- `max_iterations` - Maximum training iterations (default: 50)
- `min_iterations` - Minimum iterations before early stopping (default: 5)
- `max_steps_per_iteration` - Steps per iteration (default: 200)
- `learning_rate_start` - Starting learning rate (default: 2e-5)
- `learning_rate_end` - Ending learning rate (default: 5e-6)

#### Data Processing
- `chunk_size` - Target words per chunk (default: 150)
- `chunk_overlap` - Overlapping words between chunks (default: 20)
- `validation_split` - Validation set ratio (default: 0.1)

#### LoRA Settings
- `use_lora` - Use LoRA for efficient training (default: True)
- `lora_r` - LoRA rank (default: 8)
- `lora_alpha` - LoRA alpha (default: 16)
- `lora_dropout` - LoRA dropout (default: 0.05)

## ModelConfig

Used for model training from model_mapping.json.

```python
from model_tea.trainers.model import ModelConfig

config = ModelConfig(
    combine_novels_method="concatenate",
    novel_separator="\n\n=== NEW NOVEL ===\n\n",
    min_novels_required=1,
    max_combined_size=2000000,
    max_iterations=14,
    learning_rate_start=5e-5,
    learning_rate_end=5e-6
)
```

### Parameters

#### Novel Combination
- `combine_novels_method` - How to combine novels (default: "concatenate")
- `novel_separator` - Separator between novels (default: "\n\n=== NEW NOVEL ===\n\n")
- `min_novels_required` - Minimum novels per model (default: 1)
- `max_combined_size` - Maximum combined word count (default: 2000000)

#### Training Overrides
- `max_iterations` - Override for max iterations (default: 14)
- `learning_rate_start` - Override for starting LR (default: 5e-5)
- `learning_rate_end` - Override for ending LR (default: 5e-6)

Note: Most training parameters are computed adaptively based on word count.

## ValidationConfig

Quality validation settings.

```python
from model_tea.core.validator import ValidationConfig

config = ValidationConfig(
    perplexity_threshold=50.0,
    quality_threshold=0.8
)
```

### Parameters
- `perplexity_threshold` - Maximum acceptable perplexity (default: 50.0)
- `quality_threshold` - Minimum quality score (default: 0.8)
