# Architecture Overview

## System Components

### Trainers
- `trainers/iterative/` - Direct single-novel training
- `trainers/model/` - Model training from model_mapping.json (1+ novels)

### Services
- `services/training.py` - Training orchestration
- `services/chat.py` - Inference orchestration

### API
- `api/routes/training.py` - Training endpoints
- `api/routes/chat.py` - Chat endpoints
- `api/routes/models.py` - Model management

### CLI
- `cli/train_cmd.py` - Training commands
- `cli/chat_cmd.py` - Chat commands
- `cli/models_cmd.py` - Model management

### Core
- `core/validator.py` - Quality validation and perplexity checks

### Utils
- `utils/file_utils.py` - File operations
- `utils/text_processing.py` - Text processing
- `utils/training_utils.py` - Training helpers
- `utils/errors.py` - Error handling

## Training Flow

### Model Training (Primary)
1. User calls `model-tea train train-model <model_key>`
2. CLI -> TrainingService.train_model()
3. ModelTrainer loads model_mapping.json
4. Gets novels list for model_key
5. Calculates total_words from novels
6. Sets adaptive parameters based on word count
7. Combines novel contents
8. Calls IterativeTrainer._train_with_content()
9. Iterative training with progressive difficulty
10. Saves final model

### Direct Training (Development)
1. User calls `model-tea train direct <novel_name>`
2. CLI -> TrainingService.train_direct()
3. IterativeTrainer loads novel from novels/ directory
4. Sets min_iterations based on word count
5. Iterative training
6. Saves final model

## Adaptive Training

Training parameters are computed based on total word count:

| Word Count | Steps/Iter | Min Iter | LR Start | LR End | Category |
|-----------|------------|----------|----------|---------|----------|
| < 80k     | 125        | 5        | 3e-5     | 8e-6    | tiny     |
| 80k-120k  | 150        | 8        | 2.5e-5   | 6e-6    | small    |
| 120k-260k | 200        | 10       | 2e-5     | 5e-6    | medium   |
| > 260k    | 300        | 12       | 1.5e-5   | 3e-6    | xlarge   |

## Data Flow

```
model_mapping.json -> ModelTrainer -> IterativeConfig -> IterativeTrainer
                                   -> novels content -> training
```
