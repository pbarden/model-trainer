# Lovecraft Mythos Extended Training Guide

## Overview

The Lovecraft Mythos model includes 8 H.P. Lovecraft stories:
1. At the Mountains of Madness (40,020 words)
2. Call of Cthulhu (11,945 words)
3. The Case of Charles Dexter Ward (44,554 words)
4. Dunwich Horror (17,357 words)
5. Horror at Red Hook (8,500 words)
6. Shadow over Innsmouth (22,000 words)
7. The Shunned House (10,748 words)
8. Through the Gates of the Silver Key (14,258 words)

**Total: 175,302 words**

## Training Methods

### Method 1: Standard Training (12 iterations)

```bash
python combined_model_trainer.py --model lovecraft_mythos
```

This uses default settings:
- 12 iterations
- Target perplexity: 12.0
- Adaptive stopping enabled

### Method 2: Extended Training (Custom iterations)

```bash
python train_lovecraft_extended.py --iterations 30
```

Available options:
- `--iterations N` - Maximum training iterations (default: 30)
- `--target-perplexity P` - Target perplexity to achieve (default: 10.0)
- `--min-iterations N` - Minimum iterations before stopping (default: 12)
- `--patience N` - Early stopping patience (default: 3)
- `--force` - Force retrain even if model exists

### Method 3: Continue Training from Existing Model

To continue training an existing model:

```bash
# First, back up your existing model
cp -r iterative_models/lovecraft_mythos/final iterative_models/lovecraft_mythos/final_backup

# Then retrain with more iterations
python train_lovecraft_extended.py --iterations 50 --force
```

## Recommended Settings for Different Goals

### High Quality Output (Recommended)
```bash
python train_lovecraft_extended.py --iterations 30 --target-perplexity 8.0 --patience 5
```
- 30 iterations max
- Target perplexity: 8.0
- Higher patience to avoid premature stopping

### Balanced Training
```bash
python train_lovecraft_extended.py --iterations 25 --target-perplexity 10.0
```
- 25 iterations max
- Target perplexity: 10.0
- Default patience (3)

### Quick Training
```bash
python combined_model_trainer.py --model lovecraft_mythos
```
- 12 iterations (default)
- Faster training
- Good baseline quality

### Maximum Quality (Long training)
```bash
python train_lovecraft_extended.py --iterations 50 --target-perplexity 6.0 --patience 7 --min-iterations 20
```
- 50 iterations max
- Very low target perplexity: 6.0
- High patience to continue fine-tuning
- Minimum 20 iterations before stopping

## Understanding the Metrics

### Perplexity
- **Lower is better**
- Measures how well the model predicts text
- Target: 8.0-12.0 for good quality
- Below 6.0 may indicate overfitting

### Quality Score
- **Higher is better** (0.0 to 1.0)
- Measures text coherence and diversity
- Target: 0.75-0.85 for good quality
- Above 0.85 is excellent

### Training Stops When:
1. Target perplexity is reached
2. No improvement for `patience` iterations
3. Maximum iterations reached
4. Overfitting is detected

## Monitoring Training Progress

Training outputs show:
```
Iteration X/Y completed:
  Time: 45.2s
  Perplexity: 10.23 (improving)
  Quality: 0.812 (stable)
  Status: converging
```

Watch for:
- **Decreasing perplexity** = Good
- **Increasing quality** = Good
- **"overfitting_risk"** = May need to stop soon
- **"stop_plateau"** = No more improvement

## After Training

### Test the Model
```bash
python interactive_chat.py
```
Select the `lovecraft_mythos/final` model and test with prompts like:
- "An ancient evil awakens beneath the ocean"
- "The professor discovered forbidden texts in the library"

### Compare Iterations
Models are saved at:
- `iterative_models/lovecraft_mythos/iteration_0/` (first)
- `iterative_models/lovecraft_mythos/iteration_11/` (12th)
- `iterative_models/lovecraft_mythos/final/` (best)

## Troubleshooting

### Training stops too early
```bash
# Increase patience and minimum iterations
python train_lovecraft_extended.py --iterations 40 --patience 6 --min-iterations 20
```

### Perplexity not improving
- May already be at optimal level
- Try adjusting learning rate in the script
- Check for overfitting (quality score dropping)

### Model generates incoherent text
- Use higher generation constraints in chat:
  - Temperature: 0.5-0.6
  - Repetition penalty: 1.3-1.5
  - Top-k: 30-40

## Files Created During Training

- `iterative_models/lovecraft_mythos/training_results.json` - Full training metrics
- `iterative_models/lovecraft_mythos/iteration_N/` - Checkpoint models
- `iterative_models/lovecraft_mythos/final/` - Final best model
- `testing_outputs/lovecraft_mythos_*.txt` - Evaluation reports

## Best Practices

1. **Start with standard training** (12 iterations)
2. **Test the model** with interactive chat
3. **If quality is insufficient**, use extended training
4. **Monitor metrics** during training
5. **Back up good models** before retraining
6. **Use appropriate generation settings** in chat
