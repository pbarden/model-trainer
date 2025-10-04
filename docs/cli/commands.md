# CLI Commands

## Training Commands

### train train-model
Train a model from model_mapping.json

```bash
model-tea train train-model <model_key>
model-tea train train-model frankenstein
model-tea train train-model horror_collection --max-iterations 20
```

Options:
- `--max-iterations` - Maximum training iterations

### train direct
Train a novel directly from novels/ directory (development/testing)

```bash
model-tea train direct <novel_name>
model-tea train direct frankenstein --max-iterations 10 --batch-size 8
```

Options:
- `--max-iterations` - Maximum training iterations
- `--batch-size` - Training batch size
- `--learning-rate` - Starting learning rate

### train list
List available novels

```bash
model-tea train list
```

## Chat Commands

### chat interactive
Start interactive chat session

```bash
model-tea chat interactive <model_name>
model-tea chat interactive frankenstein/final
model-tea chat interactive frankenstein/final --max-length 300 --temperature 0.7
```

Options:
- `--max-length` - Maximum generation length
- `--temperature` - Sampling temperature

### chat generate
Generate single response

```bash
model-tea chat generate <model_name> --prompt "Once upon a time"
```

Options:
- `--prompt` - Input prompt (required)
- `--max-length` - Maximum generation length
- `--temperature` - Sampling temperature

## Model Commands

### models list
List all trained models

```bash
model-tea models list
model-tea models list --all
```

### models info
Get model information

```bash
model-tea models info <model_name>
model-tea models info frankenstein/final
```

### models delete
Delete a model

```bash
model-tea models delete <model_name>
model-tea models delete old_model --force
```

Options:
- `--force` - Skip confirmation
