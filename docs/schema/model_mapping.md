# Model Mapping Schema

## Files

The model mapping system uses two separate JSON files:

- `novels.json` - Novel metadata (relational data)
- `models.json` - Model definitions (references to novels)

## novels.json Structure

```json
{
  "novel_key": {
    "original_name": "Novel Title",
    "directory_name": "folder_name",
    "word_count": 50000
  }
}
```

### Example novels.json

```json
{
  "frankenstein": {
    "original_name": "Frankenstein",
    "directory_name": "frankenstein",
    "word_count": 78000
  },
  "dracula": {
    "original_name": "Dracula",
    "directory_name": "dracula",
    "word_count": 164000
  }
}
```

## models.json Structure

```json
{
  "model_key": {
    "novels": ["novel_key"]
  }
}
```

### Example models.json

```json
{
  "frankenstein": {
    "novels": ["frankenstein"]
  },
  "dracula": {
    "novels": ["dracula"]
  },
  "gothic_horror": {
    "novels": ["frankenstein", "dracula"]
  }
}
```

## Field Requirements

### novels.json
- Key: novel identifier (string)
- `original_name` (string, required) - display name
- `directory_name` (string, required) - folder in novels/
- `word_count` (number, required) - total words

### models.json
- Key: model identifier (string)
- `novels` (array of strings, required) - references to novel keys from novels.json

## Code Usage

```python
# Load novels
with open("novels.json") as f:
    novels = json.load(f)

# Load models
with open("models.json") as f:
    models = json.load(f)

# Get novel keys for a model
novel_keys = models["model_key"]["novels"]

# Get novel metadata
for novel_key in novel_keys:
    novel = novels[novel_key]
    directory = novel["directory_name"]
    name = novel["original_name"]
    words = novel["word_count"]
```
