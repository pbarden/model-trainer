# Scripts

Standalone executable scripts for Model Tea utilities.

## Available Scripts

### generate_mapping.py

Generates `novels.json` by scanning the `novels/` directory.

```bash
python scripts/generate_mapping.py
```

### clean_mapping.py

Cleans `models.json` by removing training parameters and keeping only relational data (which novels belong to which models).

```bash
python scripts/clean_mapping.py
```

## Usage

All scripts are executable and can be run directly:

```bash
./scripts/generate_mapping.py
./scripts/clean_mapping.py
```

Or via Python:

```bash
python scripts/generate_mapping.py
python scripts/clean_mapping.py
```
