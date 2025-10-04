# Utilities Plan

## Overview

Utilities are standalone tools that perform specific operational tasks for the Model Tea system.

## Current State

### Existing Utilities

**src/model_tea/utils/**
- `file_utils.py` - FileSystemUtils (directory creation, JSON I/O)
- `text_processing.py` - TextProcessingUtils, QualityMetrics (chunking, text analysis)
- `training_utils.py` - TrainingUtils (LR calculation, time formatting)
- `errors.py` - Error handling classes
- `clean_mapping.py` - Remove invalid fields from model_mapping.json
- `generate_mapping.py` - Create model_mapping.json from novels directory

## Required Utilities

### Novel Management (`src/model_tea/utils/novel/`)

#### import_novel.py
```bash
model-tea utils import-novel <file_path> [--name <name>]
```
- Copy novel file to novels/ directory
- Create proper directory structure
- Detect encoding
- Initial word count

#### preprocess_novel.py
```bash
model-tea utils preprocess <novel_name>
```
- Remove formatting artifacts
- Fix encoding issues
- Normalize whitespace
- Remove headers/footers (Project Gutenberg boilerplate)
- Save preprocessed version

#### clean_novel.py
```bash
model-tea utils clean <novel_name>
```
- Remove special characters
- Fix broken sentences
- Normalize punctuation
- Remove excessive line breaks

#### analyze_novel.py
```bash
model-tea utils analyze <novel_name>
```
- Word count
- Character count
- Sentence count
- Readability metrics
- Vocabulary size
- Save analysis.json

#### tag_novel.py
```bash
model-tea utils tag <novel_name> --tags genre,author,year
```
- Add metadata tags
- Genre classification
- Author attribution
- Publication year
- Custom tags

### Model Management (`src/model_tea/utils/model/`)

#### create_model.py
```bash
model-tea utils create-model <model_key> --novels novel1,novel2,novel3
```
- Create new model entry in model_mapping.json
- Validate novels exist
- Calculate total word count
- Interactive mode for selection

#### edit_model.py
```bash
model-tea utils edit-model <model_key> --add novel --remove novel
```
- Add novels to existing model
- Remove novels from model
- Update model metadata

#### delete_model.py
```bash
model-tea utils delete-model <model_key>
```
- Remove model from model_mapping.json
- Optionally delete trained files

#### list_models.py
```bash
model-tea utils list-models [--trained] [--untrained]
```
- List all models from mapping
- Show trained status
- Show novel count and word count

### Mapping Management (`src/model_tea/utils/mapping/`)

Move existing tools here and add new ones:

#### validate_mapping.py
```bash
model-tea utils validate-mapping
```
- Check model_mapping.json syntax
- Verify all directory_names exist in novels/
- Check for duplicate novels
- Validate word counts
- Report errors

#### rebuild_mapping.py
```bash
model-tea utils rebuild-mapping [--backup]
```
- Scan novels/ directory
- Regenerate word counts
- Preserve existing model definitions
- Update metadata

#### merge_mappings.py
```bash
model-tea utils merge-mappings mapping1.json mapping2.json --output merged.json
```
- Combine multiple mapping files
- Resolve conflicts
- Deduplicate models

### Novel Discovery (`src/model_tea/utils/discovery/`)

#### scan_novels.py
```bash
model-tea utils scan-novels
```
- Find all novels in novels/ directory
- Report which are in mapping
- Report which are missing from mapping
- Suggest creating models

#### find_unmapped.py
```bash
model-tea utils find-unmapped
```
- List novels not in any model
- Suggest model combinations

#### stats.py
```bash
model-tea utils stats
```
- Total novels count
- Total models count
- Word count distribution
- Training status summary

### System Utilities (`src/model_tea/utils/system/`)

#### setup.py
```bash
model-tea utils setup
```
- Initialize directory structure
- Create empty model_mapping.json
- Verify dependencies
- System health check

#### backup.py
```bash
model-tea utils backup [--include-models]
```
- Backup model_mapping.json
- Backup novels/ metadata
- Optionally backup trained models

#### restore.py
```bash
model-tea utils restore <backup_file>
```
- Restore from backup
- Verify integrity

## Organization Structure

```
src/model_tea/utils/
├── __init__.py                 # Core utilities
├── file_utils.py              # File operations
├── text_processing.py         # Text processing
├── training_utils.py          # Training helpers
├── errors.py                  # Error handling
├── novel/
│   ├── __init__.py
│   ├── import_novel.py
│   ├── preprocess_novel.py
│   ├── clean_novel.py
│   ├── analyze_novel.py
│   └── tag_novel.py
├── model/
│   ├── __init__.py
│   ├── create_model.py
│   ├── edit_model.py
│   ├── delete_model.py
│   └── list_models.py
├── mapping/
│   ├── __init__.py
│   ├── clean_mapping.py       # Existing
│   ├── generate_mapping.py    # Existing
│   ├── validate_mapping.py
│   ├── rebuild_mapping.py
│   └── merge_mappings.py
├── discovery/
│   ├── __init__.py
│   ├── scan_novels.py
│   ├── find_unmapped.py
│   └── stats.py
└── system/
    ├── __init__.py
    ├── setup.py
    ├── backup.py
    └── restore.py
```

## CLI Integration

Add new `utils` command group:

```bash
model-tea utils <subcommand>
```

All utilities accessible via CLI and programmatically.

## Implementation Priority

### Phase 1 (Essential)
1. validate_mapping.py
2. analyze_novel.py
3. scan_novels.py
4. create_model.py

### Phase 2 (Important)
5. import_novel.py
6. preprocess_novel.py
7. edit_model.py
8. stats.py

### Phase 3 (Nice to have)
9. clean_novel.py
10. tag_novel.py
11. backup.py
12. merge_mappings.py
