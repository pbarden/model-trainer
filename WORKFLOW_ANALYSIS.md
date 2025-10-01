# Model Tea Training Workflow - Code Analysis

## Complete Workflow Overview

This document traces the execution flow through all scripts based on code review only.

---

## Script 1: `analyze_novel_lengths.py`

### Purpose
Categorize novels into tiers based on word count for optimal training.

### Execution Flow

**1. Startup (`main()`)**
- Prints header
- Calls `analyze_novels("novels")`

**2. Novel Discovery (`analyze_novels()`)**
- Iterates through `novels/` directory
- For each subdirectory:
  - Finds `*.txt` files using `glob("*.txt")`
  - Takes first txt file found
  - Calls `count_words_in_file()` to get word count
  - Creates dict: `{directory_name, file_name, words, file_path}`
  - Appends to `novel_data` list
- Returns list of all novel metadata

**3. Tier Categorization (`categorize_into_tiers()`)**
- Sorts `novel_data` by word count (ascending)
- Iterates through novels, assigns to tier based on word count:
  - `< 20,000` → `tiny`
  - `20,000 - 43,999` → `very_short`
  - `44,000 - 66,999` → `short`
  - `67,000 - 93,999` → `long`
  - `94,000+` → `very_long`
- Returns dict with 5 tier lists

**4. Analysis Output**
- `print_tier_analysis()`: Prints counts, word ranges, averages per tier
- `show_very_long_novels()`: Lists novels needing splitting, calculates recommended parts (2-4)
- `calculate_training_recommendations()`: Calculates how many 4-novel models per tier
- `save_tier_files()`: Writes 5 JSON files:
  - `tier_tiny.json`
  - `tier_very_short.json`
  - `tier_short.json`
  - `tier_long.json`
  - `tier_very_long.json`

**Output Files:**
- 5 tier JSON files containing novel metadata

---

## Script 2: `split_long_novels.py`

### Purpose
Split novels over 94K words into multiple parts.

### Execution Flow

**1. Startup (`main()`)**
- Parses args: `--max-words` (default 94000), `--novels-dir` (default "novels")
- Calls `split_very_long_novels()`

**2. Load Tier Data (`load_tier_file()`)**
- Opens `tier_very_long.json`
- Returns list of novels to split

**3. Novel Splitting Loop**
- For each novel in very_long tier:
  - Determines `num_parts` based on word count:
    - `< 140K` → 2 parts
    - `140K - 190K` → 3 parts
    - `190K+` → 4 parts
  - Calls `extract_part_number()` to check if already a part (e.g., "novel_part2")
  - Calls `get_base_name()` to remove existing part suffix
  - Calls `calculate_new_part_numbers()` to determine new part numbers
    - If no existing part: [1, 2, 3, ...]
    - If already part N: renumbers sequentially from N
  - Calls `split_novel_into_parts()`:
    - Reads all lines from file
    - Divides by `num_parts`
    - Splits at line boundaries
    - Returns list of content strings
  - For each part:
    - Creates directory: `{base_name}_part{N}/`
    - Writes file: `{base_name}_part{N}/{base_name}_part{N}.txt`
  - Deletes original file and directory

**Output:**
- Multiple part directories with split novel files
- Original files deleted

---

## Script 3: `generate_optimized_models.py`

### Purpose
Generate optimized model combinations with adaptive training parameters.

### Execution Flow

**1. Initialization (`__init__()`)**
- Creates `research_log` dict with metadata, methodology
- Initializes empty `tier_data`, `existing_models`, `generated_models`

**2. Load Data**

**`load_tier_data()`:**
- Opens each tier JSON file (tiny, very_short, short, long, very_long)
- Loads into `self.tier_data[tier_name]`
- Logs to `research_log['data_sources']`

**`load_existing_models()`:**
- Opens `model_mapping.json`
- Loads `data['models']` into `self.existing_models`
- Analyzes each model:
  - Extracts prefix (2-letter code)
  - Counts novels
  - Logs if >4 novels (too large)

**3. Generate Models (`generate_all_models()`)**

**Setup:**
- Calls `create_novel_pool()`:
  - Copies all tier data
  - Adds `'assigned': False` flag to each novel
- Extracts prefix from existing models (or defaults to 'rm')

**Model Generation:**
- Defines 5 model types with tier selection patterns:
  1. **balanced**: [('tiny', 1), ('very_short', 1), ('short', 1), ('long', 1)] × 15
  2. **shortfocus**: [('tiny', 2), ('very_short', 2)] × 12
  3. **mediumfocus**: [('very_short', 2), ('short', 2)] × 15
  4. **longfocus**: [('short', 1), ('long', 3)] × 7
  5. **escalating**: [('tiny', 1), ('very_short', 1), ('short', 1), ('long', 1)] × 8

**For each model type:**
- Calls `create_model_group(pools, tier_selection, prefix, type, count)`

**`create_model_group()` logic:**
- Loops `count` times (number of models to create)
- For each model:
  - Selects novels based on tier_selection pattern
  - Picks random available (not assigned) novel from each tier
  - Marks novels as assigned
  - If all required novels found:
    - Calculates `total_words` = sum of novel word counts
    - Calls `AdaptiveTrainingParameters.calculate_parameters(total_words)`
    - Creates model dict with novels, params, tier distribution
    - Logs decision to `research_log`

**`AdaptiveTrainingParameters.calculate_parameters()`:**
- Based on total_words, returns dict with:
  - `< 80K`: 10 iter, 12 steps, LR 3e-5→8e-6, category='tiny'
  - `80K-120K`: 12 iter, 14 steps, LR 2.5e-5→6e-6, category='small'
  - `120K-180K`: 14 iter, 16 steps, LR 2e-5→5e-6, category='medium'
  - `180K-260K`: 16 iter, 18 steps, LR 1.5e-5→4e-6, category='large'
  - `260K+`: 18 iter, 20 steps, LR 1e-5→3e-6, category='xlarge'

**Special handling:**
- For 'escalating' models: sorts novels by word count (ascending)

**4. Convert to Model Mapping Format (`convert_to_mapping_format()`)**
- Creates final structure:
```json
{
  "metadata": {
    "total_models": N,
    "created_by": "...",
    "date": "...",
    "adaptive_parameters": true
  },
  "models": {
    "rm_balanced1": {
      "description": "Balanced 1",
      "type": "balanced",
      "novel_count": 4,
      "total_word_count": 175000,
      "training_parameters": {
        "max_iterations": 14,
        "max_steps_per_iteration": 16,
        "learning_rate_start": 2e-5,
        "learning_rate_end": 5e-6,
        "size_category": "medium"
      },
      "tier_distribution": {...},
      "novels": [...]
    }
  }
}
```

**5. Statistics (`calculate_statistics()`)**
- Counts models by type and size category
- Calculates word count distribution
- Tracks novel utilization by tier
- Counts parameter distributions

**6. Save Outputs (`save_outputs()`)**
- Backs up existing `model_mapping.json` → `model_mapping_backup_{timestamp}.json`
- Saves new `model_mapping.json` (for combined_model_trainer.py)
- Saves `research_model_mapping.json` (research copy)
- Saves `model_generation_log.yaml` (complete decision log)
- Saves `model_statistics.yaml` (statistics only)

**Output Files:**
- `model_mapping.json` - Used by trainer
- `model_mapping_backup_*.json` - Backup
- `research_model_mapping.json` - Research copy
- `model_generation_log.yaml` - Decision log
- `model_statistics.yaml` - Stats

---

## Script 4: `combined_model_trainer.py`

### Purpose
Train combined models using adaptive parameters from model_mapping.json.

---

### SINGLE MODEL TRAINING: `--model rm_balanced1`

**1. Startup (`main()`)**
- Parses args
- Creates `CombinedModelTrainer()`

**2. Initialization (`__init__()`)**
- Creates `CombinedModelConfig()` (uses defaults)
- Calls `_load_model_mapping()`:
  - Opens `model_mapping.json`
  - Returns entire JSON structure
  - Stores in `self.model_mapping`
- Creates `QualityValidator`
- Sets paths: `novels_dir = "novels"`, `models_dir = "iterative_models"`
- Calls `validate_system_setup()`

**3. Check if Already Trained**
- Checks if `iterative_models/rm_balanced1/final/` exists
- If exists and not `--force`: exits with message
- If not exists or `--force`: continues

**4. Train Model (`train_combined_model("rm_balanced1")`)**

**Step 1: Get Novels**
- Calls `get_novels_for_model("rm_balanced1")`
  - Looks up `self.model_mapping["models"]["rm_balanced1"]`
  - Returns `model_info["novels"]` list (4 novels)
- Validates: at least 2 novels required

**Step 2: Check Individual Training Status**
- Calls `check_individual_novels_trained(novels)`
  - For each novel: checks if `iterative_models/{directory_name}/final/` exists
  - Returns dict: `{directory_name: True/False}`
- Logs warning if any not individually trained (optional)

**Step 3: Combine Novel Contents**
- Calls `combine_novel_contents(novels, "concatenate")`
  - For each novel:
    - Calls `load_novel_content(directory_name)`:
      - Looks in `novels/{directory_name}/` for:
        - `content.txt`
        - `{directory_name}.txt`
        - `novel.txt`
      - Returns first found file content
    - Counts words
    - If method == "concatenate":
      - Adds separator: `"\n\n=== NEW NOVEL ===\n\n"`
      - Adds header: `"=== {original_name} ===\n\n"`
      - Adds content
  - Joins all content with `"\n"`
  - Returns single combined string

**Step 4: Load Adaptive Training Parameters**
- Gets `model_info = self.model_mapping["models"]["rm_balanced1"]`
- Gets `training_params = model_info.get("training_parameters", {})`
- If `training_params` exists:
  - Logs adaptive parameters
  - Creates `IterativeConfig` with:
    - `iterations_per_novel = training_params['max_iterations']` (e.g., 14)
    - `max_steps_per_iteration = training_params['max_steps_per_iteration']` (e.g., 16)
    - `learning_rate_start = training_params['learning_rate_start']` (e.g., 2e-5)
    - `learning_rate_end = training_params['learning_rate_end']` (e.g., 5e-6)
- Else:
  - Uses defaults from `self.config` (12 iter, default LR)

**Step 5: Create Trainer and Train**
- Creates `IterativeTrainer(training_config)`
- Calls `trainer._train_with_content(combined_content, "rm_balanced1")`
  - **This goes to iterative_novel_trainer.py** (see below)
- Returns `training_results` dict

**Step 6: Update Results**
- Adds to training_results:
  - `model_type: "combined"`
  - `model_key: "rm_balanced1"`
  - `novels_included: [list of original names]`
  - `novel_count: 4`
  - `total_training_time`
  - `combination_method: "concatenate"`

**Step 7: Create Memories (if available)**
- Calls `_create_combined_memories()` if episodic memory system available
  - Creates `MemoryConfig` with 350 memories
  - Creates `EpisodicMemorySystem`
  - Writes temp file with combined content
  - Calls `memory_system.build_memory_for_model()`
  - Deletes temp file
  - Adds memory analysis to results

**Step 8: Save Results**
- Saves `iterative_models/rm_balanced1/training_results.json`
- Logs completion

**Output:**
- `iterative_models/rm_balanced1/final/` - Trained model
- `iterative_models/rm_balanced1/iteration_*/` - Checkpoints
- `iterative_models/rm_balanced1/training_results.json` - Metrics

---

### ALL MODELS TRAINING: `--all-models`

**Execution (`train_all_combined_models()`)**

**1. Get Available Models**
- Calls `get_available_models()`:
  - Returns list of all keys in `self.model_mapping["models"]`
  - E.g., ["rm_balanced1", "rm_balanced2", ..., "rm_longfocus7"]

**2. Loop Through Models**
- For each `model_key` in available_models:
  - Logs: `"=== Training Model {i+1}/{total}: {model_key} ==="`
  - Checks if `iterative_models/{model_key}/final/` exists
  - If exists:
    - Logs: "already trained, skipping"
    - Adds to results: `{model_key: {"status": "already_trained", "path": ...}}`
    - Continues to next model
  - If not exists:
    - Calls `train_combined_model(model_key)`
      - **Executes full single model flow described above**
    - On success:
      - Adds to results: `{model_key: {"status": "success", "results": ...}}`
    - On exception:
      - Logs error
      - Adds to results: `{model_key: {"status": "failed", "error": ...}}`

**3. Summary**
- Counts success, already_trained, failed
- Prints summary stats

**Output:**
- One trained model per model key in `iterative_models/`
- Each with final/, iteration_*/, training_results.json

---

## Script 5: `iterative_novel_trainer.py`

### Purpose
Core training logic called by combined_model_trainer.py.

### Execution Flow: `_train_with_content(combined_content, "rm_balanced1")`

**1. Setup**
- Logs content length (characters, words)
- Imports transformers classes
- Loads base model and tokenizer:
  - `AutoTokenizer.from_pretrained("gpt2")`
  - `AutoModelForCausalLM.from_pretrained("gpt2")`
- Sets `pad_token = eos_token` if needed
- Initializes results dict

**2. Iterative Training Loop**
- Loops from 0 to `config.iterations_per_novel` (e.g., 14):

**Iteration N:**

**Step 1: Create Progressive Chunks**
- Calls `processor.create_progressive_chunks(content, iteration)`
  - Calculates chunk size with progression:
    - `base_size = 150` (from config)
    - `progression_factor = 1 + (iteration * 0.2)`
    - `current_chunk_size = base_size * progression_factor`
    - E.g., iter 0: 150 words, iter 5: 300 words, iter 10: 450 words
  - Splits content into sentences
  - Groups sentences into chunks of ~current_chunk_size words
  - Adds overlap (30 words from previous chunk)
  - Returns list of chunk strings

**Step 2: Limit Chunks (for CPU efficiency)**
- If chunks > 15: samples evenly to reduce to 15 max
  - `step = len(chunks) // 15`
  - Takes every Nth chunk

**Step 3: Train/Val Split**
- Calls `processor.create_train_val_split(chunks)`
  - Shuffles chunks randomly
  - Splits at 85/15 ratio (15% validation)
  - Returns train_chunks, val_chunks

**Step 4: Prepare Dataset**
- Calls `_prepare_dataset(train_chunks, tokenizer)`
  - Creates HuggingFace Dataset from text chunks
  - Tokenizes with max_length=512, truncation
  - Returns tokenized dataset

**Step 5: Calculate Learning Rate**
- Linear interpolation between start and end:
  - `lr_progress = iteration / (total_iterations - 1)`
  - `current_lr = start_lr * (1 - progress) + end_lr * progress`
  - E.g., iter 0: 2e-5, iter 7: 1.25e-5, iter 14: 5e-6

**Step 6: Training Arguments**
- Creates `TrainingArguments`:
  - `output_dir = iterative_models/{model_key}/iteration_{N}`
  - `max_steps = config.max_steps_per_iteration` (e.g., 16)
  - `per_device_train_batch_size = 2`
  - `gradient_accumulation_steps = 8` (effective batch = 16)
  - `learning_rate = current_lr`
  - `warmup_ratio = 0.15`
  - `use_cpu = True`

**Step 7: Train**
- Creates `Trainer` with model, args, train_dataset
- Calls `trainer.train()`
  - Runs 16 steps (max_steps_per_iteration)
  - Uses gradient accumulation
  - Updates model weights

**Step 8: Validate**
- Calls `validator.test_generation_quality(model, tokenizer, "The story begins")`
  - Generates text with temperature 0.7, 0.9
  - Assesses quality (repetition ratio, sentence structure)
  - Returns quality score
- Calls `validator.calculate_perplexity(model, tokenizer, val_chunks[:5])`
  - Runs model on validation chunks
  - Calculates loss
  - Returns perplexity score

**Step 9: Log Results**
- Appends to `results["iterations"]`:
  - iteration number
  - perplexity
  - quality_score
  - training_time
  - learning_rate
  - chunk_size
  - num_chunks

**Step 10: Save Checkpoint**
- If `save_checkpoints = True`:
  - Saves model to `iteration_{N}/`
  - Saves tokenizer to `iteration_{N}/`

**Step 11: Check Continue**
- Calls `validator.should_continue_training(iteration)`
  - **Always returns True** (completes all iterations)

**3. Post-Training**

**Save Final Model:**
- Creates `iterative_models/{model_key}/final/`
- Calls `model.save_pretrained(final_model_dir)`
- Calls `tokenizer.save_pretrained(final_model_dir)`

**Build Episodic Memory:**
- Calls `_build_episodic_memory(model_name, novel_path)`
  - If available, creates memory system
  - Extracts 250 memories from content
  - Returns memory analysis

**Comprehensive Testing:**
- Calls `_conduct_comprehensive_testing()`
  - Tests generation quality with multiple prompts
  - Calculates academic metrics
  - Analyzes novel characteristics
  - Returns testing results

**Save Academic Outputs:**
- Calls `_save_academic_outputs()`
  - Saves `testing_outputs/{model_name}_model_analysis.json`
  - Saves `testing_outputs/{model_name}_summary_report.txt`

**4. Return Results**
- Returns full results dict with:
  - All iteration metrics
  - Final quality score
  - Training time
  - Memory analysis
  - Test results

---

## Script 6: `interactive_chat.py`

### Purpose
Interactive chat interface for trained models.

### Execution Flow: `python interactive_chat.py`

**1. Initialization (`__init__()`)**
- Sets `models_dir = "iterative_models"`
- Sets default generation parameters:
  - `max_length = 250`
  - `temperature = 0.5`
  - `top_p = 0.85`
  - `top_k = 30`
  - `repetition_penalty = 1.4`
- Stores as instance variables
- Registers command handlers

**2. Model Selection (`select_model()`)**
- Calls `list_available_models()`:
  - Scans `iterative_models/` directory
  - For each subdirectory:
    - Checks if contains `config.json` or `model.safetensors`
    - If not, checks subdirectories (like `final/`)
    - Prioritizes `final/` directories
  - Returns sorted list of model paths
- Displays numbered list to user
- User enters number or command
- Returns selected model path

**3. Model Loading (`load_model(model_path)`)**
- Detects device: "cuda" if available, else "cpu"
- Calls `AutoTokenizer.from_pretrained(model_path)`
- Calls `AutoModelForCausalLM.from_pretrained(model_path)`:
  - Uses `torch.float32` for CPU
  - Uses `torch.float16` for GPU
- Moves model to device
- Sets `pad_token = eos_token` if needed
- Stores model, tokenizer, model_name

**4. Chat Loop (`chat_loop()`)**

**Setup:**
- Creates `settings` dict with default generation parameters
- Initializes empty conversation history
- Initializes empty conversation log

**Main Loop:**
- Prompts: `"You: "`
- Gets user input

**Command Handling:**
- If `/quit`, `/exit`, `/q`: breaks loop
- If `/clear`: clears history and log
- If `/history`: displays all past exchanges
- If `/export`: saves conversation to `conversation_{model_name}_{timestamp}.txt`
- If `/switch`: calls `select_model()` and `load_model()` again
- If `/models all`: calls `show_models_list(show_all=True)`
- If `/settings`: calls `show_settings_menu(settings)`
  - Allows adjusting parameters or loading presets
- Other commands: calls registered handler

**Text Generation:**
- Builds prompt:
  - If history exists: `"{history}\nYou: {input}\nAssistant:"`
  - Else: `"You: {input}\nAssistant:"`
- Calls `generate_response()`:
  - Tokenizes prompt with padding and attention mask
  - Calls `model.generate()` with:
    - `max_new_tokens = settings['max_length']`
    - `temperature = settings['temperature']`
    - `top_p = settings['top_p']`
    - `top_k = settings['top_k']`
    - `repetition_penalty = settings['repetition_penalty']`
    - `do_sample = True`
    - `no_repeat_ngram_size = 3`
  - Decodes only new tokens (skips prompt)
  - Returns response string
- Prints response
- Appends to conversation_log: `{user: input, assistant: response, timestamp}`
- Updates conversation_history with prompt + response
- Keeps last 4 exchanges only (for memory management)

**5. Exit**
- Prints "Goodbye!"
- Returns to shell

---

## Complete Pipeline Flow

### Optimal Workflow

```
1. python analyze_novel_lengths.py
   → Creates tier_*.json files

2. python split_long_novels.py (if tier_very_long.json has novels)
   → Splits novels >94K words
   → Re-run analyze_novel_lengths.py to verify

3. python generate_optimized_models.py
   → Creates model_mapping.json with adaptive parameters
   → Backs up old model_mapping.json
   → Creates research logs (YAML)

4. python combined_model_trainer.py --all-models
   → For each model in model_mapping.json:
     - Reads adaptive training_parameters
     - Combines 4 novels
     - Trains with adaptive iterations/steps/LR
     - Saves to iterative_models/{model_key}/

5. python interactive_chat.py
   → Select trained model
   → Chat with model using adaptive generation settings
```

---

## Data Flow

```
novels/                          (Raw text files)
    ↓
analyze_novel_lengths.py
    ↓
tier_*.json                      (Novel categorization)
    ↓
generate_optimized_models.py
    ↓
model_mapping.json               (Model definitions + adaptive params)
    ↓
combined_model_trainer.py
    ↓
iterative_models/                (Trained models)
    model_key/
        final/                   (Ready for inference)
        iteration_*/             (Checkpoints)
        training_results.json    (Metrics)
    ↓
interactive_chat.py              (Inference)
```

---

## Key Parameters Flow

### Novel → Tier → Model → Training

**Example: 4 novels with 15K, 35K, 55K, 80K words**

1. **Tier Assignment:**
   - 15K → tiny
   - 35K → very_short
   - 55K → short
   - 80K → long

2. **Model Creation (balanced mix):**
   - Selects 1 from each tier
   - Combined: 185K words

3. **Adaptive Parameters Calculation:**
   - 185K words → "large" category
   - Iterations: 16
   - Steps/iter: 18
   - LR: 1.5e-5 → 4e-6

4. **Training:**
   - Uses these exact parameters
   - Logs to training_results.json

5. **Inference:**
   - Uses model from iterative_models/{model_key}/final/
   - Default generation params (can be customized)

---

## Error Handling

### analyze_novel_lengths.py
- No novels found: Exits with message
- Unreadable file: Logs warning, skips novel

### split_long_novels.py
- tier_very_long.json missing: Prints error, exits
- File not found: Logs warning, skips novel
- Split failure: Logs error, keeps original

### generate_optimized_models.py
- Tier JSON missing: Logs warning, continues with available tiers
- Insufficient novels for tier pattern: Skips model creation (silently)
- No model_mapping.json: Creates from scratch

### combined_model_trainer.py
- model_mapping.json missing: Raises FileNotFoundError
- Novel directory missing: Logs error, skips novel
- Model key not found: Raises ValueError
- Training failure: Logs error, continues to next model (if --all-models)

### iterative_novel_trainer.py
- No text files: Raises FileNotFoundError
- Training iteration failure: Logs error, breaks loop
- Memory system unavailable: Logs warning, continues without memories

### interactive_chat.py
- No models found: Prints message, exits
- Model load failure: Prints error, exits
- Generation failure: Prints error, continues loop

---

## Logging and Output

### Console Logs
- All scripts use Python `logging` module
- Level: INFO
- Format: `timestamp - level - message`

### File Outputs

**analyze_novel_lengths.py:**
- `tier_tiny.json`
- `tier_very_short.json`
- `tier_short.json`
- `tier_long.json`
- `tier_very_long.json`

**generate_optimized_models.py:**
- `model_mapping.json` (for training)
- `model_mapping_backup_{timestamp}.json` (backup)
- `research_model_mapping.json` (research copy)
- `model_generation_log.yaml` (decisions + rationale)
- `model_statistics.yaml` (stats only)

**combined_model_trainer.py:**
- `iterative_models/{model_key}/final/` (model files)
- `iterative_models/{model_key}/iteration_{N}/` (checkpoints)
- `iterative_models/{model_key}/training_results.json` (metrics)
- `testing_outputs/{model_key}_model_analysis.json` (tests)
- `testing_outputs/{model_key}_summary_report.txt` (summary)

**interactive_chat.py:**
- `conversation_{model_name}_{timestamp}.txt` (exported chats)

---

## Configuration Hierarchy

### Global Defaults
- `IterativeConfig` in iterative_novel_trainer.py: 12 iter, 16 steps, LR 2e-5→5e-6
- `CombinedModelConfig` in combined_model_trainer.py: 12 iter, LR 5e-5→5e-6
- `InteractiveChat` in interactive_chat.py: max_length 250, temp 0.5

### Model-Specific Overrides
- `model_mapping.json` → `training_parameters` → per-model adaptive settings
- Read by `combined_model_trainer.py` in `train_combined_model()`
- Passed to `IterativeConfig` constructor

### Runtime Overrides
- Command-line args: `python interactive_chat.py --temperature 0.7 --max-length 300`
- In-session commands: `/settings` menu

### Priority Order
1. Runtime args/commands (highest)
2. Model-specific parameters in model_mapping.json
3. Script-level defaults
4. Class-level defaults (lowest)

---

## Summary

This workflow creates a research-grade training pipeline where:

1. **Novels are categorized** by length for optimal grouping
2. **Models are strategically designed** with cross-tier mixing
3. **Training adapts automatically** based on combined corpus size
4. **All decisions are logged** in structured formats (YAML/JSON)
5. **Each model trains independently** with its own optimal parameters
6. **Results are reproducible** and traceable through logs

The system is designed for comprehensive research analysis, allowing later extraction of data showing how different novel length combinations and training parameters affect model quality.
