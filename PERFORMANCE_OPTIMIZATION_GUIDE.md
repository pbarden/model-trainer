# Model Training & Inference Performance Optimization Guide

## Executive Summary

**Current Performance:**
- Batch size: 2 (very small)
- Gradient accumulation: 8 steps
- Dataloader workers: 0 (no parallelization)
- Effective batch size: 2 × 8 = 16
- CPU threads: 4 (underutilized)

**Target: 2X Performance Improvement + Quality Maximization**

---

## PART 1: TRAINING OPTIMIZATIONS (2X Performance)

### 🔴 CRITICAL - Immediate 2X+ Speedup

#### 1. **Enable DataLoader Parallelization** (30-50% speedup)
```python
# Current (SLOW):
dataloader_num_workers=0

# Optimized:
dataloader_num_workers=min(4, os.cpu_count() - 1)  # Leave 1 core for training
dataloader_pin_memory=False  # CPU only, no benefit
```

**Why:** Data loading is currently blocking - the CPU waits idle while loading batches from disk.

#### 2. **Increase Batch Size + Reduce Gradient Accumulation** (40-60% speedup)
```python
# Current (INEFFICIENT):
batch_size: int = 2
gradient_accumulation_steps: int = 8
# Effective batch = 16, but processes in tiny chunks

# Optimized:
batch_size: int = 8  # 4x larger
gradient_accumulation_steps: int = 2  # 4x smaller
# Same effective batch = 16, but 4x fewer forward passes
```

**Why:** Larger batches utilize CPU vectorization better. Gradient accumulation adds overhead.

#### 3. **Optimize PyTorch Threading** (20-30% speedup)
```python
# Current:
torch.set_num_threads(os.cpu_count())  # May cause oversubscription

# Optimized:
import torch
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(2)  # Control parallel ops

# Add environment variables BEFORE importing torch:
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
```

#### 4. **Enable PyTorch CPU Optimizations** (15-25% speedup)
```python
# Add at initialization:
torch.backends.mkldnn.enabled = True  # Intel MKL-DNN optimizations
torch.backends.quantized.engine = 'qnnpack'  # CPU quantization support

# For training, consider channels_last memory format:
model = model.to(memory_format=torch.channels_last)  # Better cache locality
```

### 🟡 MODERATE - Quality + Performance Balance

#### 5. **Optimize Tokenization** (10-15% speedup)
```python
# Current: Tokenizes in batched map
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

# Optimized: Add num_proc for parallel tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,  # Larger batches
    num_proc=min(4, os.cpu_count() - 1),  # Parallel processing
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)
```

#### 6. **Gradient Checkpointing** (Memory vs Speed tradeoff)
```python
# For larger batch sizes without OOM:
model.gradient_checkpointing_enable()

# Trade: ~20% slower but allows 2x larger batches
# Net result: Still faster overall due to better vectorization
```

#### 7. **Optimize Learning Rate Schedule** (Quality improvement)
```python
# Current: Linear warmup + linear decay
warmup_ratio: float = 0.15

# Better: Cosine annealing with warmup
from transformers import get_cosine_schedule_with_warmup

lr_scheduler_type="cosine"  # Add to TrainingArguments
# Provides better convergence in fewer iterations
```

### 🟢 ADVANCED - Maximum Performance

#### 8. **Mixed Precision Training (CPU BF16)** (30-50% speedup on modern CPUs)
```python
# PyTorch 2.x supports BF16 on CPU with AVX512
training_args = TrainingArguments(
    bf16=True,  # BF16 on CPU (requires AVX512)
    bf16_full_eval=True,
    # ... other args
)

# Check if supported:
if torch.cuda.is_available() or hasattr(torch.cpu, 'amp'):
    use_bf16 = True
```

**Requires:** Intel CPUs with AVX512 (10th gen+) or AMD Zen4+

#### 9. **Compiled Models (PyTorch 2.0+)** (20-40% speedup)
```python
# After loading model:
model = torch.compile(
    model,
    mode="reduce-overhead",  # Best for training
    backend="inductor"  # CPU backend
)

# First iteration is slow (compilation), then much faster
```

#### 10. **Optimize Chunk Processing**
```python
# Current: Creates many small chunks
max_chunks = min(len(chunks), 15)  # Artificial limit

# Better: Dynamic chunking based on model size
def calculate_optimal_chunks(total_words, target_chunk_size):
    # Aim for ~100-200 chunks for good gradient diversity
    optimal_chunks = max(100, total_words // target_chunk_size)
    return min(optimal_chunks, 500)  # Cap at 500 for memory
```

---

## PART 2: QUALITY OPTIMIZATIONS

### 📈 Training Quality Improvements

#### 1. **Better Data Augmentation**
```python
# Add text augmentation for robustness:
def augment_training_data(text):
    # Random dropout of punctuation (5%)
    # Synonym replacement (10%)
    # Back-translation paraphrasing
    return augmented_text

# Increases effective training data 2-3x
```

#### 2. **Curriculum Learning Enhancement**
```python
# Current: Simple chunk size progression
progression_factor = 1 + (iteration * 0.2)

# Better: Difficulty-based curriculum
def calculate_difficulty_score(chunk):
    # Vocabulary diversity
    # Sentence complexity
    # Dialogue density
    return score

# Train on easier chunks first, harder later
```

#### 3. **Better Validation Strategy**
```python
# Current: Random 15% split
validation_split: float = 0.15

# Better: Stratified split by complexity
# Ensures validation set represents full difficulty range
def stratified_split(chunks):
    # Group by difficulty
    # Take 15% from each group
    return train, val
```

#### 4. **Label Smoothing** (Prevents overconfidence)
```python
training_args = TrainingArguments(
    label_smoothing_factor=0.1,  # Add this
    # Smooths target distribution, improves generalization
)
```

#### 5. **Gradient Clipping** (Training stability)
```python
training_args = TrainingArguments(
    max_grad_norm=1.0,  # Already present, ensure it's used
    # Prevents exploding gradients
)
```

---

## PART 3: INFERENCE OPTIMIZATIONS (Interactive Chat)

### ⚡ 1-Bit & Quantization Techniques

#### 1. **INT8 Quantization** (2-3x faster, minimal quality loss)
```python
# After loading model:
from torch.quantization import quantize_dynamic

model = quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8
)

# Expected: 2-3x faster inference, 4x smaller model
# Quality: ~1-2% perplexity increase
```

#### 2. **BitNet-style 1-Bit Weights** (8-10x faster)
```python
# Requires custom implementation or library
# Concept: Replace fp32 weights with 1-bit {-1, +1}

from transformers import AutoModelForCausalLM
import torch

def binarize_weights(model):
    """Convert weights to 1-bit representation"""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            # Binarize: sign(weight) * mean(|weight|)
            scale = param.abs().mean()
            param.data = param.data.sign() * scale
    return model

# WARNING: Requires fine-tuning after binarization
# Quality: 15-25% perplexity increase, but 8-10x faster
```

#### 3. **GPTQ Quantization** (Best quality/speed tradeoff)
```python
# Install: pip install auto-gptq optimum
from optimum.gptq import GPTQQuantizer

# Quantize to 4-bit:
quantizer = GPTQQuantizer(
    bits=4,  # 4-bit weights
    dataset="c4",  # Calibration dataset
    model_seqlen=512
)

model = quantizer.quantize_model(model, tokenizer)

# Expected: 3-4x faster, 4x smaller, <5% quality loss
```

#### 4. **Static KV-Cache** (50-70% faster for chat)
```python
def generate_with_static_cache(self, prompt, **kwargs):
    """Use static KV cache for faster generation"""

    # Pre-allocate KV cache
    past_key_values = self._create_static_cache(
        batch_size=1,
        max_length=kwargs.get('max_length', 250)
    )

    outputs = self.model.generate(
        inputs.input_ids,
        past_key_values=past_key_values,
        use_cache=True,  # Reuse cache
        **kwargs
    )

    return outputs

# Eliminates cache reallocation overhead
```

#### 5. **Speculative Decoding** (2-3x faster generation)
```python
# Use a small draft model + your main model
# Draft model generates quickly, main model verifies

from transformers import pipeline

# Small draft model (10x faster)
draft_model = AutoModelForCausalLM.from_pretrained("gpt2")

def speculative_generate(prompt, num_tokens=250):
    # Draft generates 5 tokens at once
    draft_tokens = draft_model.generate(
        prompt, max_new_tokens=5, do_sample=False
    )

    # Main model verifies and accepts/rejects
    verified = main_model.verify_tokens(draft_tokens)

    # Accept verified tokens, reject and retry rest
    # Net: 2-3x faster while maintaining quality
```

#### 6. **Prompt Caching** (Huge speedup for repeated contexts)
```python
class PromptCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size

    def get_cached_kv(self, prompt_prefix):
        """Cache KV for common prompt prefixes"""
        if prompt_prefix in self.cache:
            return self.cache[prompt_prefix]
        return None

    def add_to_cache(self, prompt_prefix, kv_cache):
        if len(self.cache) >= self.max_size:
            # Remove oldest
            self.cache.pop(next(iter(self.cache)))
        self.cache[prompt_prefix] = kv_cache

# For chatbots with system prompts, this is 10x faster
```

### 🚀 Advanced Inference Techniques

#### 7. **Token Healing** (Better first token quality)
```python
def generate_with_token_healing(prompt):
    """Recompute first token with fuller context"""

    # Generate normally
    output = model.generate(prompt, max_new_tokens=250)

    # Recompute first token with full context
    first_token_logits = model(
        torch.cat([prompt, output[:, :1]], dim=1)
    ).logits[:, -1, :]

    # Replace if better
    better_token = torch.argmax(first_token_logits)
    output[:, 0] = better_token

    return output
```

#### 8. **Batch Inference** (When processing multiple prompts)
```python
def batch_generate(prompts_list, batch_size=4):
    """Process multiple prompts in batches"""

    # Pad to same length
    inputs = tokenizer(
        prompts_list,
        padding=True,
        return_tensors="pt"
    )

    # Generate in batch (4x faster than sequential)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        **generation_config
    )

    return [tokenizer.decode(out) for out in outputs]
```

---

## IMPLEMENTATION PRIORITY

### Phase 1: Quick Wins (Implement first - 2x speedup)
1. ✅ DataLoader workers: 4
2. ✅ Batch size: 2→8, grad_accum: 8→2
3. ✅ PyTorch threading optimization
4. ✅ INT8 quantization for inference

### Phase 2: Advanced (Additional 1.5x speedup)
5. ✅ BF16 training (if CPU supports)
6. ✅ torch.compile()
7. ✅ GPTQ quantization for inference
8. ✅ Static KV cache

### Phase 3: Quality Improvements
9. ✅ Cosine learning rate schedule
10. ✅ Label smoothing
11. ✅ Better curriculum learning
12. ✅ Stratified validation split

---

## EXPECTED RESULTS

### Training Performance:
- **Before:** ~200 words/sec, 12 iterations × 200 steps = ~8-10 hours
- **After:** ~400-500 words/sec, 11 iterations × 200 steps = ~4-5 hours
- **Speedup:** 2-2.5x faster

### Training Quality:
- Better convergence (cosine schedule)
- More stable training (label smoothing)
- Better generalization (stratified validation)
- Expected: 5-10% better final quality metrics

### Inference Performance:
- **Before:** ~10-15 tokens/sec (CPU)
- **After (INT8):** ~30-40 tokens/sec
- **After (GPTQ 4-bit):** ~40-50 tokens/sec
- **After (Static cache):** ~60-80 tokens/sec
- **Speedup:** 4-8x faster

### Inference Quality:
- INT8: <1% quality loss
- GPTQ 4-bit: <5% quality loss
- 1-Bit: ~15-20% quality loss (not recommended without retraining)

---

## HARDWARE-SPECIFIC NOTES

**Your System:**
- PyTorch 2.8.0 (latest)
- MKL enabled ✓
- OpenMP enabled ✓
- CPU threads: 4

**Recommended Settings:**
```python
# Set BEFORE importing torch:
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# After importing:
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.mkldnn.enabled = True
```

**Check for AVX512 support:**
```bash
# Windows PowerShell:
wmic cpu get caption
# Look for Intel 10th gen+ or AMD Zen4+

# If supported, use BF16:
training_args.bf16 = True
```
