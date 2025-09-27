# Novel Training System

A specialized language model training system that creates **18 genre-specific models** from classic literature using GPT-OSS RL training principles. Each model becomes an expert in specific literary genres and styles.

## 🎯 Overview

This system trains **18 specialized models** on carefully categorized collections of classic novels:

- **bc_sprinkles**: 26 novels, 961k words - Classic adventure/mystery
- **sf_parfait**: 25 novels, 1.05M words - Science fiction collection
- **vs_mintchip**: 26 novels, 1.4M words - Gothic/horror themes
- **tr_creamsoda**: 25 novels, 1.36M words - Literary fiction
- *...and 14 more specialized categories*

Built on GPT-OSS reinforcement learning with novel-specific reward functions, adaptive chunking, and intelligent categorization for optimal literary style learning.

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install unsloth torch transformers datasets trl

# Requires CUDA-capable GPU for training
# Works on Google Colab free tier (15GB GPU)
```

### One-Time Setup
```bash
# Clean and process all 250+ novels (already done)
python simple_novel_processor.py --list  # View processed novels

# Verify model categorizations
python analyze_categories.py
```

### Train All Models
```bash
# Train 18 specialized models on categorized collections
python multi_model_trainer.py

# Each model trains on 20-30 related novels (600k-1.4M words)
# Training time: ~60-90 minutes for all models
# Results saved to multi_model_experiments/
```

### Interactive Generation
```bash
# Chat with trained models
python novel_chat.py --experiment [model_name]

# Example: Generate science fiction with sf_parfait model
# Example: Generate horror with vs_mintchip model
```

## 📁 Project Structure

```
model-trainer/
├── 🎯 Core Training
│   ├── multi_model_trainer.py      # Main training system
│   └── novel_chat.py              # Interactive generation
│
├── 🔧 Data Processing
│   ├── simple_novel_processor.py   # Novel cleaning and analysis
│   ├── analyze_categories.py       # Model categorization analysis
│   └── create_clean_mapping.py     # Mapping optimization
│
├── 📊 Configuration
│   ├── cleaned_model_mapping.json  # Model-to-novel assignments (18 models)
│   └── simple_processing_status.json # Processing status (250 novels)
│
├── 📚 Processed Novels (250 total)
│   └── novels/[novel_name]/
│       ├── [novel_name].txt        # Cleaned text
│       └── analysis.json           # Style analysis
│
└── 🏗️ Training Results
    └── multi_model_experiments/     # 18 trained models
        ├── bc_sprinkles/
        ├── sf_parfait/
        ├── vs_mintchip/
        └── ... (15 more)
```

## 🎭 Specialized Models

### Genre Categories (18 Models Total)

**Science Fiction & Fantasy**
- `sf_parfait` - Space exploration, technology, future worlds
- `fx_keylimepie` - Fantasy adventures, magical realms

**Mystery & Adventure**
- `bc_sprinkles` - Classic detective stories, mysteries
- `tr_creamsoda` - Adventure tales, exploration narratives

**Horror & Gothic**
- `vs_mintchip` - Gothic atmosphere, supernatural horror
- `hl_tiramisu` - Psychological tension, dark themes

**Literary Fiction**
- `f8_cheesecake` - Character development, literary prose
- `mt_sorbet` - Modern literary techniques

**Historical & Classical**
- `xg_rootbeer` - Historical fiction, period pieces
- `pe_peachcobbler` - Classical literature, formal prose

**..and 8 additional specialized categories**

### Benefits of Multi-Model Approach
- ✅ **Genre Expertise**: Each model specializes in specific literary styles
- ✅ **Optimal Data Size**: 600k-1.4M words per model (sweet spot for learning)
- ✅ **Production Ready**: 18 manageable models vs 250 individual ones
- ✅ **Better Generalization**: Multiple novels reinforce style consistency
- ✅ **Selective Deployment**: Choose the right model for your content type

## 🧠 Technical Features

### Adaptive Processing
- **Smart Chunking**: 800-token chunks with sentence boundaries
- **Variable Overlap**: Context preservation between chunks
- **Quality Filtering**: Minimum word counts and style analysis
- **Genre-Specific**: Chunking strategy adapts to collection size

### Reinforcement Learning Training
- **GRPO Algorithm**: Group Relative Policy Optimization
- **Style Fidelity Rewards**: Match literary characteristics of the genre
- **Narrative Quality Rewards**: Ensure coherent storytelling
- **Creativity Rewards**: Generate original content within style constraints
- **Anti-Reward Hacking**: Sophisticated techniques to prevent gaming

### Model Architecture
- **Base Model**: Llama 3.2 3B (4-bit quantized for efficiency)
- **LoRA Adapters**: Efficient fine-tuning (rank 8, alpha 16)
- **Context Length**: 1024 tokens
- **Training Steps**: 200 per model (4x more than single-novel)
- **Memory Efficient**: Works on 15GB GPU memory

## 📈 Performance & Results

### Training Efficiency
- **Total Training Time**: 60-90 minutes for all 18 models
- **Per Model**: ~3-5 minutes on modern GPU
- **Memory Usage**: 15GB VRAM (Google Colab compatible)
- **Data Efficiency**: 600k-1.4M words per specialized model

### Quality Metrics
- **Style Consistency**: Genre-specific expertise
- **Narrative Coherence**: Multi-novel training improves story structure
- **Creative Range**: Broader vocabulary and patterns vs single-novel
- **Deployment Ready**: Professional-grade model collection

## 🎮 Interactive Chat Interface

```bash
# Launch interactive mode
python novel_chat.py

# Available commands
/help          - Show available commands
/models        - List available trained models
/switch [name] - Switch to different model
/info          - Display current model information
/temp X        - Set generation temperature (0.1-2.0)
/tokens X      - Set max generation length (50-800)
/save          - Save conversation to file
/clear         - Clear conversation history
/quit          - Exit chat

# Example generation session
> /switch sf_parfait
Switched to sf_parfait (Science Fiction)

> Write about first contact with an alien civilization
The deep-space monitoring station's instruments detected the anomaly
at 0347 hours: a geometric pattern of electromagnetic signatures
unlike anything in the stellar cartography databases. Dr. Chen's
hands trembled as she realized the implications—this was no natural
phenomenon, but the unmistakable signature of intelligence...

> /switch vs_mintchip
Switched to vs_mintchip (Gothic Horror)

> Write about an old mansion
The Blackwood estate loomed against the storm-darkened sky, its
Victorian spires twisted like arthritic fingers clawing at the
heavens. Eleanor approached the wrought-iron gates with mounting
dread, for she knew that crossing the threshold would seal her
fate irrevocably...
```

## 🔧 Configuration & Customization

### Training Parameters
```python
# In multi_model_trainer.py - MultiModelConfig
base_model = "unsloth/llama-3.2-3b-bnb-4bit"
max_seq_length = 1024
learning_rate = 2e-4
max_steps = 200
temperature = 0.8
lora_rank = 8
```

### Model Categorization
```bash
# View current categorizations
python analyze_categories.py

# Regenerate clean mapping
python create_clean_mapping.py

# Add new novels to existing categories
# 1. Place .txt files in novels__uncleaned/
# 2. Run simple_novel_processor.py
# 3. Update data/ folder assignments
# 4. Regenerate mapping and retrain
```

## 📊 Dataset Statistics

### Novel Collection
- **Total Novels**: 250 processed and categorized
- **Total Words**: 17.2M across all categories
- **Average per Model**: 959k words (optimal for style learning)
- **Size Distribution**:
  - Small (20k words): 121 novels
  - Medium (20k-80k): 95 novels
  - Large (80k+): 34 novels

### Literary Genres Covered
- **Science Fiction**: Space exploration, technology, aliens
- **Fantasy**: Magic, mythical creatures, alternate worlds
- **Mystery**: Detective stories, puzzles, crime solving
- **Horror**: Gothic atmosphere, supernatural, psychological
- **Adventure**: Exploration, action, heroic journeys
- **Literary Fiction**: Character studies, modern prose
- **Historical**: Period pieces, historical events
- **Classical**: Formal prose, traditional narratives

## 🚀 Hardware Requirements

### Minimum Requirements
- **GPU**: 12GB VRAM (RTX 3060, RTX 4060 Ti)
- **RAM**: 16GB system memory
- **Storage**: 15GB free space
- **CUDA**: Version 12.0 or higher

### Recommended Setup
- **GPU**: 16GB+ VRAM (RTX 4080, RTX 4090, A100)
- **RAM**: 32GB system memory
- **Storage**: 25GB free space (for all experiments)

### Cloud Training
- **Google Colab Free**: Works perfectly (15GB GPU)
- **Google Colab Pro**: Faster training, more reliability
- **Local Development**: NVIDIA GPU with proper drivers

## 🧪 Applications & Use Cases

### Creative Writing
- **Genre-Specific Generation**: Choose the right model for your story type
- **Style Consistency**: Maintain genre conventions and atmosphere
- **Interactive Storytelling**: Collaborative writing with AI assistance
- **Character Development**: Genre-appropriate dialogue and narrative voice

### Research Applications
- **Literary Analysis**: Study genre characteristics and evolution
- **Style Transfer**: Compare writing techniques across periods
- **Content Analysis**: Automated genre classification
- **Educational Tools**: Demonstrate literary style differences

### Commercial Applications
- **Content Creation**: Generate genre-specific marketing copy
- **Game Development**: Dynamic narrative generation for RPGs
- **Publishing**: Assist with style-consistent content
- **Entertainment**: Interactive fiction and storytelling apps

## 🛠️ Development & Extension

### Adding New Genres
1. **Collect Novels**: Add .txt files to `novels__uncleaned/`
2. **Process**: Run `simple_novel_processor.py`
3. **Categorize**: Create new folder in `data/` with novel assignments
4. **Map**: Run `analyze_categories.py` and `create_clean_mapping.py`
5. **Train**: Execute `multi_model_trainer.py`

### Custom Reward Functions
```python
# Extend MultiModelRewards in multi_model_trainer.py
def genre_specific_reward(self, completions, **kwargs):
    """Custom reward for your specific genre requirements"""
    scores = []
    for completion in completions:
        # Implement your scoring logic
        score = analyze_genre_specific_features(completion)
        scores.append(score)
    return scores
```

### Model Fine-Tuning
- Adjust training steps for genre complexity
- Modify chunk sizes for different narrative structures
- Experiment with different base models
- Customize reward weightings per genre

## 🔍 Troubleshooting

### Common Issues
```bash
# CUDA/GPU Issues
nvidia-smi                    # Check GPU status
pip install torch --upgrade   # Update PyTorch

# Memory Issues
# Reduce batch size in training config
# Use gradient checkpointing
# Close other GPU applications

# Model Loading Issues
# Ensure models exist in multi_model_experiments/
# Check file permissions
# Verify model format compatibility
```

### Performance Optimization
- **Memory Efficiency**: Use 4-bit quantization (already enabled)
- **Speed**: Enable torch.compile for faster inference
- **Quality**: Increase training steps for better convergence
- **Storage**: Use model compression for deployment

## 📄 License & Acknowledgments

**License**: MIT License

**Acknowledgments**:
- **OpenAI**: GPT-OSS RL training methodology
- **Unsloth**: Efficient training framework
- **Project Gutenberg**: Classic literature source
- **Hugging Face**: Transformers and model ecosystem

**Original Contributions**:
- Multi-model genre categorization system
- Novel-specific reward function design
- Adaptive chunking strategies for literature
- Genre-specialized training pipeline
- Interactive multi-model chat interface

---

**🎯 Ready to Train**: Run `python multi_model_trainer.py` to create your collection of 18 specialized literary AI models!

**💡 Next Steps**: After training, use `python novel_chat.py` to explore the unique voice and style each model has learned from its literary genre.