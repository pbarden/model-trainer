#!/usr/bin/env python3
"""
Single Novel Training Experiment
Tests dataset size implications using GPT-OSS RL infrastructure
Each model trains on one novel to study data efficiency and style learning
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from datasets import Dataset
import time
import statistics

# Try to import Unsloth components
try:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Unsloth not available. Install with: pip install unsloth")
    UNSLOTH_AVAILABLE = False

@dataclass
class ExperimentConfig:
    """Configuration for single novel experiments"""
    base_model: str = "unsloth/llama-3.2-3b-bnb-4bit"
    max_seq_length: int = 1024
    lora_rank: int = 8
    learning_rate: float = 2e-4
    max_steps: int = 50  # Small for quick experiments
    temperature: float = 0.8
    max_new_tokens: int = 300
    novels_dir: str = "novels"
    experiments_dir: str = "experiments"

class NovelProcessor:
    """Processes individual novels for training"""

    def __init__(self, novel_path: Path):
        self.novel_path = novel_path
        self.title = novel_path.stem.replace('_', ' ').title()

    def load_and_analyze(self) -> Dict[str, Any]:
        """Load novel and extract characteristics"""
        # Check if this is a directory (new structure) or file (old structure)
        if self.novel_path.is_dir():
            # New structure - find the text file in the directory
            text_files = list(self.novel_path.glob("*.txt"))
            if not text_files:
                raise FileNotFoundError(f"No .txt file found in {self.novel_path}")

            novel_file = text_files[0]
            analysis_file = self.novel_path / "analysis.json"

            # Load existing analysis if available
            existing_analysis = {}
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    existing_analysis = json.load(f)

            with open(novel_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Use directory name as title
            self.title = self.novel_path.name.replace('_', ' ').title()
        else:
            # Old structure - direct file
            with open(self.novel_path, 'r', encoding='utf-8') as f:
                content = f.read()
            existing_analysis = {}

        # Clean and preprocess if needed (might already be clean)
        if not existing_analysis:
            content = self._clean_text(content)

        # Extract style characteristics
        style_profile = existing_analysis.get('style_profile') or self._analyze_style(content)

        # Split into training chunks
        chunks = self._create_chunks(content)

        return {
            "title": self.title,
            "content": content,
            "word_count": existing_analysis.get('word_count', len(content.split())),
            "style_profile": style_profile,
            "training_chunks": chunks,
            "chunk_count": len(chunks)
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove Project Gutenberg headers/footers
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if 'START OF THE PROJECT GUTENBERG' in line.upper():
                start_idx = i + 1
            elif 'END OF THE PROJECT GUTENBERG' in line.upper():
                end_idx = i
                break

        clean_lines = lines[start_idx:end_idx]
        clean_text = '\n'.join(clean_lines)

        # Basic cleaning
        clean_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_text)  # Multiple newlines
        clean_text = re.sub(r'[^\w\s\.,;:!?"\'-]', '', clean_text)   # Special chars

        return clean_text.strip()

    def _analyze_style(self, text: str) -> Dict[str, float]:
        """Analyze writing style metrics"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()

        # Vocabulary analysis
        unique_words = set(word.lower() for word in words)

        # Sentence structure
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        # Punctuation patterns
        punctuation_count = sum(1 for c in text if c in '.,;:!?')

        # Dialogue detection
        dialogue_ratio = text.count('"') / len(text) * 100 if text else 0

        # Descriptive language
        descriptive_words = len(re.findall(r'\b\w+ly\b', text))  # Adverbs
        adjective_density = len(re.findall(r'\b\w+ed\b|\b\w+ing\b', text)) / len(words) if words else 0

        return {
            "vocabulary_richness": len(unique_words) / len(words) if words else 0,
            "avg_sentence_length": avg_sentence_length,
            "punctuation_density": punctuation_count / len(text) if text else 0,
            "dialogue_ratio": dialogue_ratio,
            "descriptive_density": descriptive_words / len(words) if words else 0,
            "adjective_density": adjective_density
        }

    def _create_chunks(self, text: str, chunk_size: int = 800) -> List[str]:
        """Split novel into training chunks"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk.split()) + len(para.split()) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.split()) > 50]  # Filter short chunks

class NovelSpecificRewards:
    """Reward functions tailored to specific novel's style"""

    def __init__(self, target_style: Dict[str, float], novel_title: str):
        self.target_style = target_style
        self.novel_title = novel_title

    def extract_generated_story(self, text: str) -> Optional[str]:
        """Extract story from completion"""
        # Find the actual generated content
        markers = ["Story:", "Here's", "```", "\n\n"]

        for marker in markers:
            if marker in text:
                parts = text.split(marker, 1)
                if len(parts) > 1:
                    return parts[1].strip()

        # Fallback
        lines = text.split('\n')
        if len(lines) > 1:
            return '\n'.join(lines[1:]).strip()
        return text

    def style_fidelity_reward(self, completions, **kwargs) -> List[float]:
        """Reward based on how well generated text matches novel's style"""
        scores = []

        for completion in completions:
            response = completion[0]["content"]
            story = self.extract_generated_story(response)

            if not story or len(story.split()) < 20:
                scores.append(-1.0)
                continue

            # Analyze generated style
            processor = NovelProcessor(Path("dummy"))  # Just for analysis methods
            generated_style = processor._analyze_style(story)

            # Calculate style similarity
            similarity_score = 0.0
            weight_sum = 0.0

            for feature, target_value in self.target_style.items():
                if feature in generated_style:
                    generated_value = generated_style[feature]

                    # Calculate normalized difference
                    if target_value > 0:
                        diff = abs(generated_value - target_value) / max(target_value, 0.1)
                        similarity = max(0, 1 - diff)
                    else:
                        similarity = 1.0 if generated_value == 0 else 0.0

                    similarity_score += similarity
                    weight_sum += 1.0

            final_score = similarity_score / weight_sum if weight_sum > 0 else 0.0
            scores.append(final_score * 4.0 - 1.0)  # Scale to roughly -1 to 3

        return scores

    def narrative_quality_reward(self, completions, **kwargs) -> List[float]:
        """Reward narrative coherence and quality"""
        scores = []

        for completion in completions:
            response = completion[0]["content"]
            story = self.extract_generated_story(response)

            if not story:
                scores.append(-2.0)
                continue

            score = 0.0

            # Length check
            word_count = len(story.split())
            if 50 <= word_count <= 500:
                score += 1.0
            elif word_count > 20:
                score += 0.5

            # Narrative elements
            if any(word in story.lower() for word in ['then', 'suddenly', 'however', 'meanwhile', 'later', 'finally']):
                score += 1.0

            # Descriptive language
            if re.search(r'\b\w+ly\b', story):  # Has adverbs
                score += 0.5

            # Character/dialogue
            if '"' in story:
                score += 0.5

            # Coherence (simple check)
            sentences = [s.strip() for s in re.split(r'[.!?]+', story) if s.strip()]
            if len(sentences) >= 3:
                score += 0.5

            scores.append(score)

        return scores

    def creativity_within_style_reward(self, completions, **kwargs) -> List[float]:
        """Reward creativity while maintaining style consistency"""
        scores = []

        for completion in completions:
            response = completion[0]["content"]
            story = self.extract_generated_story(response)

            if not story:
                scores.append(0.0)
                continue

            score = 0.0

            # Vocabulary diversity
            words = story.lower().split()
            unique_ratio = len(set(words)) / len(words) if words else 0
            score += unique_ratio * 2.0

            # Avoid repetition
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            max_freq = max(word_freq.values()) if word_freq else 0
            if max_freq <= len(words) * 0.1:  # No word appears more than 10% of time
                score += 1.0

            # Creative descriptions
            descriptive_patterns = [
                r'\b\w+ing\b',  # Present participles
                r'\b\w+ of \w+\b',  # Descriptive phrases
                r'\b(shimmering|gleaming|mysterious|ancient|ethereal)\b'
            ]

            for pattern in descriptive_patterns:
                if re.search(pattern, story.lower()):
                    score += 0.3

            scores.append(min(score, 3.0))

        return scores

class SingleNovelExperiment:
    """Manages experiments with individual novels"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.novels_dir = Path(config.novels_dir)
        self.experiments_dir = Path(config.experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)

    def run_experiment(self, novel_path: Path) -> Dict[str, Any]:
        """Run complete experiment on a single novel"""
        print(f"\n{'='*50}")
        print(f"Experiment: {novel_path.name}")
        print(f"{'='*50}")

        # Process novel
        processor = NovelProcessor(novel_path)
        novel_data = processor.load_and_analyze()

        print(f"Novel: {novel_data['title']}")
        print(f"Word count: {novel_data['word_count']:,}")
        print(f"Training chunks: {novel_data['chunk_count']}")
        print(f"Style profile: {novel_data['style_profile']}")

        # Create experiment directory
        exp_dir = self.experiments_dir / f"{novel_path.stem}_experiment"
        exp_dir.mkdir(exist_ok=True)

        # Save novel analysis
        with open(exp_dir / "novel_analysis.json", 'w') as f:
            json.dump(novel_data, f, indent=2)

        experiment_results = {
            "novel_info": novel_data,
            "config": self.config.__dict__,
            "training_results": None,
            "generation_samples": []
        }

        if UNSLOTH_AVAILABLE:
            # Setup and train model
            model, tokenizer = self._setup_model()

            # Prepare training data
            dataset = self._prepare_dataset(novel_data)

            # Train model
            training_results = self._train_model(model, tokenizer, dataset, novel_data['style_profile'])
            experiment_results["training_results"] = training_results

            # Generate samples
            samples = self._generate_samples(model, tokenizer, novel_data['title'])
            experiment_results["generation_samples"] = samples

            # Save model
            model_path = exp_dir / "trained_model"
            model.save_pretrained_merged(str(model_path), tokenizer, save_method="merged_16bit")

        # Save results
        with open(exp_dir / "experiment_results.json", 'w') as f:
            json.dump(experiment_results, f, indent=2)

        return experiment_results

    def _setup_model(self):
        """Setup model for training"""
        print("Setting up model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
            offload_embedding=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.config.lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        return model, tokenizer

    def _prepare_dataset(self, novel_data: Dict[str, Any]) -> Dataset:
        """Prepare training dataset from novel chunks"""
        training_examples = []

        prompts = [
            f"Write a passage in the style of {novel_data['title']}:",
            f"Continue this story in the same literary style:",
            f"Create a scene with similar atmosphere and tone:",
            f"Write descriptively in this author's voice:"
        ]

        # Use each chunk to create training examples
        for chunk in novel_data['training_chunks'][:20]:  # Limit for quick experiments
            for prompt in prompts:
                training_examples.append({
                    "prompt": [{"role": "user", "content": prompt}],
                    "answer": 0  # Placeholder for RL
                })

        print(f"Created {len(training_examples)} training examples")
        return Dataset.from_list(training_examples)

    def _train_model(self, model, tokenizer, dataset: Dataset, target_style: Dict[str, float]) -> Dict[str, Any]:
        """Train the model using GRPO"""
        print("Starting training...")

        # Calculate sequence lengths
        sample_prompt = tokenizer.apply_chat_template(
            dataset[0]["prompt"], tokenize=False, add_generation_prompt=True
        )
        max_prompt_length = len(tokenizer(sample_prompt)["input_ids"]) + 10
        max_completion_length = self.config.max_seq_length - max_prompt_length

        # Setup reward functions
        reward_funcs = NovelSpecificRewards(target_style, "Novel")

        # Training configuration
        training_args = GRPOConfig(
            temperature=self.config.temperature,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_generations=2,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            logging_steps=5,
            save_steps=self.config.max_steps,
            output_dir="temp_training",
            report_to="none",
        )

        # Train
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                reward_funcs.style_fidelity_reward,
                reward_funcs.narrative_quality_reward,
                reward_funcs.creativity_within_style_reward,
            ],
            args=training_args,
            train_dataset=dataset,
        )

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        return {
            "training_time": training_time,
            "max_steps": self.config.max_steps,
            "dataset_size": len(dataset)
        }

    def _generate_samples(self, model, tokenizer, novel_title: str) -> List[Dict[str, str]]:
        """Generate sample stories for evaluation"""
        test_prompts = [
            "Write a mysterious scene with rich descriptions:",
            "Create a dialogue between two characters:",
            "Describe a dramatic moment:",
            "Write about a character's inner thoughts:"
        ]

        samples = []

        for prompt in test_prompts:
            chat_prompt = [
                {"role": "system", "content": f"Write in the style of {novel_title}. Be creative and descriptive."},
                {"role": "user", "content": prompt}
            ]

            text = tokenizer.apply_chat_template(
                chat_prompt, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(text):].strip()

            samples.append({
                "prompt": prompt,
                "generated": generated
            })

        return samples

    def run_comparative_study(self) -> Dict[str, Any]:
        """Run experiments on all novels and compare results"""
        # Check for both old structure (txt files) and new structure (directories)
        novel_files = list(self.novels_dir.glob("*.txt"))
        novel_dirs = [d for d in self.novels_dir.iterdir() if d.is_dir()]

        novels_to_process = []

        # Prioritize directories (new structure) over files
        if novel_dirs:
            novels_to_process = novel_dirs
            print(f"Found {len(novel_dirs)} novel directories for experimentation")
        elif novel_files:
            novels_to_process = novel_files
            print(f"Found {len(novel_files)} novel files for experimentation")
        else:
            print(f"No novels found in {self.novels_dir}.")
            print("Run the novel processor first: python simple_novel_processor.py")
            return {}

        all_results = {}

        for novel_path in novels_to_process:
            try:
                results = self.run_experiment(novel_path)
                key = novel_path.name if novel_path.is_dir() else novel_path.stem
                all_results[key] = results
            except Exception as e:
                print(f"Error processing {novel_path.name}: {e}")

        # Generate comparative analysis
        comparison = self._analyze_results(all_results)

        # Save comparative study
        with open(self.experiments_dir / "comparative_study.json", 'w') as f:
            json.dump({
                "individual_results": all_results,
                "comparative_analysis": comparison
            }, f, indent=2)

        return {
            "results": all_results,
            "analysis": comparison
        }

    def _analyze_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare results across novels"""
        analysis = {
            "dataset_size_impact": {},
            "style_complexity_impact": {},
            "training_efficiency": {}
        }

        # Extract metrics
        word_counts = []
        chunk_counts = []
        training_times = []

        for novel_name, results in all_results.items():
            if results.get("training_results"):
                word_count = results["novel_info"]["word_count"]
                chunk_count = results["novel_info"]["chunk_count"]
                training_time = results["training_results"]["training_time"]

                word_counts.append(word_count)
                chunk_counts.append(chunk_count)
                training_times.append(training_time)

                analysis["dataset_size_impact"][novel_name] = {
                    "word_count": word_count,
                    "chunk_count": chunk_count,
                    "training_time": training_time
                }

        # Calculate correlations and insights
        if len(word_counts) > 1:
            analysis["insights"] = {
                "avg_word_count": statistics.mean(word_counts),
                "avg_training_time": statistics.mean(training_times),
                "word_count_range": (min(word_counts), max(word_counts)),
                "training_time_range": (min(training_times), max(training_times))
            }

        return analysis

def main():
    """Main function"""
    print("Single Novel Training Experiment")
    print("=" * 50)

    # Setup
    config = ExperimentConfig()
    experiment = SingleNovelExperiment(config)

    # Create directories
    Path(config.novels_dir).mkdir(exist_ok=True)

    # Check for novels
    novel_files = list(Path(config.novels_dir).glob("*.txt"))

    if not novel_files:
        print(f"""
Setup Instructions:
1. Create a '{config.novels_dir}' directory
2. Add individual novel .txt files (e.g., from Project Gutenberg)
3. Examples: alice_in_wonderland.txt, dracula.txt, pride_and_prejudice.txt
4. Each novel will be trained as a separate experiment

This tests how dataset size (single novel) affects model performance
using the GPT-OSS RL infrastructure.
""")
        return

    print(f"Found {len(novel_files)} novels ready for experimentation")

    if UNSLOTH_AVAILABLE:
        # Run comparative study
        results = experiment.run_comparative_study()

        print("\nExperiment Summary:")
        print(f"Processed {len(results['results'])} novels")

        if results['analysis'].get('insights'):
            insights = results['analysis']['insights']
            print(f"Average novel size: {insights['avg_word_count']:,.0f} words")
            print(f"Average training time: {insights['avg_training_time']:.1f} seconds")

        print(f"\nResults saved to: {config.experiments_dir}")

    else:
        print("Install Unsloth to run experiments: pip install unsloth")

if __name__ == "__main__":
    main()