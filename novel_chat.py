#!/usr/bin/env python3
"""
Model Tea - Novel Generation Chat Interface
Copyright © ChaiQ LLC

Interactive chat app for generating novels using trained single-novel models
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional
import json
import argparse

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Unsloth not available. Install with: pip install unsloth")
    UNSLOTH_AVAILABLE = False

class NovelChatBot:
    """Interactive chat interface for novel generation"""

    def __init__(self, model_path: str, novel_info: Dict = None):
        self.model_path = Path(model_path)
        self.novel_info = novel_info or {}
        self.model = None
        self.tokenizer = None
        self.conversation_history = []

    def load_model(self):
        """Load the trained model"""
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth required for model loading")

        print(f"Loading model from {self.model_path}...")

        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=2048,
                load_in_4bit=True,
                offload_embedding=True,
            )
            FastLanguageModel.for_inference(self.model)
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Note: Model path should point to a trained model directory")
            return False

        return True

    def generate_response(self, user_prompt: str, temperature: float = 0.8, max_tokens: int = 400) -> str:
        """Generate a novel excerpt based on user prompt"""
        if not self.model:
            return "Error: Model not loaded"

        # Create system message based on novel info
        novel_title = self.novel_info.get('title', 'classic literature')
        style_info = self._format_style_info()

        system_message = f"""You are a creative writer trained on {novel_title}.
Generate compelling, original stories that capture the essence and style of this work.
{style_info}
Write vivid, engaging prose with rich character development and atmospheric descriptions."""

        # Build conversation
        messages = [{"role": "system", "content": system_message}]

        # Add conversation history (last 3 exchanges to maintain context)
        for msg in self.conversation_history[-6:]:  # Last 3 user + 3 assistant messages
            messages.append(msg)

        # Add current prompt
        messages.append({"role": "user", "content": user_prompt})

        # Generate
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(text):].strip()

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.conversation_history.append({"role": "assistant", "content": generated})

        return generated

    def _format_style_info(self) -> str:
        """Format style information for system prompt"""
        if not self.novel_info.get('style_profile'):
            return ""

        style = self.novel_info['style_profile']
        style_desc = []

        if style.get('avg_sentence_length', 0) > 20:
            style_desc.append("Use longer, flowing sentences")
        elif style.get('avg_sentence_length', 0) < 15:
            style_desc.append("Use shorter, punchy sentences")

        if style.get('dialogue_ratio', 0) > 2:
            style_desc.append("Include dialogue between characters")

        if style.get('descriptive_density', 0) > 0.05:
            style_desc.append("Use rich, descriptive language with vivid imagery")

        if style_desc:
            return "Style guidance: " + "; ".join(style_desc) + "."

        return ""

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")

    def save_conversation(self, filename: str):
        """Save conversation to file"""
        conversation_data = {
            "novel_info": self.novel_info,
            "conversation": self.conversation_history
        }

        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)

        print(f"Conversation saved to {filename}")

    def show_model_info(self):
        """Display information about the loaded model"""
        if self.novel_info:
            print(f"\nModel Information:")
            print(f"Novel: {self.novel_info.get('title', 'Unknown')}")
            print(f"Word count: {self.novel_info.get('word_count', 'Unknown'):,}")

            if 'style_profile' in self.novel_info:
                style = self.novel_info['style_profile']
                print(f"Style Profile:")
                for key, value in style.items():
                    print(f"  {key}: {value:.3f}")
        else:
            print("No novel information available")

def load_experiment_model(experiment_dir: Path) -> tuple[str, Dict]:
    """Load model and info from experiment directory"""
    model_path = experiment_dir / "trained_model"
    info_path = experiment_dir / "novel_analysis.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found in {experiment_dir}")

    novel_info = {}
    if info_path.exists():
        with open(info_path, 'r') as f:
            novel_info = json.load(f)

    return str(model_path), novel_info

def interactive_chat(chatbot: NovelChatBot):
    """Run interactive chat session"""
    print("\n" + "="*60)
    print("Novel Generation Chat Interface")
    print("="*60)

    chatbot.show_model_info()

    print(f"""
Commands:
  /help     - Show this help
  /info     - Show model information
  /clear    - Clear conversation history
  /save     - Save conversation to file
  /temp X   - Set temperature (0.1-2.0)
  /tokens X - Set max tokens (50-800)
  /quit     - Exit

Type your storytelling prompts and I'll generate novel excerpts!
""")

    temperature = 0.8
    max_tokens = 400

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()

                if command == '/quit':
                    break
                elif command == '/help':
                    print("""
Available commands:
  /help     - Show this help
  /info     - Show model information
  /clear    - Clear conversation history
  /save     - Save conversation to file
  /temp X   - Set temperature (0.1-2.0)
  /tokens X - Set max tokens (50-800)
  /quit     - Exit
                    """)
                elif command == '/info':
                    chatbot.show_model_info()
                elif command == '/clear':
                    chatbot.clear_history()
                elif command.startswith('/temp '):
                    try:
                        temp = float(command.split()[1])
                        if 0.1 <= temp <= 2.0:
                            temperature = temp
                            print(f"Temperature set to {temperature}")
                        else:
                            print("Temperature must be between 0.1 and 2.0")
                    except (IndexError, ValueError):
                        print("Usage: /temp <number>")
                elif command.startswith('/tokens '):
                    try:
                        tokens = int(command.split()[1])
                        if 50 <= tokens <= 800:
                            max_tokens = tokens
                            print(f"Max tokens set to {max_tokens}")
                        else:
                            print("Max tokens must be between 50 and 800")
                    except (IndexError, ValueError):
                        print("Usage: /tokens <number>")
                elif command == '/save':
                    filename = f"conversation_{chatbot.novel_info.get('title', 'unknown').replace(' ', '_').lower()}.json"
                    chatbot.save_conversation(filename)
                else:
                    print("Unknown command. Type /help for available commands.")

                continue

            # Generate response
            print("Generating...")
            response = chatbot.generate_response(user_input, temperature, max_tokens)
            print(f"\n{response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Novel Generation Chat Interface")
    parser.add_argument("--model", help="Path to trained model directory")
    parser.add_argument("--experiment", help="Path to experiment directory")
    parser.add_argument("--list", action="store_true", help="List available experiments")

    args = parser.parse_args()

    experiments_dir = Path("experiments")

    # List available experiments
    if args.list:
        if experiments_dir.exists():
            experiments = [d for d in experiments_dir.iterdir() if d.is_dir()]
            if experiments:
                print("Available experiments:")
                for exp in experiments:
                    print(f"  {exp.name}")
            else:
                print("No experiments found.")
        else:
            print("No experiments directory found.")
        return

    # Determine model path
    model_path = None
    novel_info = {}

    if args.model:
        model_path = args.model
    elif args.experiment:
        exp_path = Path(args.experiment)
        if not exp_path.is_absolute():
            exp_path = experiments_dir / exp_path
        try:
            model_path, novel_info = load_experiment_model(exp_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        # Try to find experiments automatically
        if experiments_dir.exists():
            experiments = [d for d in experiments_dir.iterdir() if d.is_dir() and (d / "trained_model").exists()]
            if experiments:
                print("Available trained models:")
                for i, exp in enumerate(experiments):
                    print(f"  {i+1}. {exp.name}")

                try:
                    choice = int(input("\nSelect experiment (number): ")) - 1
                    if 0 <= choice < len(experiments):
                        model_path, novel_info = load_experiment_model(experiments[choice])
                    else:
                        print("Invalid choice.")
                        return
                except (ValueError, KeyboardInterrupt):
                    print("Invalid input.")
                    return
            else:
                print("No trained models found. Run single_novel_trainer.py first.")
                return
        else:
            print("No experiments found. Run single_novel_trainer.py first.")
            return

    if not model_path:
        print("No model specified. Use --model, --experiment, or select from available experiments.")
        return

    # Check if Unsloth is available
    if not UNSLOTH_AVAILABLE:
        print("Unsloth not available. Install with: pip install unsloth")
        return

    # Initialize chatbot
    chatbot = NovelChatBot(model_path, novel_info)

    if not chatbot.load_model():
        return

    # Run interactive chat
    interactive_chat(chatbot)

if __name__ == "__main__":
    main()