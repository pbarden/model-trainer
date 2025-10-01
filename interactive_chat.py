#!/usr/bin/env python3
"""
Interactive Chat with Trained Models
Allows you to select a trained model and chat with it in the console
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


class InteractiveChat:
    """Interactive console chat with trained models"""

    def __init__(self, models_dir: str = "iterative_models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Command registry
        self.commands = {
            '/help': self.show_help,
            '/h': self.show_help,
            '/commands': self.show_commands,
            '/settings': self.show_settings_menu,
            '/set': self.show_settings_menu,
            '/models': self.show_models_list,
            '/switch': self.switch_model,
            '/info': self.show_model_info,
            '/clear': self.clear_history,
            '/history': self.show_history,
            '/export': self.export_conversation,
            '/status': self.show_status,
            '/quit': None,
            '/exit': None,
            '/q': None
        }

    def list_available_models(self, show_all: bool = False) -> list:
        """List all available trained models"""
        if not self.models_dir.exists():
            print(f"Models directory '{self.models_dir}' not found!")
            return []

        models = []
        final_models = []

        for item in self.models_dir.iterdir():
            if item.is_dir():
                # Check if it contains model files directly
                if (item / "config.json").exists() or (item / "pytorch_model.bin").exists() or (item / "model.safetensors").exists():
                    models.append(str(item.relative_to(self.models_dir)))
                else:
                    # Check subdirectories (like final/, iteration_N/)
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            if (subitem / "config.json").exists() or (subitem / "pytorch_model.bin").exists() or (subitem / "model.safetensors").exists():
                                rel_path = str(subitem.relative_to(self.models_dir))
                                if subitem.name == "final":
                                    final_models.append(rel_path)
                                elif show_all:
                                    models.append(rel_path)

        # Prioritize final models, then others
        all_models = sorted(final_models) + sorted(models)
        return all_models if all_models else []

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        model_path = self.models_dir / model_name
        info = {
            "name": model_name,
            "path": str(model_path),
            "exists": model_path.exists()
        }

        if model_path.exists():
            # Get model size
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            info["size_mb"] = round(total_size / (1024 * 1024), 2)

            # Check for config
            config_path = model_path / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    info["model_type"] = config.get("model_type", "unknown")
                    info["vocab_size"] = config.get("vocab_size", "unknown")
                    info["hidden_size"] = config.get("hidden_size", "unknown")
                except:
                    pass

        return info

    def select_model(self) -> Optional[str]:
        """Interactive model selection"""
        models = self.list_available_models()

        if not models:
            print("\n" + "!" * 60)
            print("No trained models found!")
            print(f"Models directory: {self.models_dir}")
            print("!" * 60)
            return None

        print("\n" + "=" * 60)
        print("AVAILABLE MODELS")
        print("=" * 60)
        for idx, model_name in enumerate(models, 1):
            info = self.get_model_info(model_name)
            size_info = f"({info.get('size_mb', '?')} MB)" if 'size_mb' in info else ""
            # Clean display name
            display_name = model_name.replace("\\", "/")
            print(f"  {idx}. {display_name} {size_info}")
        print("=" * 60)
        print("\nTip: Type 'help' or '?' after loading for available commands")
        print("Note: Showing final models only. Use '/models all' in chat to see iterations.")

        while True:
            try:
                choice = input("\nSelect model [1-{}], 'list' for details, or 'q' to quit: ".format(len(models))).strip()

                if choice.lower() in ['q', 'quit', 'exit']:
                    return None

                if choice.lower() in ['?', 'help']:
                    self.show_startup_help()
                    continue

                if choice.lower() in ['list', 'ls', 'models']:
                    # Show detailed model list
                    print("\n" + "=" * 60)
                    print("DETAILED MODEL LIST")
                    print("=" * 60)
                    for idx, model_name in enumerate(models, 1):
                        info = self.get_model_info(model_name)
                        display_name = model_name.replace("\\", "/")
                        print(f"\n{idx}. {display_name}")
                        print(f"   Size: {info.get('size_mb', '?')} MB")
                        print(f"   Type: {info.get('model_type', 'unknown')}")
                        print(f"   Path: {info.get('path', 'unknown')}")
                    print("=" * 60)
                    continue

                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(models):
                        return models[choice_idx]
                    else:
                        print(f"[!] Please enter a number between 1 and {len(models)}")
                except ValueError:
                    print("[!] Invalid input. Enter a number, 'list' for details, or 'q' to quit.")
            except KeyboardInterrupt:
                print("\n")
                return None

    def load_model(self, model_name: str) -> bool:
        """Load the selected model"""
        model_path = self.models_dir / model_name

        print(f"\nLoading model '{model_name}'...")
        print(f"Device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            ).to(self.device)

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model_name = model_name
            print("Model loaded successfully!\n")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    def generate_response(self, prompt: str, max_length: int = 250,
                         temperature: float = 0.5, top_p: float = 0.85,
                         top_k: int = 30, repetition_penalty: float = 1.4) -> str:
        """Generate a response from the model"""
        if self.model is None or self.tokenizer is None:
            return "Error: No model loaded"

        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to(self.device)

        # Generate with attention mask
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        # Decode the response (only the new tokens)
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()

    def show_startup_help(self):
        """Show help at startup"""
        print("\n" + "=" * 60)
        print("QUICK START")
        print("=" * 60)
        print("Enter a number to select a model, or:")
        print("  help, ? - Show this help")
        print("  q, quit - Exit")
        print("=" * 60)

    def show_help(self, settings=None):
        """Show comprehensive help menu"""
        print("\n" + "=" * 60)
        print("INTERACTIVE CHAT - HELP MENU")
        print("=" * 60)
        print("\nAVAILABLE COMMANDS:")
        print("  /help, /h        - Show this help menu")
        print("  /commands        - List all available commands")
        print("  /status          - Show current session status")
        print("")
        print("MODEL MANAGEMENT:")
        print("  /models          - List all available models")
        print("  /switch          - Switch to a different model")
        print("  /info            - Show current model information")
        print("")
        print("GENERATION SETTINGS:")
        print("  /settings, /set  - Adjust generation parameters")
        print("")
        print("CONVERSATION:")
        print("  /clear           - Clear conversation history")
        print("  /history         - Show current conversation history")
        print("  /export          - Export conversation to file")
        print("")
        print("EXIT:")
        print("  /quit, /exit, /q - Exit the chat")
        print("=" * 60 + "\n")

    def show_commands(self, settings=None):
        """Show quick command list"""
        print("\n" + "=" * 60)
        print("AVAILABLE COMMANDS")
        print("=" * 60)
        for cmd in sorted(self.commands.keys()):
            print(f"  {cmd}")
        print("=" * 60 + "\n")

    def show_models_list(self, settings=None, show_all=False):
        """Show list of available models"""
        models = self.list_available_models(show_all=show_all)
        print("\n" + "=" * 60)
        print("AVAILABLE MODELS" + (" (ALL)" if show_all else " (FINAL ONLY)"))
        print("=" * 60)
        for idx, model_name in enumerate(models, 1):
            info = self.get_model_info(model_name)
            size_info = f"{info.get('size_mb', '?')} MB"
            current = " [CURRENT]" if model_name == self.model_name else ""
            display_name = model_name.replace("\\", "/")
            print(f"  {idx}. {display_name} - {size_info}{current}")
        print("=" * 60)
        if not show_all:
            print("Tip: Use '/models all' to show iteration checkpoints too")
        print()

    def show_model_info(self, settings=None):
        """Show detailed information about current model"""
        if not self.model_name:
            print("\n[!] No model currently loaded\n")
            return

        info = self.get_model_info(self.model_name)
        print("\n" + "=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)
        print(f"  Name: {info['name']}")
        print(f"  Path: {info['path']}")
        print(f"  Size: {info.get('size_mb', '?')} MB")
        print(f"  Type: {info.get('model_type', 'unknown')}")
        print(f"  Vocabulary Size: {info.get('vocab_size', 'unknown')}")
        print(f"  Hidden Size: {info.get('hidden_size', 'unknown')}")
        print(f"  Device: {self.device}")
        print("=" * 60 + "\n")

    def show_status(self, settings=None):
        """Show current session status"""
        print("\n" + "=" * 60)
        print("SESSION STATUS")
        print("=" * 60)
        print(f"  Current Model: {self.model_name or 'None'}")
        print(f"  Device: {self.device}")
        if settings:
            print(f"\n  Generation Settings:")
            print(f"    Max Length: {settings['max_length']}")
            print(f"    Temperature: {settings['temperature']}")
            print(f"    Top-p: {settings['top_p']}")
            print(f"    Top-k: {settings['top_k']}")
            print(f"    Repetition Penalty: {settings['repetition_penalty']}")
        print("=" * 60 + "\n")

    def clear_history(self, settings=None):
        """Clear conversation history"""
        print("\n[Conversation history cleared]\n")
        return True  # Signal to clear history

    def show_history(self, settings=None):
        """Show conversation history - to be implemented in chat_loop"""
        pass  # Will be handled in chat_loop

    def export_conversation(self, settings=None):
        """Export conversation - to be implemented in chat_loop"""
        pass  # Will be handled in chat_loop

    def switch_model(self, settings=None):
        """Switch to a different model"""
        print("\n[Switching models...]")
        new_model = self.select_model()
        if new_model:
            return new_model  # Signal to reload
        return None

    def chat_loop(self):
        """Main interactive chat loop"""
        print("\n" + "=" * 60)
        print("INTERACTIVE CHAT")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print("\nType '/help' or '/h' for available commands")
        print("=" * 60 + "\n")

        # Generation settings - optimized for coherent, focused output
        settings = {
            "max_length": 250,
            "temperature": 0.5,
            "top_p": 0.85,
            "top_k": 30,
            "repetition_penalty": 1.4
        }

        conversation_history = ""
        conversation_log = []  # For history and export

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("\nGoodbye!")
                    break

                elif user_input.lower() == '/clear':
                    conversation_history = ""
                    conversation_log = []
                    print("\n[Conversation history cleared]\n")
                    continue

                elif user_input.lower() == '/history':
                    if not conversation_log:
                        print("\n[No conversation history]\n")
                    else:
                        print("\n" + "=" * 60)
                        print("CONVERSATION HISTORY")
                        print("=" * 60)
                        for entry in conversation_log:
                            print(f"\nYou: {entry['user']}")
                            print(f"Assistant: {entry['assistant']}")
                        print("=" * 60 + "\n")
                    continue

                elif user_input.lower() == '/export':
                    if not conversation_log:
                        print("\n[No conversation to export]\n")
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"conversation_{self.model_name}_{timestamp}.txt"
                        with open(filename, 'w') as f:
                            f.write(f"Model: {self.model_name}\n")
                            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("=" * 60 + "\n\n")
                            for entry in conversation_log:
                                f.write(f"You: {entry['user']}\n")
                                f.write(f"Assistant: {entry['assistant']}\n\n")
                        print(f"\n[Conversation exported to: {filename}]\n")
                    continue

                elif user_input.lower() == '/switch':
                    new_model = self.switch_model()
                    if new_model:
                        if self.load_model(new_model):
                            conversation_history = ""
                            conversation_log = []
                            print("[Model switched successfully]\n")
                    continue

                elif user_input.lower() == '/models all':
                    self.show_models_list(settings, show_all=True)
                    continue

                elif user_input.lower() in self.commands:
                    cmd_func = self.commands[user_input.lower()]
                    if cmd_func:
                        cmd_func(settings)
                    continue

                # Build prompt with conversation history
                if conversation_history:
                    prompt = conversation_history + "\nYou: " + user_input + "\nAssistant:"
                else:
                    prompt = "You: " + user_input + "\nAssistant:"

                # Generate response
                print("\nAssistant: ", end="", flush=True)
                response = self.generate_response(
                    prompt,
                    max_length=settings["max_length"],
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    top_k=settings["top_k"],
                    repetition_penalty=settings["repetition_penalty"]
                )
                print(response + "\n")

                # Log the exchange
                conversation_log.append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": datetime.now().isoformat()
                })

                # Update conversation history
                conversation_history = prompt + " " + response

                # Keep history manageable (last 3 exchanges)
                exchanges = conversation_history.split("\nYou:")
                if len(exchanges) > 4:
                    conversation_history = "\nYou:".join(exchanges[-4:])

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}\n")

    def show_settings_menu(self, settings: dict):
        """Show and adjust generation settings"""
        print("\n" + "=" * 60)
        print("GENERATION SETTINGS")
        print("=" * 60)
        print(f"  1. Max Length: {settings['max_length']} tokens")
        print(f"     (Controls response length)")
        print(f"  2. Temperature: {settings['temperature']}")
        print(f"     (Randomness: lower=focused, higher=creative)")
        print(f"  3. Top-p: {settings['top_p']}")
        print(f"     (Nucleus sampling: lower=focused, higher=diverse)")
        print(f"  4. Top-k: {settings['top_k']}")
        print(f"     (Limits vocabulary per step: lower=focused, higher=diverse)")
        print(f"  5. Repetition Penalty: {settings['repetition_penalty']}")
        print(f"     (Discourages repetition: higher=less repetitive)")
        print("\n  6. Load Preset (coherent/balanced/creative)")
        print("=" * 60)
        print("\nEnter setting number to change, or press Enter to cancel")

        choice = input("> ").strip()
        if not choice:
            return

        try:
            choice_num = int(choice)
            if choice_num == 1:
                print("\nMax Length (50-500 tokens)")
                print("Recommended: 100-200 for coherent output, 150-300 for longer")
                val = int(input("New value: "))
                if 50 <= val <= 500:
                    settings['max_length'] = val
                    print(f"[Updated] Max length set to {val}")
                else:
                    print("[!] Value must be between 50 and 500")
            elif choice_num == 2:
                print("\nTemperature (0.1-2.0)")
                print("Recommended: 0.5-0.6=very focused, 0.7=focused, 0.8=balanced, 1.0=creative")
                print("For coherent output, try 0.6-0.7")
                val = float(input("New value: "))
                if 0.1 <= val <= 2.0:
                    settings['temperature'] = val
                    print(f"[Updated] Temperature set to {val}")
                else:
                    print("[!] Value must be between 0.1 and 2.0")
            elif choice_num == 3:
                print("\nTop-p (0.1-1.0)")
                print("Recommended: 0.85-0.90=focused, 0.92-0.95=diverse")
                print("Lower values reduce rambling")
                val = float(input("New value: "))
                if 0.1 <= val <= 1.0:
                    settings['top_p'] = val
                    print(f"[Updated] Top-p set to {val}")
                else:
                    print("[!] Value must be between 0.1 and 1.0")
            elif choice_num == 4:
                print("\nTop-k (10-200)")
                print("Recommended: 40-50=focused, 50-80=balanced, 100+=creative")
                print("Lower values prevent unlikely word choices")
                val = int(input("New value: "))
                if 10 <= val <= 200:
                    settings['top_k'] = val
                    print(f"[Updated] Top-k set to {val}")
                else:
                    print("[!] Value must be between 10 and 200")
            elif choice_num == 5:
                print("\nRepetition Penalty (1.0-2.0)")
                print("Recommended: 1.15-1.25=slight, 1.3-1.4=moderate, 1.5+=strong")
                print("Higher values prevent run-on sentences")
                val = float(input("New value: "))
                if 1.0 <= val <= 2.0:
                    settings['repetition_penalty'] = val
                    print(f"[Updated] Repetition penalty set to {val}")
                else:
                    print("[!] Value must be between 1.0 and 2.0")
            elif choice_num == 6:
                print("\nAvailable Presets:")
                print("  coherent  - Best for focused, readable output")
                print("  balanced  - Mix of coherence and creativity")
                print("  creative  - More varied but less predictable")
                preset = input("Select preset: ").strip().lower()
                if preset == "coherent":
                    settings.update({"max_length": 100, "temperature": 0.5, "top_p": 0.85, "top_k": 30, "repetition_penalty": 1.4})
                    print("[Updated] Loaded 'coherent' preset")
                elif preset == "balanced":
                    settings.update({"max_length": 150, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.2})
                    print("[Updated] Loaded 'balanced' preset")
                elif preset == "creative":
                    settings.update({"max_length": 200, "temperature": 0.9, "top_p": 0.95, "top_k": 80, "repetition_penalty": 1.15})
                    print("[Updated] Loaded 'creative' preset")
                else:
                    print("[!] Unknown preset")
            else:
                print("[!] Invalid setting number")
        except ValueError:
            print("[!] Invalid input. Settings unchanged.")
        except KeyboardInterrupt:
            print("\n[Cancelled]")

    def run(self):
        """Main entry point"""
        # Welcome screen
        print("\n" + "=" * 60)
        print(" MODEL CHAT - Interactive Console Chat System")
        print("=" * 60)
        print(" Chat with your trained language models")
        print(" Type '/help' after loading for full command list")
        print("=" * 60)

        # Select model
        model_name = self.select_model()
        if model_name is None:
            print("\nExiting...")
            return

        # Load model
        if not self.load_model(model_name):
            print("\n[ERROR] Failed to load model. Exiting...")
            return

        # Start chat
        self.chat_loop()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive chat with trained language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_chat.py
  python interactive_chat.py --models-dir custom_models

Commands available in chat:
  /help      - Show all available commands
  /models    - List available models
  /settings  - Adjust generation parameters
  /quit      - Exit chat
        """
    )
    parser.add_argument(
        '--models-dir',
        default='iterative_models',
        help='Directory containing trained models (default: iterative_models)'
    )

    args = parser.parse_args()

    chat = InteractiveChat(models_dir=args.models_dir)
    chat.run()


if __name__ == "__main__":
    main()
