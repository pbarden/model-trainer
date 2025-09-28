#!/usr/bin/env python3
"""
Generate a LinkedIn tagline using memory-enhanced model generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from episodic_memory_system import EpisodicMemorySystem, MemoryConfig
from pathlib import Path

def generate_tagline():
    """Generate professional tagline with Call of Cthulhu inspiration"""

    # Load model and memory
    model_path = Path("iterative_models/call_of_cthulhu/final")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    memory_system = EpisodicMemorySystem(MemoryConfig())
    memory_system.load_memory_for_model("call_of_cthulhu")

    # More natural prompts that avoid marketing speak
    tagline_prompts = [
        "The professor's research revealed that knowledge",
        "In the manuscript, I discovered that memory",
        "What the ancient texts taught us about learning",
        "The scholar found that intelligence",
        "After years of study, we learned that minds"
    ]

    print("=== AI-GENERATED LINKEDIN TAGLINES ===")
    print()

    for prompt in tagline_prompts:
        print(f"Prompt: '{prompt}'")

        # Activate memories
        activated_memories, _ = memory_system.activate_memories("call_of_cthulhu", prompt)

        # Create enhanced prompt
        if activated_memories:
            memory_context = " ".join([mem.content for mem in activated_memories[:2]])
            enhanced_prompt = f"{memory_context} {prompt}"
        else:
            enhanced_prompt = prompt

        # Generate
        model.eval()
        inputs = tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=400)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated[len(enhanced_prompt):].strip()

        print(f"Generated: {new_text}")
        print(f"Complete tagline: {prompt} {new_text}")
        print("-" * 60)
        print()

if __name__ == "__main__":
    generate_tagline()