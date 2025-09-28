#!/usr/bin/env python3
"""
Complete Memory + Model Demo
Shows the full pipeline: Prompt -> Memory Activation -> Model Generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from episodic_memory_system import EpisodicMemorySystem, MemoryConfig
from pathlib import Path

def load_trained_model_and_memory():
    """Load both the trained model and its memory system"""
    model_path = Path("iterative_models/call_of_cthulhu/final")

    print("Loading trained model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading memory system...")
    memory_system = EpisodicMemorySystem(MemoryConfig())
    success = memory_system.load_memory_for_model("call_of_cthulhu")

    if not success:
        raise Exception("Failed to load memory system")

    return model, tokenizer, memory_system

def generate_with_memory(model, tokenizer, memory_system, prompt):
    """Generate text using both the model and activated memories"""

    print(f"\n=== COMPLETE PIPELINE DEMO ===")
    print(f"Input: '{prompt}'")
    print("-" * 50)

    # Step 1: Activate memories
    print("STEP 1: Memory Activation")
    activated_memories, stats = memory_system.activate_memories("call_of_cthulhu", prompt)
    print(f"Memories activated: {len(activated_memories)}")

    if activated_memories:
        print("Activated memory content:")
        for i, memory in enumerate(activated_memories[:3], 1):
            print(f"  {i}. [{memory.memory_type}] {memory.content[:80]}...")

    # Step 2: Create enhanced prompt with memory context
    print("\nSTEP 2: Enhanced Prompt Creation")
    memory_context = ""
    if activated_memories:
        memory_texts = [mem.content for mem in activated_memories[:3]]
        memory_context = " ".join(memory_texts)
        enhanced_prompt = f"{memory_context} {prompt}"
        print(f"Enhanced prompt length: {len(enhanced_prompt)} chars")
        print(f"Memory context: {memory_context[:100]}...")
    else:
        enhanced_prompt = prompt
        print("No memory context added")

    # Step 3: Generate with the model
    print("\nSTEP 3: Model Generation")
    model.eval()
    inputs = tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the enhanced prompt to show only new generation
    if memory_context:
        # Remove both memory context and original prompt
        new_text = generated[len(enhanced_prompt):].strip()
    else:
        new_text = generated[len(prompt):].strip()

    print(f"Generated text: {new_text}")

    # Step 4: Compare with no-memory generation
    print("\nSTEP 4: Comparison (No Memory)")
    inputs_no_memory = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs_no_memory = model.generate(
            inputs_no_memory.input_ids,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    generated_no_memory = tokenizer.decode(outputs_no_memory[0], skip_special_tokens=True)
    new_text_no_memory = generated_no_memory[len(prompt):].strip()

    print(f"No-memory generation: {new_text_no_memory}")

    print("\n" + "="*60)
    return new_text, new_text_no_memory, activated_memories

def main():
    """Run complete demo"""
    try:
        model, tokenizer, memory_system = load_trained_model_and_memory()

        test_prompts = [
            "The professor discovered",
            "In the ancient city",
            "The cultists were",
            "Cthulhu appeared when"
        ]

        for prompt in test_prompts:
            with_memory, without_memory, memories = generate_with_memory(
                model, tokenizer, memory_system, prompt
            )

            print(f"\nSUMMARY for '{prompt}':")
            print(f"  Memories used: {len(memories)}")
            print(f"  With memory: {with_memory[:60]}...")
            print(f"  Without memory: {without_memory[:60]}...")
            print()

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()