#!/usr/bin/env python3
"""
Test specifically for professor-related memories to verify accuracy
"""

from episodic_memory_system import EpisodicMemorySystem, MemoryConfig

def test_professor_memories():
    """Test what the memory system actually knows about professors"""

    print("=== Testing Professor-Related Memory Retrieval ===")
    print("Expected: George Gammell Angell, William Channing Webb")
    print("=" * 60)

    # Load memory system
    memory_system = EpisodicMemorySystem(MemoryConfig())
    success = memory_system.load_memory_for_model("call_of_cthulhu")

    if not success:
        print("Failed to load memory system")
        return

    memories = memory_system.model_memories["call_of_cthulhu"]
    print(f"Total memories loaded: {len(memories)}")

    # Test professor-related queries
    test_queries = [
        "Tell me about Professor Angell",
        "What did Professor Webb discover?",
        "Tell me about the professor",
        "Who is the narrator's uncle?",
        "What about the anthropologist professor?"
    ]

    for query in test_queries:
        print(f"\n**Query: '{query}'**")
        print("-" * 40)

        activated_memories, stats = memory_system.activate_memories("call_of_cthulhu", query)

        print(f"Memories activated: {len(activated_memories)}")

        if activated_memories:
            for i, memory in enumerate(activated_memories, 1):
                print(f"\n  Memory {i} ({memory.memory_type}):")
                print(f"    Content: {memory.content}")
                print(f"    Keywords: {memory.keywords}")

                # Check for actual character names
                content_lower = memory.content.lower()
                if 'angell' in content_lower:
                    print("    FOUND: George Gammell Angell")
                if 'webb' in content_lower:
                    print("    FOUND: William Channing Webb")
                if 'olaf' in content_lower:
                    print("    ERROR: Olaf (NOT in original story!)")
        else:
            print("  No memories activated")

    # Search all memories for professor references
    print(f"\n**Full Memory Scan for Professor References**")
    print("-" * 50)

    professor_memories = []
    for memory in memories:
        content_lower = memory.content.lower()
        if 'professor' in content_lower or 'angell' in content_lower or 'webb' in content_lower:
            professor_memories.append(memory)

    print(f"Found {len(professor_memories)} memories containing professor references:")

    for i, memory in enumerate(professor_memories[:10], 1):  # Show first 10
        print(f"\n{i}. [{memory.memory_type}] {memory.content[:120]}...")

        # Check for character names
        content_lower = memory.content.lower()
        if 'angell' in content_lower:
            print("    FOUND: Angell")
        if 'webb' in content_lower:
            print("    FOUND: Webb")
        if 'olaf' in content_lower:
            print("    ERROR: Olaf (ERROR!)")

if __name__ == "__main__":
    test_professor_memories()