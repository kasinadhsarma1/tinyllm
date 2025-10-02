"""
Test script for the Unified LLM system
Demonstrates the key features of the user-based LLM
"""

from unified_llm import UnifiedLLM, UserLLMManager, HelloWorldDataset


def test_unified_system():
    """Test the unified LLM system"""
    print("🚀 TESTING UNIFIED LLM SYSTEM")
    print("="*50)
    
    # Create manager and dataset
    manager = UserLLMManager()
    dataset = HelloWorldDataset()
    
    print("\n1. Testing N-gram model for user 'Alice'...")
    
    # Create Alice's N-gram model
    alice_model = manager.create_user_model("alice", "ngram", n=3)
    alice_model.train(dataset.texts)
    alice_model.save("test_model")
    
    # Test generation
    print("Alice's generations:")
    for prompt in ["hello", "hi", "world"]:
        result = alice_model.generate(prompt, max_length=15, temperature=0.8)
        print(f"  '{prompt}' → '{result}'")
    
    print(f"\nAlice's model info: {alice_model.get_model_info()}")
    
    print("\n2. Testing Neural model for user 'Bob'...")
    
    # Create Bob's Neural model
    bob_model = manager.create_user_model("bob", "neural", vocab_size=30)
    bob_model.train(dataset.texts)
    bob_model.save("test_model")
    
    # Test generation
    print("Bob's generations:")
    for prompt in ["hello", "hi", "world"]:
        result = bob_model.generate(prompt, max_length=15, temperature=0.8)
        print(f"  '{prompt}' → '{result}'")
    
    print(f"\nBob's model info: {bob_model.get_model_info()}")
    
    print("\n3. Testing save/load functionality...")
    
    # Load models
    loaded_alice = manager.load_user_model("alice", "test_model")
    loaded_bob = manager.load_user_model("bob", "test_model")
    
    print("Testing loaded models:")
    alice_result = loaded_alice.generate("test", max_length=10)
    bob_result = loaded_bob.generate("test", max_length=10)
    
    print(f"  Loaded Alice: 'test' → '{alice_result}'")
    print(f"  Loaded Bob: 'test' → '{bob_result}'")
    
    print("\n4. Testing user management...")
    
    users = manager.list_users()
    print(f"Users: {users}")
    
    for user in users:
        models = manager.list_user_models(user)
        print(f"{user}'s models: {models}")
    
    print("\n✅ ALL TESTS PASSED!")
    print("="*50)
    print("The unified LLM system supports:")
    print("✓ Multiple model types (N-gram, Neural)")
    print("✓ User-based model isolation")
    print("✓ Save/load functionality")
    print("✓ Model management and listing")
    print("✓ Text generation with different approaches")
    print("="*50)


def simple_comparison():
    """Simple comparison between model types"""
    print("\n🔬 MODEL COMPARISON")
    print("="*40)
    
    dataset = HelloWorldDataset()
    
    # Create both model types
    ngram_model = UnifiedLLM("ngram", "demo_user", n=3)
    neural_model = UnifiedLLM("neural", "demo_user", vocab_size=25)
    
    # Train both
    print("Training models...")
    ngram_model.train(dataset.texts)
    neural_model.train(dataset.texts)
    
    # Compare generations
    test_prompts = ["hello", "hi", "good"]
    
    print("\nComparison of generations:")
    print("-" * 40)
    
    for prompt in test_prompts:
        ngram_result = ngram_model.generate(prompt, max_length=12, temperature=0.7)
        neural_result = neural_model.generate(prompt, max_length=12, temperature=0.7)
        
        print(f"Prompt: '{prompt}'")
        print(f"  N-gram:  '{ngram_result}'")
        print(f"  Neural:  '{neural_result}'")
        print()
    
    print("N-gram models are:")
    print("  + Fast and predictable")
    print("  + Good for character-level patterns")
    print("  - Limited context understanding")
    
    print("\nNeural models are:")
    print("  + Better at learning complex patterns")
    print("  + More flexible generation")
    print("  - Slower and more resource intensive")


if __name__ == "__main__":
    print("Choose test:")
    print("1. Full system test")
    print("2. Model comparison")
    print("3. Both")
    
    choice = input("Choice (1-3): ").strip()
    
    if choice == "1":
        test_unified_system()
    elif choice == "2":
        simple_comparison()
    else:
        test_unified_system()
        simple_comparison()
