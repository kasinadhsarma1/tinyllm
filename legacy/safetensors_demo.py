"""
SafeTensors Demo for Unified LLM System
Demonstrates SafeTensors integration and benefits
"""

from unified_llm import UnifiedLLM, UserLLMManager, HelloWorldDataset
import os
import time


def safetensors_demo():
    """Demonstrate SafeTensors functionality"""
    print("🔒 SAFETENSORS DEMO")
    print("="*50)
    
    manager = UserLLMManager()
    dataset = HelloWorldDataset()
    
    print("\n1. Creating Neural model with SafeTensors...")
    
    # Create and train neural model
    neural_model = manager.create_user_model("safetensors_user", "neural", vocab_size=40)
    neural_model.train(dataset.texts)
    
    # Save using SafeTensors
    print("\n📦 Saving model in SafeTensors format...")
    start_time = time.time()
    neural_model.save("safetensors_demo")
    save_time = time.time() - start_time
    
    print(f"⏱️  Save time: {save_time:.3f} seconds")
    
    print("\n2. Loading model from SafeTensors...")
    
    # Load using SafeTensors
    start_time = time.time()
    loaded_model = manager.load_user_model("safetensors_user", "safetensors_demo")
    load_time = time.time() - start_time
    
    print(f"⏱️  Load time: {load_time:.3f} seconds")
    
    print("\n3. Testing model functionality...")
    
    # Test generation
    test_prompts = ["hello", "hi", "good"]
    print("\n🎯 Generated text:")
    for prompt in test_prompts:
        result = loaded_model.generate(prompt, max_length=15, temperature=0.8)
        print(f"  '{prompt}' → '{result}'")
    
    print("\n4. File analysis...")
    
    # Check file sizes
    models_dir = f"models/safetensors_user"
    safetensors_file = os.path.join(models_dir, "safetensors_demo_neural.safetensors")
    vocab_file = os.path.join(models_dir, "safetensors_demo_vocab.json")
    
    if os.path.exists(safetensors_file):
        st_size = os.path.getsize(safetensors_file)
        vocab_size = os.path.getsize(vocab_file)
        print(f"📄 SafeTensors file: {st_size/1024:.1f} KB")
        print(f"📄 Vocabulary file: {vocab_size/1024:.1f} KB")
        print(f"📄 Total size: {(st_size + vocab_size)/1024:.1f} KB")
    
    print("\n✅ SafeTensors demo completed successfully!")


def compare_formats():
    """Compare SafeTensors vs legacy NPZ format"""
    print("\n🆚 FORMAT COMPARISON")
    print("="*40)
    
    manager = UserLLMManager()
    dataset = HelloWorldDataset()
    
    # Create two identical models
    print("Creating identical models for comparison...")
    
    # Model 1: SafeTensors (default for new models)
    model_st = manager.create_user_model("format_test_st", "neural", vocab_size=50)
    model_st.train(dataset.texts)
    
    print("\n📦 Saving in SafeTensors format...")
    start_time = time.time()
    model_st.save("comparison_model")
    st_save_time = time.time() - start_time
    
    print("\n📦 Loading from SafeTensors...")
    start_time = time.time()
    loaded_st = manager.load_user_model("format_test_st", "comparison_model")
    st_load_time = time.time() - start_time
    
    # Check file sizes
    st_dir = "models/format_test_st"
    st_model_file = os.path.join(st_dir, "comparison_model_neural.safetensors")
    st_vocab_file = os.path.join(st_dir, "comparison_model_vocab.json")
    
    st_total_size = 0
    if os.path.exists(st_model_file):
        st_total_size += os.path.getsize(st_model_file)
    if os.path.exists(st_vocab_file):
        st_total_size += os.path.getsize(st_vocab_file)
    
    print("\n📊 Performance Comparison:")
    print("-" * 40)
    print("SafeTensors Format:")
    print(f"  💾 Total size: {st_total_size/1024:.1f} KB")
    print(f"  💾 Save time: {st_save_time:.3f}s")
    print(f"  💾 Load time: {st_load_time:.3f}s")
    
    print("\n🔒 SafeTensors Benefits:")
    print("  ✅ Memory safe - no arbitrary code execution")
    print("  ✅ Fast loading with memory mapping")
    print("  ✅ Cross-platform compatibility")
    print("  ✅ Lazy loading support")
    print("  ✅ Better error handling")
    print("  ✅ Metadata validation")


def migration_demo():
    """Demonstrate migration from legacy format"""
    print("\n🔄 MIGRATION DEMO")
    print("="*30)
    
    manager = UserLLMManager()
    
    # Check for legacy models
    users = manager.list_users()
    legacy_found = False
    
    for user in users:
        models = manager.list_user_models(user)
        for model_info in models:
            if model_info.get('format') == 'legacy_npz':
                legacy_found = True
                print(f"📦 Found legacy model: {user}/{model_info['name']}")
    
    if legacy_found:
        print("\n🔄 Converting legacy models to SafeTensors...")
        converted, failed = manager.batch_convert_legacy_models()
        
        if converted > 0:
            print(f"\n✅ Successfully converted {converted} models to SafeTensors!")
        if failed > 0:
            print(f"⚠️  {failed} models failed to convert")
    else:
        print("ℹ️  No legacy models found. All models are using modern formats!")
    
    print("\n📋 Current model inventory:")
    for user in users:
        models = manager.list_user_models(user)
        if models:
            print(f"\n👤 {user}:")
            for model_info in models:
                format_icon = "🔒" if model_info.get('format') == 'safetensors' else "📄"
                print(f"  {format_icon} {model_info['name']} ({model_info['type']}, {model_info.get('format', 'unknown')})")


def main():
    print("🚀 SAFETENSORS UNIFIED LLM DEMO")
    print("="*60)
    
    print("\nChoose demo:")
    print("1. SafeTensors Basic Demo")
    print("2. Format Comparison")
    print("3. Legacy Migration Demo")
    print("4. All Demos")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        safetensors_demo()
    elif choice == "2":
        compare_formats()
    elif choice == "3":
        migration_demo()
    else:
        safetensors_demo()
        compare_formats()
        migration_demo()
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETE!")
    print("✅ Your LLM system now uses SafeTensors for secure model storage")
    print("✅ Faster loading, better security, cross-platform compatibility")
    print("="*60)


if __name__ == "__main__":
    main()
