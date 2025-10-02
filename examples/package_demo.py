#!/usr/bin/env python3
"""
TinyLLM Package Demo
Demonstrates the modular structure and SafeTensors integration
"""

import sys
import os

# Add the package to Python path for development
sys.path.insert(0, '/home/kasinadhsarma/tinyllm')

from tinyllm.core import UnifiedLLM, UserLLMManager
from tinyllm.models import NGramModel, NeuralModel
from tinyllm.chat import ChatPersonality, MultiPersonalityChat, create_sample_personalities
from tinyllm.utils import scan_models_directory, compare_model_formats, export_model_report


def demo_unified_llm():
    """Demonstrate UnifiedLLM with SafeTensors"""
    print("🧠 === TinyLLM Unified Model Demo ===")
    
    # Create N-gram model
    print("\n1. Creating N-gram model...")
    ngram_model = UnifiedLLM(model_type="ngram", user_id="demo_user")
    training_text = "Hello world! How are you today? I hope you are doing well!"
    ngram_model.train(training_text)
    ngram_model.save("demo_ngram")
    
    # Create Neural model with SafeTensors
    print("\n2. Creating Neural model with SafeTensors...")
    neural_model = UnifiedLLM(model_type="neural", user_id="demo_user")
    neural_model.train(training_text, epochs=2)
    neural_model.save("demo_neural")
    
    # Test generation
    print("\n3. Testing text generation...")
    prompt = "Hello"
    ngram_result = ngram_model.generate(prompt, 20)
    neural_result = neural_model.generate(prompt, 20)
    
    print(f"N-gram: {ngram_result}")
    print(f"Neural: {neural_result}")
    
    print("✓ Unified model demo completed!")


def demo_individual_models():
    """Demonstrate individual model classes"""
    print("\n🔧 === Individual Models Demo ===")
    
    # NGramModel
    print("\n1. NGramModel demonstration...")
    ngram = NGramModel(user_id="individual_demo")
    ngram.train("The quick brown fox jumps over the lazy dog.")
    ngram.save("individual_ngram")
    
    result = ngram.generate("The", 15)
    print(f"NGram result: {result}")
    
    # NeuralModel
    print("\n2. NeuralModel demonstration...")
    neural = NeuralModel(user_id="individual_demo", vocab_size=100, embedding_dim=32)
    neural.train("The quick brown fox jumps over the lazy dog.", epochs=3)
    neural.save("individual_neural")
    
    result = neural.generate("The", 15)
    print(f"Neural result: {result}")
    
    print("✓ Individual models demo completed!")


def demo_chat_personalities():
    """Demonstrate chat personalities"""
    print("\n🎭 === Chat Personalities Demo ===")
    
    # Create sample personalities
    chat_system = create_sample_personalities()
    
    # Test conversations
    test_inputs = ["Hello!", "How are you?", "Tell me about yourself"]
    
    for personality_name in ["Alice", "Bob", "Pirate"]:
        print(f"\n--- {personality_name} ---")
        chat_system.switch_personality(personality_name)
        
        for user_input in test_inputs:
            response = chat_system.chat(user_input)
            print(f"User: {user_input}")
            print(f"{response}")
            print()
    
    # Save personalities
    chat_system.save_all_personalities()
    print("✓ Chat personalities demo completed!")


def demo_user_management():
    """Demonstrate user management"""
    print("\n👥 === User Management Demo ===")
    
    manager = UserLLMManager()
    
    # Create models for different users
    users_data = [
        ("alice", "I love reading books and learning new things every day."),
        ("bob", "I am a professional software developer who enjoys coding."),
        ("charlie", "I'm interested in science and astronomy, especially space exploration.")
    ]
    
    for user_id, training_text in users_data:
        print(f"\n📝 Creating model for {user_id}...")
        
        # Create both N-gram and Neural models
        ngram_model = manager.create_user_model(user_id, "ngram")
        ngram_model.train(training_text)
        ngram_model.save(f"{user_id}_personality")
        
        neural_model = manager.create_user_model(user_id, "neural")
        neural_model.train(training_text, epochs=2)
        neural_model.save(f"{user_id}_personality")
    
    # List all users and models
    print("\n📋 Current users and models:")
    users = manager.list_users()
    for user_id in users:
        models = manager.list_user_models(user_id)
        print(f"  {user_id}: {len(models)} models")
        for model in models:
            print(f"    - {model['name']} ({model['type']}, {model['format']})")
    
    # Get system stats
    stats = manager.get_stats()
    print(f"\n📊 System Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("✓ User management demo completed!")


def demo_safetensors_features():
    """Demonstrate SafeTensors specific features"""
    print("\n🔐 === SafeTensors Features Demo ===")
    
    # Create a model and save in SafeTensors format
    model = NeuralModel(user_id="safetensors_demo", vocab_size=50, embedding_dim=16)
    model.train("SafeTensors is secure and efficient for model storage.", epochs=2)
    model.save("safetensors_test")
    
    # Compare formats if legacy exists (simulate)
    print("\n🔍 Comparing model formats...")
    comparison = compare_model_formats("safetensors_demo", "safetensors_test")
    
    if comparison['formats']['safetensors']['exists']:
        st_info = comparison['formats']['safetensors']
        print(f"SafeTensors model:")
        print(f"  File size: {st_info['file_size']}")
        print(f"  Load time: {st_info['load_time_ms']} ms")
        print(f"  Tensors: {st_info['tensor_count']}")
        print(f"  Names: {st_info['tensor_names']}")
    
    print("✓ SafeTensors features demo completed!")


def demo_utils_and_reporting():
    """Demonstrate utility functions and reporting"""
    print("\n📊 === Utils and Reporting Demo ===")
    
    # Scan models directory
    print("\n🔍 Scanning models directory...")
    scan_result = scan_models_directory()
    
    if 'summary' in scan_result:
        summary = scan_result['summary']
        print(f"Total users: {summary['total_users']}")
        print(f"Total models: {summary['total_models']}")
        print(f"N-gram models: {summary['ngram_models']}")
        print(f"Neural models: {summary['neural_models']}")
        print(f"SafeTensors models: {summary['safetensors_models']}")
        print(f"Legacy NPZ models: {summary['legacy_npz_models']}")
    
    # Export comprehensive report
    print("\n📄 Exporting model report...")
    report_file = export_model_report(output_file="tinyllm_demo_report.json")
    print(f"Report saved to: {report_file}")
    
    print("✓ Utils and reporting demo completed!")


def main():
    """Run all demonstrations"""
    print("🚀 TinyLLM Package Structure Demo")
    print("=" * 50)
    
    try:
        demo_unified_llm()
        demo_individual_models() 
        demo_chat_personalities()
        demo_user_management()
        demo_safetensors_features()
        demo_utils_and_reporting()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nTinyLLM package features:")
        print("✓ Unified LLM interface")
        print("✓ Individual model classes")
        print("✓ Chat personalities system")
        print("✓ User-based model management")
        print("✓ SafeTensors integration")
        print("✓ Comprehensive utilities")
        print("✓ Modular package structure")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
