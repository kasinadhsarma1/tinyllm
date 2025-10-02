#!/usr/bin/env python3
"""
TinyLLM Package Validation Script
Tests all major functionality to ensure the package works correctly
"""

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Core modules
        from tinyllm.core import UnifiedLLM, UserLLMManager
        from tinyllm.models import NGramModel, NeuralModel
        from tinyllm.chat import ChatPersonality, MultiPersonalityChat
        from tinyllm.utils import scan_models_directory, compare_model_formats
        
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic model creation and training"""
    print("🧠 Testing basic functionality...")
    
    try:
        from tinyllm.core import UnifiedLLM
        
        # Test N-gram model
        ngram_model = UnifiedLLM(model_type="ngram", user_id="test_validation")
        ngram_model.train("Hello world test")
        ngram_result = ngram_model.generate("Hello", 10)
        
        # Test Neural model
        neural_model = UnifiedLLM(model_type="neural", user_id="test_validation")
        neural_model.train("Hello world test", epochs=1)
        neural_result = neural_model.generate("Hello", 10)
        
        print("✅ Basic functionality works!")
        print(f"  N-gram: {ngram_result}")
        print(f"  Neural: {neural_result}")
        return True
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        return False


def test_safetensors():
    """Test SafeTensors functionality"""
    print("🔐 Testing SafeTensors...")
    
    try:
        from tinyllm.models import NeuralModel
        import os
        
        # Create and save model
        model = NeuralModel(user_id="safetensors_validation", vocab_size=50, embedding_dim=16)
        model.train("SafeTensors test", epochs=1)
        model.save("safetensors_validation")
        
        # Load model
        loaded_model = NeuralModel.load("safetensors_validation", "safetensors_validation")
        result = loaded_model.generate("Test", 10)
        
        print("✅ SafeTensors functionality works!")
        print(f"  Generated: {result}")
        return True
    except Exception as e:
        print(f"❌ SafeTensors failed: {e}")
        return False


def test_chat_system():
    """Test chat personality system"""
    print("🎭 Testing chat system...")
    
    try:
        from tinyllm.chat import ChatPersonality
        
        # Create personality
        alice = ChatPersonality("Alice", "ngram", {
            "greeting": "Hello! I'm Alice!",
            "style": "friendly"
        })
        
        # Train and test
        alice.train_on_data("Hello how are you today? I'm doing great!")
        response = alice.respond("Hello")
        
        print("✅ Chat system works!")
        print(f"  Alice: {response}")
        return True
    except Exception as e:
        print(f"❌ Chat system failed: {e}")
        return False


def test_utilities():
    """Test utility functions"""
    print("🛠️ Testing utilities...")
    
    try:
        from tinyllm.utils import scan_models_directory, format_bytes
        
        # Test directory scanning
        scan_result = scan_models_directory()
        
        # Test format utilities
        size_str = format_bytes(1024)
        
        print("✅ Utilities work!")
        print(f"  Found {scan_result.get('summary', {}).get('total_users', 0)} users")
        print(f"  Format test: {size_str}")
        return True
    except Exception as e:
        print(f"❌ Utilities failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("🚀 TinyLLM Package Validation")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_safetensors,
        test_chat_system,
        test_utilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("📊 Validation Results:")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! TinyLLM package is working correctly!")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/kasinadhsarma/tinyllm')
    
    success = main()
    sys.exit(0 if success else 1)
