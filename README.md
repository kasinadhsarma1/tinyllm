# Unified Hello World LLM with SafeTensors

# TinyLLM 🧠

A lightweight, modular language model library with SafeTensors support and user-based model isolation.

## ✨ Features

- **🔧 Unified Interface**: Single `UnifiedLLM` class supporting both N-gram and Neural models
- **🔐 SafeTensors Integration**: Secure and efficient model serialization
- **👥 User-Based Isolation**: Multi-user model management with isolated storage
- **🎭 Chat Personalities**: Create and manage multiple AI personalities
- **📦 Modular Architecture**: Clean package structure with separate modules
- **🔄 Legacy Support**: Automatic conversion from NPZ to SafeTensors format
- **🛠️ Comprehensive Utils**: Model scanning, validation, and reporting tools

## 📁 Package Structure

```
tinyllm/
├── core/                    # Core functionality
│   ├── unified_llm.py      # Main UnifiedLLM class
│   ├── manager.py          # UserLLMManager for multi-user support
│   └── dataset.py          # Data handling utilities
├── models/                  # Individual model classes
│   └── base_models.py      # NGramModel and NeuralModel classes
├── chat/                    # Chat interface modules
│   └── personalities.py    # ChatPersonality and MultiPersonalityChat
├── utils/                   # Utility functions
│   └── helpers.py          # Helper functions for model management
└── __init__.py             # Package initialization
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tinyllm

# Install the package
pip install -e .

# Or install dependencies manually
pip install numpy safetensors
```

### Basic Usage

```python
from tinyllm.core import UnifiedLLM

# Create and train an N-gram model
model = UnifiedLLM(model_type="ngram", user_id="alice")
model.train("Hello world! How are you today?")
model.save("my_model")

# Generate text
result = model.generate("Hello", max_length=20)
print(result)

# Create a Neural model with SafeTensors
neural_model = UnifiedLLM(model_type="neural", user_id="bob")
neural_model.train("The quick brown fox jumps.", epochs=3)
neural_model.save("neural_model")
```

### Chat Personalities

```python
from tinyllm.chat import create_sample_personalities

# Create chat system with personalities
chat_system = create_sample_personalities()

# Chat with different personalities
response = chat_system.chat("Hello there!", "Alice")
print(response)  # Alice: Hi there! I'm Alice, nice to meet you!

chat_system.switch_personality("Pirate") 
response = chat_system.chat("How are you?")
print(response)  # Pirate: Ahoy matey! Captain Pirate here!
```

## 🔐 SafeTensors Integration

TinyLLM uses SafeTensors for secure and efficient model storage with automatic legacy conversion and comprehensive utilities.

**TinyLLM** - Making language models accessible, secure, and user-friendly! 🚀 Now featuring **SafeTensors** for secure and efficient model serialization.

## Files

- **`unified_llm.py`** - Main unified LLM system with SafeTensors support (START HERE)
- **`enhanced_chat.py`** - Multi-personality chat interface  
- **`test_system.py`** - System testing and demonstration
- **`safetensors_demo.py`** - SafeTensors features and migration demo
- **`models/`** - User-specific model storage directory

## Quick Start

```bash
# Run the main unified system
python3 unified_llm.py

# Try SafeTensors features
python3 safetensors_demo.py

# Try the enhanced chat with multiple personalities
python3 enhanced_chat.py

# Run system tests
python3 test_system.py
```

## Features

✅ **User-Based Models**: Each user gets their own isolated model storage  
✅ **Multiple Model Types**: N-gram (fast, character-based) and Neural (smarter, embedding-based)  
✅ **SafeTensors Integration**: Secure, fast, and efficient model serialization  
✅ **Legacy Support**: Automatic detection and conversion of old NPZ models  
✅ **Save/Load**: Persistent model storage with automatic file management  
✅ **Interactive Chat**: Real-time conversation with your trained models  
✅ **Multi-Personality**: Chat with different AI personalities simultaneously  
✅ **Model Management**: List users, models, and switch between them easily  

## SafeTensors Benefits

🔒 **Security**: No arbitrary code execution - models can't contain malicious code  
⚡ **Performance**: Faster loading with memory mapping and lazy loading  
🌐 **Compatibility**: Cross-platform support and better error handling  
📊 **Validation**: Built-in metadata validation and integrity checking  

## Model Storage Formats

### Neural Models (SafeTensors - Default)
- **Format**: `.safetensors` + `.json` (metadata/vocab)
- **Benefits**: Secure, fast, memory-efficient
- **Location**: `models/username/modelname_neural.safetensors`

### N-gram Models (JSON)
- **Format**: `.json` (lightweight text format)
- **Benefits**: Human-readable, simple structure
- **Location**: `models/username/modelname_ngram.json`

### Legacy Support
- **Format**: `.npz` (NumPy compressed - deprecated)
- **Migration**: Automatic conversion available
- **Note**: Legacy models are auto-detected and can be converted

## Model Types

### N-gram Models
- Fast training and generation
- Character-level pattern learning
- Good for simple text completion
- Lightweight and predictable

### Neural Models  
- More sophisticated pattern learning
- Embedding-based representations
- Better context understanding
- Slower but more flexible

## User Isolation

Each user gets their own directory under `models/username/` containing:
- **SafeTensors files** (`.safetensors`) for Neural model weights
- **JSON files** (`.json`) for N-gram models and metadata/vocabulary
- **Multiple named models** per user with format detection
- **Automatic migration** from legacy formats

## Example Usage

```python
from unified_llm import UnifiedLLM, UserLLMManager, HelloWorldDataset

# Create manager
manager = UserLLMManager()

# Create user-specific Neural model (uses SafeTensors automatically)
model = manager.create_user_model("alice", "neural", vocab_size=50)

# Train on data
dataset = HelloWorldDataset()
model.train(dataset.texts)

# Save model (automatically uses SafeTensors for Neural models)
model.save("my_chat_model")

# Generate text
result = model.generate("hello", max_length=20)
print(result)

# Load later (auto-detects SafeTensors format)
loaded_model = manager.load_user_model("alice", "my_chat_model")

# Convert legacy models to SafeTensors
manager.convert_legacy_to_safetensors("alice", "old_model")

# Batch convert all legacy models
manager.batch_convert_legacy_models()
```

## Migration from Legacy

If you have existing `.npz` models, they will be automatically detected:

```python
# The system auto-detects and loads legacy models
model = manager.load_user_model("user", "legacy_model")  # Works with .npz

# Convert to modern SafeTensors format
manager.convert_legacy_to_safetensors("user", "legacy_model")

# Batch convert all legacy models at once
converted, failed = manager.batch_convert_legacy_models()
```

Enjoy your personalized AI language models! 🚀
