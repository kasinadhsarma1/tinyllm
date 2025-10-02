"""
Unified Hello World LLM with User-Based Model Management
A single LLM class that supports both N-gram and Neural models with user interaction
Uses SafeTensors for secure and efficient model serialization
"""

import numpy as np
import json
import os
from collections import defaultdict
from safetensors import safe_open
from safetensors.numpy import save_file, load_file


class UnifiedLLM:
    """Unified LLM supporting both N-gram and Neural models with user management"""
    
    def __init__(self, model_type="ngram", user_id="default", **kwargs):
        self.model_type = model_type.lower()
        self.user_id = user_id
        self.models_dir = f"models/{user_id}"
        os.makedirs(self.models_dir, exist_ok=True)
        
        if self.model_type == "ngram":
            self._init_ngram_model(kwargs.get('n', 3))
        elif self.model_type == "neural":
            self._init_neural_model(
                kwargs.get('vocab_size', 100),
                kwargs.get('embed_dim', 32),
                kwargs.get('hidden_dim', 64)
            )
        else:
            raise ValueError("Model type must be 'ngram' or 'neural'")
    
    def _init_ngram_model(self, n):
        """Initialize N-gram model"""
        self.n = n
        self.model = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        print(f"✓ Initialized N-gram model (n={n}) for user '{self.user_id}'")
    
    def _init_neural_model(self, vocab_size, embed_dim, hidden_dim):
        """Initialize Neural model"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        np.random.seed(42)
        self.W_embed = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_hidden = np.random.randn(embed_dim, hidden_dim) * 0.01
        self.b_hidden = np.zeros(hidden_dim)
        self.W_out = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b_out = np.zeros(vocab_size)
        print(f"✓ Initialized Neural model for user '{self.user_id}'")
    
    def train(self, texts):
        """Train the model on text data"""
        print(f"Training {self.model_type} model for user '{self.user_id}' on {len(texts)} examples...")
        
        if self.model_type == "ngram":
            self._train_ngram(texts)
        elif self.model_type == "neural":
            self._train_neural(texts)
    
    def _train_ngram(self, texts):
        """Train N-gram model"""
        for text in texts:
            text = '^' * (self.n - 1) + text + '$'
            self.vocab.update(text)
            
            for i in range(len(text) - self.n):
                context = text[i:i + self.n - 1]
                next_char = text[i + self.n - 1]
                self.model[context][next_char] += 1
        
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  Learned {len(self.model)} context patterns")
    
    def _train_neural(self, texts):
        """Train Neural model"""
        # Create vocabulary if not exists
        if not hasattr(self, 'char_to_idx'):
            self._create_vocab(texts)
        
        for epoch in range(100):  # Reduced for faster demo
            total_loss = 0
            count = 0
            
            for text in texts:
                indices = self._text_to_indices(text)
                for i in range(len(indices) - 1):
                    context = indices[max(0, i-5):i+1]
                    target = indices[i + 1]
                    loss = self._neural_train_step(context, target, learning_rate=0.01)
                    total_loss += loss
                    count += 1
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}/100 - Loss: {total_loss/count:.4f}")
    
    def _create_vocab(self, texts):
        """Create vocabulary for neural model"""
        self.char_to_idx = {'<PAD>': 0}
        self.idx_to_char = {0: '<PAD>'}
        
        all_chars = sorted(set(''.join(texts)))
        for idx, char in enumerate(all_chars, 1):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_idx)
        print(f"  Created vocabulary with {self.vocab_size} characters")
    
    def _text_to_indices(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def _indices_to_text(self, indices):
        return ''.join([self.idx_to_char.get(i, '') for i in indices if i != 0])
    
    def save(self, model_name="model"):
        """Save model to user-specific directory"""
        if self.model_type == "ngram":
            filepath = os.path.join(self.models_dir, f"{model_name}_ngram.json")
            self._save_ngram(filepath)
        elif self.model_type == "neural":
            filepath = os.path.join(self.models_dir, f"{model_name}_neural.safetensors")
            vocab_path = os.path.join(self.models_dir, f"{model_name}_vocab.json")
            self._save_neural(filepath, vocab_path)
    
    def _save_ngram(self, filepath):
        """Save N-gram model"""
        model_dict = {
            context: dict(chars) 
            for context, chars in self.model.items()
        }
        
        save_data = {
            'model_type': 'ngram',
            'user_id': self.user_id,
            'n': self.n,
            'vocab': list(self.vocab),
            'model': model_dict
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
        
        print(f"✓ N-gram model saved to {filepath}")
    
    def _save_neural(self, filepath, vocab_path):
        """Save Neural model using SafeTensors format"""
        # Prepare tensors dictionary for SafeTensors
        tensors = {
            "W_embed": self.W_embed.astype(np.float32),
            "W_hidden": self.W_hidden.astype(np.float32),
            "b_hidden": self.b_hidden.astype(np.float32),
            "W_out": self.W_out.astype(np.float32),
            "b_out": self.b_out.astype(np.float32)
        }
        
        # Save model weights using SafeTensors
        save_file(tensors, filepath)
        
        # Save metadata and vocabulary separately
        metadata = {
            'model_type': 'neural',
            'user_id': self.user_id,
            'vocab_size': int(self.vocab_size),
            'embed_dim': int(self.embed_dim),
            'hidden_dim': int(self.hidden_dim),
            'safetensors_version': True
        }
        
        # Save vocabulary and metadata
        vocab_data = {
            'metadata': metadata,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {int(k): v for k, v in self.idx_to_char.items()}
        }
        
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"✓ Neural model saved to {filepath} (SafeTensors format)")
        print(f"✓ Vocabulary and metadata saved to {vocab_path}")
        
        # Show file sizes
        model_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        vocab_size_kb = os.path.getsize(vocab_path) / 1024
        print(f"  Model file size: {model_size_mb:.3f} MB")
        print(f"  Vocab file size: {vocab_size_kb:.1f} KB")
    
    @classmethod
    def load(cls, user_id, model_name="model", model_type=None):
        """Load model for specific user"""
        models_dir = f"models/{user_id}"
        
        # Auto-detect model type if not specified
        if model_type is None:
            ngram_path = os.path.join(models_dir, f"{model_name}_ngram.json")
            neural_path = os.path.join(models_dir, f"{model_name}_neural.safetensors")
            neural_legacy_path = os.path.join(models_dir, f"{model_name}_neural.npz")
            
            if os.path.exists(ngram_path):
                model_type = "ngram"
            elif os.path.exists(neural_path):
                model_type = "neural"
            elif os.path.exists(neural_legacy_path):
                model_type = "neural_legacy"
            else:
                raise FileNotFoundError(f"No model found for user '{user_id}' with name '{model_name}'")
        
        if model_type == "ngram":
            return cls._load_ngram(user_id, model_name, models_dir)
        elif model_type == "neural":
            return cls._load_neural_safetensors(user_id, model_name, models_dir)
        elif model_type == "neural_legacy":
            return cls._load_neural_legacy(user_id, model_name, models_dir)
    
    @classmethod
    def _load_ngram(cls, user_id, model_name, models_dir):
        """Load N-gram model"""
        filepath = os.path.join(models_dir, f"{model_name}_ngram.json")
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        model = cls(model_type="ngram", user_id=user_id, n=save_data['n'])
        model.vocab = set(save_data['vocab'])
        
        for context, chars in save_data['model'].items():
            for char, count in chars.items():
                model.model[context][char] = count
        
        print(f"✓ N-gram model loaded for user '{user_id}'")
        return model
    
    @classmethod
    def _load_neural_safetensors(cls, user_id, model_name, models_dir):
        """Load Neural model from SafeTensors format"""
        filepath = os.path.join(models_dir, f"{model_name}_neural.safetensors")
        vocab_path = os.path.join(models_dir, f"{model_name}_vocab.json")
        
        # Load vocabulary and metadata
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        # Extract metadata
        if 'metadata' in vocab_data:
            metadata = vocab_data['metadata']
            vocab_size = metadata['vocab_size']
            embed_dim = metadata['embed_dim']
            hidden_dim = metadata['hidden_dim']
        else:
            # Fallback for older format
            vocab_size = len(vocab_data['char_to_idx'])
            embed_dim = 32
            hidden_dim = 64
        
        # Create model instance
        model = cls(
            model_type="neural", 
            user_id=user_id,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        
        # Load weights from SafeTensors
        tensors = load_file(filepath)
        
        model.W_embed = tensors["W_embed"]
        model.W_hidden = tensors["W_hidden"]
        model.b_hidden = tensors["b_hidden"]
        model.W_out = tensors["W_out"]
        model.b_out = tensors["b_out"]
        
        # Load vocabulary
        model.char_to_idx = vocab_data['char_to_idx']
        model.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        
        print(f"✓ Neural model loaded from {filepath} (SafeTensors format)")
        return model
    
    @classmethod
    def _load_neural_legacy(cls, user_id, model_name, models_dir):
        """Load Neural model from legacy NPZ format"""
        filepath = os.path.join(models_dir, f"{model_name}_neural.npz")
        vocab_path = os.path.join(models_dir, f"{model_name}_vocab.json")
        
        print(f"⚠ Loading legacy NPZ format. Consider re-saving as SafeTensors.")
        
        # Load weights
        data = np.load(filepath)
        vocab_size = int(data['vocab_size'])
        embed_dim = int(data['embed_dim'])
        hidden_dim = int(data['hidden_dim'])
        
        model = cls(
            model_type="neural", 
            user_id=user_id,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        
        model.W_embed = data['W_embed']
        model.W_hidden = data['W_hidden']
        model.b_hidden = data['b_hidden']
        model.W_out = data['W_out']
        model.b_out = data['b_out']
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        model.char_to_idx = vocab_data['char_to_idx']
        model.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        
        print(f"✓ Neural model loaded from {filepath} (Legacy NPZ format)")
        return model

    @classmethod
    def _load_neural(cls, user_id, model_name, models_dir):
        """Load Neural model (deprecated - redirects to legacy loader)"""
        return cls._load_neural_legacy(user_id, model_name, models_dir)
    
    def generate(self, prompt="", max_length=50, temperature=1.0):
        """Generate text using the appropriate model"""
        if self.model_type == "ngram":
            return self._generate_ngram(prompt, max_length, temperature)
        elif self.model_type == "neural":
            return self._generate_neural(prompt, max_length, temperature)
    
    def _generate_ngram(self, prompt, max_length, temperature):
        """Generate text using N-gram model"""
        generated = '^' * (self.n - 1) + prompt
        
        for _ in range(max_length):
            context = generated[-(self.n - 1):]
            next_char = self._predict_next_ngram(context, temperature)
            
            if next_char == '$':
                break
            
            generated += next_char
        
        return generated.replace('^', '')
    
    def _generate_neural(self, prompt, max_length, temperature):
        """Generate text using Neural model"""
        generated_indices = self._text_to_indices(prompt)
        
        for _ in range(max_length):
            next_idx = self._predict_neural(generated_indices[-5:], temperature)
            if next_idx == 0:
                break
            generated_indices.append(next_idx)
        
        return self._indices_to_text(generated_indices)
    
    def _predict_next_ngram(self, context, temperature=1.0):
        """Predict next character for N-gram model"""
        if context not in self.model:
            if len(context) > 1:
                return self._predict_next_ngram(context[1:], temperature)
            return np.random.choice(list(self.vocab)) if self.vocab else ' '
        
        char_counts = self.model[context]
        chars = list(char_counts.keys())
        counts = np.array([char_counts[c] for c in chars], dtype=float)
        
        if temperature != 1.0:
            counts = counts ** (1.0 / temperature)
        
        probs = counts / counts.sum()
        return np.random.choice(chars, p=probs)
    
    def _predict_neural(self, indices, temperature=1.0):
        """Predict next index for Neural model"""
        logits = self._neural_forward(np.array(indices))
        logits = logits / temperature
        probs = self._softmax(logits)
        return np.random.choice(len(probs), p=probs)
    
    def _neural_forward(self, x):
        """Forward pass for neural model"""
        embedded = self.W_embed[x]
        pooled = embedded.mean(axis=0)
        hidden = self._relu(np.dot(pooled, self.W_hidden) + self.b_hidden)
        logits = np.dot(hidden, self.W_out) + self.b_out
        return logits
    
    def _neural_train_step(self, x, y, learning_rate=0.01):
        """Training step for neural model"""
        embedded = self.W_embed[x]
        pooled = embedded.mean(axis=0)
        hidden_input = np.dot(pooled, self.W_hidden) + self.b_hidden
        hidden = self._relu(hidden_input)
        logits = np.dot(hidden, self.W_out) + self.b_out
        probs = self._softmax(logits)
        
        loss = -np.log(probs[y] + 1e-10)
        
        d_logits = probs.copy()
        d_logits[y] -= 1
        
        self.W_out -= learning_rate * np.outer(hidden, d_logits)
        self.b_out -= learning_rate * d_logits
        
        d_hidden = np.dot(d_logits, self.W_out.T)
        d_hidden[hidden_input <= 0] = 0
        
        self.W_hidden -= learning_rate * np.outer(pooled, d_hidden)
        self.b_hidden -= learning_rate * d_hidden
        
        return loss
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def get_model_info(self):
        """Get information about the model"""
        info = {
            'model_type': self.model_type,
            'user_id': self.user_id,
            'models_dir': self.models_dir
        }
        
        if self.model_type == "ngram":
            info.update({
                'n': self.n,
                'vocab_size': len(self.vocab),
                'context_patterns': len(self.model)
            })
        elif self.model_type == "neural":
            info.update({
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim
            })
        
        return info


class UserLLMManager:
    """Manager for user-based LLM models"""
    
    def __init__(self):
        self.base_dir = "models"
        os.makedirs(self.base_dir, exist_ok=True)
    
    def list_users(self):
        """List all users with models"""
        users = []
        if os.path.exists(self.base_dir):
            for item in os.listdir(self.base_dir):
                user_dir = os.path.join(self.base_dir, item)
                if os.path.isdir(user_dir):
                    users.append(item)
        return users
    
    def list_user_models(self, user_id):
        """List all models for a specific user"""
        user_dir = os.path.join(self.base_dir, user_id)
        models = []
        
        if os.path.exists(user_dir):
            files = os.listdir(user_dir)
            
            # Find unique model names
            model_names = set()
            for file in files:
                if file.endswith('_ngram.json'):
                    model_names.add(file.replace('_ngram.json', ''))
                elif file.endswith('_neural.safetensors'):
                    model_names.add(file.replace('_neural.safetensors', ''))
                elif file.endswith('_neural.npz'):  # Legacy format
                    model_names.add(file.replace('_neural.npz', ''))
            
            for name in model_names:
                ngram_exists = f"{name}_ngram.json" in files
                neural_safetensors_exists = f"{name}_neural.safetensors" in files
                neural_legacy_exists = f"{name}_neural.npz" in files
                
                if ngram_exists:
                    models.append({'name': name, 'type': 'ngram', 'format': 'json'})
                if neural_safetensors_exists:
                    models.append({'name': name, 'type': 'neural', 'format': 'safetensors'})
                elif neural_legacy_exists:
                    models.append({'name': name, 'type': 'neural', 'format': 'legacy_npz'})
        
        return models
    
    def create_user_model(self, user_id, model_type="ngram", **kwargs):
        """Create a new model for a user"""
        return UnifiedLLM(model_type=model_type, user_id=user_id, **kwargs)
    
    def load_user_model(self, user_id, model_name="model", model_type=None):
        """Load a model for a user"""
        return UnifiedLLM.load(user_id, model_name, model_type)
    
    def convert_legacy_to_safetensors(self, user_id, model_name="model"):
        """Convert legacy NPZ neural models to SafeTensors format"""
        models_dir = f"models/{user_id}"
        legacy_path = os.path.join(models_dir, f"{model_name}_neural.npz")
        
        if not os.path.exists(legacy_path):
            print(f"❌ No legacy model found: {legacy_path}")
            return False
        
        try:
            # Load legacy model
            print(f"🔄 Converting {user_id}/{model_name} from NPZ to SafeTensors...")
            model = UnifiedLLM._load_neural_legacy(user_id, model_name, models_dir)
            
            # Save in new SafeTensors format
            model.save(model_name)
            
            # Optionally remove legacy files
            backup_path = legacy_path + ".backup"
            os.rename(legacy_path, backup_path)
            print(f"✓ Legacy model backed up to: {backup_path}")
            print(f"✓ Model converted to SafeTensors format successfully!")
            
            return True
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
    
    def batch_convert_legacy_models(self):
        """Convert all legacy models to SafeTensors format"""
        converted_count = 0
        failed_count = 0
        
        users = self.list_users()
        for user_id in users:
            models = self.list_user_models(user_id)
            for model_info in models:
                if model_info.get('format') == 'legacy_npz':
                    if self.convert_legacy_to_safetensors(user_id, model_info['name']):
                        converted_count += 1
                    else:
                        failed_count += 1
        
        print(f"\n📊 Conversion Summary:")
        print(f"✓ Converted: {converted_count}")
        print(f"❌ Failed: {failed_count}")
        
        return converted_count, failed_count


class HelloWorldDataset:
    """Dataset for training"""
    
    def __init__(self):
        self.texts = [
            "hello world", "hello there", "hi world", "greetings world",
            "hello friend", "hello everyone", "hi there", "hey world",
            "hello universe", "good morning world", "hello beautiful world",
            "hi friend", "hello to the world", "greetings everyone",
            "hello hello world", "world hello", "the world says hello",
            "hello from the world", "world of hello", "in a world we say hello",
        ]


def simple_demo():
    """Simple demonstration of the unified system"""
    print("\n🚀 UNIFIED HELLO WORLD LLM - SIMPLE DEMO")
    print("="*60)
    
    dataset = HelloWorldDataset()
    manager = UserLLMManager()
    
    # Create and train models for different users
    print("\n1. Creating N-gram model for Alice...")
    alice_model = manager.create_user_model("alice", "ngram", n=3)
    alice_model.train(dataset.texts)
    alice_model.save("chat_model")
    
    print("\n2. Creating Neural model for Bob...")
    bob_model = manager.create_user_model("bob", "neural", vocab_size=50)
    bob_model.train(dataset.texts)
    bob_model.save("smart_model")
    
    print("\n3. Generating text with both models...")
    
    # Generate with Alice's model
    print("\n📝 Alice's N-gram generations:")
    for prompt in ["hello", "hi", "world"]:
        result = alice_model.generate(prompt, max_length=20, temperature=0.8)
        print(f"  '{prompt}' → '{result}'")
    
    # Generate with Bob's model
    print("\n🧠 Bob's Neural generations:")
    for prompt in ["hello", "hi", "world"]:
        result = bob_model.generate(prompt, max_length=15, temperature=0.8)
        print(f"  '{prompt}' → '{result}'")
    
    print("\n4. Model information:")
    print(f"\nAlice's model: {alice_model.get_model_info()}")
    print(f"Bob's model: {bob_model.get_model_info()}")
    
    print("\n5. Testing save/load functionality...")
    
    # Load models
    loaded_alice = manager.load_user_model("alice", "chat_model")
    loaded_bob = manager.load_user_model("bob", "smart_model")
    
    print("✓ Models loaded successfully!")
    
    # Test loaded models
    print("\n📝 Loaded Alice's model:")
    result = loaded_alice.generate("hello", max_length=15)
    print(f"  'hello' → '{result}'")
    
    print("\n🧠 Loaded Bob's model:")
    result = loaded_bob.generate("hello", max_length=15)
    print(f"  'hello' → '{result}'")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("✓ Created user-specific models")
    print("✓ Trained both N-gram and Neural models")
    print("✓ Generated text with different approaches")
    print("✓ Saved and loaded models successfully")
    print("✓ Demonstrated user-based model management")
    print("="*60)


def interactive_chat():
    """Simple interactive chat with user selection"""
    print("\n🤖 UNIFIED LLM CHAT")
    print("="*40)
    
    manager = UserLLMManager()
    dataset = HelloWorldDataset()
    
    # Setup or load user model
    user_id = input("Enter your username: ").strip() or "default"
    
    print(f"\nModel types:")
    print("1. N-gram (fast, character-based)")
    print("2. Neural (smarter, slower)")
    
    choice = input("Choose model type (1-2): ").strip()
    model_type = "ngram" if choice == "1" else "neural"
    
    # Try to load existing model
    try:
        model = manager.load_user_model(user_id, "chat_model", model_type)
        print(f"✓ Loaded existing {model_type} model for {user_id}")
    except:
        print(f"Creating new {model_type} model for {user_id}...")
        if model_type == "ngram":
            model = manager.create_user_model(user_id, "ngram", n=3)
        else:
            model = manager.create_user_model(user_id, "neural", vocab_size=50)
        
        model.train(dataset.texts)
        model.save("chat_model")
        print(f"✓ Model trained and saved!")
    
    print(f"\n💬 Chat with {user_id}'s {model_type} model!")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        user_input = input(f"\n{user_id}: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print(f"\n👋 Goodbye {user_id}!")
            break
        
        if not user_input:
            continue
        
        # Generate response
        response = model.generate(
            prompt=user_input[:3],  # Use first 3 chars as prompt
            max_length=30,
            temperature=0.8
        )
        
        print(f"Bot: {response}")


def main():
    print("\n🚀 UNIFIED HELLO WORLD LLM")
    print("   User-Based Model Management")
    print("="*60)
    
    print("\nChoose your experience:")
    print("1. Simple Demo (See all features)")
    print("2. Interactive Chat (Chat with your model)")
    print("3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == '1':
        simple_demo()
    elif choice == '2':
        interactive_chat()
    else:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
