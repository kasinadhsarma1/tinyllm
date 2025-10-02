"""
Individual model classes for TinyLLM
"""

import os
import json
import numpy as np
from safetensors.numpy import save_file, load_file
from typing import Dict, Optional


class NGramModel:
    """N-gram character-based language model"""
    
    def __init__(self, user_id: str = "default", models_dir: str = "models"):
        self.user_id = user_id
        self.models_dir = models_dir
        self.char_counts = {}
        self.char_probs = {}
        
        # Ensure user directory exists
        user_dir = os.path.join(models_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
    
    def train(self, text: str) -> None:
        """Train the N-gram model on text"""
        for i in range(len(text) - 1):
            current_char = text[i]
            next_char = text[i + 1]
            
            if current_char not in self.char_counts:
                self.char_counts[current_char] = {}
            if next_char not in self.char_counts[current_char]:
                self.char_counts[current_char][next_char] = 0
            
            self.char_counts[current_char][next_char] += 1
        
        # Calculate probabilities
        for char in self.char_counts:
            total = sum(self.char_counts[char].values())
            self.char_probs[char] = {
                next_char: count / total 
                for next_char, count in self.char_counts[char].items()
            }
    
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Generate text using N-gram model"""
        if not self.char_probs:
            return prompt + " [Model not trained]"
        
        result = prompt
        current_char = prompt[-1] if prompt else 'a'
        
        for _ in range(max_length):
            if current_char in self.char_probs:
                # Choose next character based on probabilities
                choices = list(self.char_probs[current_char].keys())
                probs = list(self.char_probs[current_char].values())
                
                if choices:
                    next_char = np.random.choice(choices, p=probs)
                    result += next_char
                    current_char = next_char
                else:
                    break
            else:
                # If character not in model, pick a random trained character
                if self.char_probs:
                    current_char = np.random.choice(list(self.char_probs.keys()))
                else:
                    break
        
        return result
    
    def save(self, model_name: str = "model") -> None:
        """Save N-gram model to JSON"""
        user_dir = os.path.join(self.models_dir, self.user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        data = {
            'model_type': 'ngram',
            'char_counts': self.char_counts,
            'char_probs': self.char_probs,
            'user_id': self.user_id
        }
        
        file_path = os.path.join(user_dir, f"{model_name}_ngram.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ N-gram model saved to {file_path}")
    
    @classmethod
    def load(cls, user_id: str, model_name: str = "model", models_dir: str = "models") -> 'NGramModel':
        """Load N-gram model from JSON"""
        user_dir = os.path.join(models_dir, user_id)
        file_path = os.path.join(user_dir, f"{model_name}_ngram.json")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        model = cls(user_id=user_id, models_dir=models_dir)
        model.char_counts = data['char_counts']
        model.char_probs = data['char_probs']
        
        print(f"✓ N-gram model loaded from {file_path}")
        return model
    
    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'ngram',
            'user_id': self.user_id,
            'char_count': len(self.char_counts),
            'trained': len(self.char_probs) > 0
        }


class NeuralModel:
    """Simple neural language model with SafeTensors support"""
    
    def __init__(self, user_id: str = "default", vocab_size: int = 1000, 
                 embedding_dim: int = 50, models_dir: str = "models"):
        self.user_id = user_id
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.models_dir = models_dir
        
        # Initialize model parameters
        self.embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        self.bias = np.zeros(vocab_size)
        self.vocab = {str(i): i for i in range(vocab_size)}
        self.char_to_id = {chr(i): i for i in range(min(256, vocab_size))}
        self.id_to_char = {i: chr(i) for i in range(min(256, vocab_size))}
        
        # Ensure user directory exists
        user_dir = os.path.join(models_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
    
    def train(self, text: str, epochs: int = 1) -> None:
        """Simple neural training simulation"""
        for epoch in range(epochs):
            # Simple embedding updates based on character co-occurrence
            for i in range(len(text) - 1):
                if text[i] in self.char_to_id and text[i + 1] in self.char_to_id:
                    curr_id = self.char_to_id[text[i]]
                    next_id = self.char_to_id[text[i + 1]]
                    
                    # Simple update rule
                    learning_rate = 0.01
                    self.embeddings[curr_id] += learning_rate * np.random.normal(0, 0.1, self.embedding_dim)
                    self.bias[next_id] += learning_rate * 0.1
    
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Generate text using neural model"""
        result = prompt
        
        for _ in range(max_length):
            if not prompt:
                break
                
            # Simple generation based on last character
            last_char = result[-1] if result else 'a'
            if last_char in self.char_to_id:
                char_id = self.char_to_id[last_char]
                
                # Simple prediction based on embeddings and bias
                scores = np.dot(self.embeddings, self.embeddings[char_id]) + self.bias
                
                # Add some randomness
                scores += np.random.normal(0, 0.1, len(scores))
                
                # Choose character with highest score among valid ones
                valid_ids = [i for i in range(len(scores)) if i in self.id_to_char]
                if valid_ids:
                    best_id = max(valid_ids, key=lambda x: scores[x])
                    next_char = self.id_to_char[best_id]
                    result += next_char
                else:
                    break
            else:
                result += np.random.choice(list(self.char_to_id.keys()))
        
        return result
    
    def save(self, model_name: str = "model") -> None:
        """Save Neural model using SafeTensors"""
        user_dir = os.path.join(self.models_dir, self.user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Prepare data for SafeTensors
        tensors = {
            'embeddings': self.embeddings,
            'bias': self.bias,
        }
        
        # Save model data
        safetensors_path = os.path.join(user_dir, f"{model_name}_neural.safetensors")
        save_file(tensors, safetensors_path)
        
        # Save metadata separately
        metadata = {
            'model_type': 'neural',
            'user_id': self.user_id,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'vocab': self.vocab,
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char
        }
        
        metadata_path = os.path.join(user_dir, f"{model_name}_neural_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Neural model saved to {safetensors_path}")
        print(f"✓ Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, user_id: str, model_name: str = "model", models_dir: str = "models") -> 'NeuralModel':
        """Load Neural model from SafeTensors"""
        user_dir = os.path.join(models_dir, user_id)
        safetensors_path = os.path.join(user_dir, f"{model_name}_neural.safetensors")
        metadata_path = os.path.join(user_dir, f"{model_name}_neural_metadata.json")
        
        # Load tensors
        tensors = load_file(safetensors_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create model instance
        model = cls(
            user_id=user_id,
            vocab_size=metadata['vocab_size'],
            embedding_dim=metadata['embedding_dim'],
            models_dir=models_dir
        )
        
        # Restore model data
        model.embeddings = tensors['embeddings']
        model.bias = tensors['bias']
        model.vocab = metadata['vocab']
        model.char_to_id = metadata['char_to_id']
        model.id_to_char = metadata['id_to_char']
        
        print(f"✓ Neural model loaded from {safetensors_path}")
        return model
    
    @classmethod
    def load_legacy(cls, user_id: str, model_name: str = "model", models_dir: str = "models") -> 'NeuralModel':
        """Load legacy Neural model from NPZ format"""
        user_dir = os.path.join(models_dir, user_id)
        legacy_path = os.path.join(user_dir, f"{model_name}_neural.npz")
        vocab_path = os.path.join(user_dir, "vocab.json")
        
        # Load legacy format
        data = np.load(legacy_path)
        
        # Load vocab if exists
        vocab = {}
        char_to_id = {}
        id_to_char = {}
        
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                vocab = vocab_data.get('vocab', {})
                char_to_id = vocab_data.get('char_to_id', {})
                id_to_char = vocab_data.get('id_to_char', {})
        
        # Create model instance
        embeddings = data['embeddings']
        bias = data.get('bias', np.zeros(embeddings.shape[0]))
        
        model = cls(
            user_id=user_id,
            vocab_size=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            models_dir=models_dir
        )
        
        model.embeddings = embeddings
        model.bias = bias
        model.vocab = vocab if vocab else model.vocab
        model.char_to_id = char_to_id if char_to_id else model.char_to_id
        model.id_to_char = id_to_char if id_to_char else model.id_to_char
        
        print(f"✓ Legacy model loaded from {legacy_path}")
        return model
    
    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'neural',
            'user_id': self.user_id,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'trained': True
        }
