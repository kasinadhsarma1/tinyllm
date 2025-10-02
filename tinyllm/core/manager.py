"""
User LLM Manager for handling multiple user models
"""

import os
from typing import List, Dict, Optional
from .unified_llm import UnifiedLLM


class UserLLMManager:
    """Manager for user-based LLM models"""
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def list_users(self) -> List[str]:
        """List all users with models"""
        users = []
        if os.path.exists(self.base_dir):
            for item in os.listdir(self.base_dir):
                user_dir = os.path.join(self.base_dir, item)
                if os.path.isdir(user_dir):
                    users.append(item)
        return users
    
    def list_user_models(self, user_id: str) -> List[Dict[str, str]]:
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
    
    def create_user_model(self, user_id: str, model_type: str = "ngram", **kwargs) -> UnifiedLLM:
        """Create a new model for a user"""
        return UnifiedLLM(model_type=model_type, user_id=user_id, **kwargs)
    
    def load_user_model(self, user_id: str, model_name: str = "model", 
                       model_type: Optional[str] = None) -> UnifiedLLM:
        """Load a model for a user"""
        return UnifiedLLM.load(user_id, model_name, model_type)
    
    def convert_legacy_to_safetensors(self, user_id: str, model_name: str = "model") -> bool:
        """Convert legacy NPZ neural models to SafeTensors format"""
        models_dir = f"{self.base_dir}/{user_id}"
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
    
    def batch_convert_legacy_models(self) -> tuple[int, int]:
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
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the system"""
        stats = {
            'total_users': len(self.list_users()),
            'total_models': 0,
            'ngram_models': 0,
            'neural_models': 0,
            'safetensors_models': 0,
            'legacy_models': 0
        }
        
        for user_id in self.list_users():
            models = self.list_user_models(user_id)
            stats['total_models'] += len(models)
            
            for model in models:
                if model['type'] == 'ngram':
                    stats['ngram_models'] += 1
                elif model['type'] == 'neural':
                    if model['format'] == 'safetensors':
                        stats['safetensors_models'] += 1
                    else:
                        stats['legacy_models'] += 1
                    stats['neural_models'] += 1
        
        return stats
