"""
Utility functions for TinyLLM
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from safetensors.numpy import load_file
import numpy as np


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)


def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(filepath) if os.path.exists(filepath) else 0


def format_bytes(bytes_size: int) -> str:
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def load_safetensors_info(filepath: str) -> Dict[str, Any]:
    """Load SafeTensors file and return metadata"""
    if not os.path.exists(filepath):
        return {}
    
    try:
        tensors = load_file(filepath)
        info = {
            'num_tensors': len(tensors),
            'tensor_names': list(tensors.keys()),
            'tensor_shapes': {name: tensor.shape for name, tensor in tensors.items()},
            'tensor_dtypes': {name: str(tensor.dtype) for name, tensor in tensors.items()},
            'file_size': format_bytes(get_file_size(filepath))
        }
        return info
    except Exception as e:
        return {'error': str(e)}


def compare_model_formats(user_id: str, model_name: str = "model", models_dir: str = "models") -> Dict[str, Any]:
    """Compare NPZ vs SafeTensors format for a model"""
    user_dir = os.path.join(models_dir, user_id)
    
    npz_path = os.path.join(user_dir, f"{model_name}_neural.npz")
    safetensors_path = os.path.join(user_dir, f"{model_name}_neural.safetensors")
    
    comparison = {
        'user_id': user_id,
        'model_name': model_name,
        'formats': {}
    }
    
    # Check NPZ format
    if os.path.exists(npz_path):
        npz_size = get_file_size(npz_path)
        
        # Time NPZ loading
        start_time = time.time()
        try:
            data = np.load(npz_path)
            keys = list(data.keys())
            npz_load_time = time.time() - start_time
            
            comparison['formats']['npz'] = {
                'exists': True,
                'file_size': format_bytes(npz_size),
                'file_size_bytes': npz_size,
                'load_time_ms': round(npz_load_time * 1000, 2),
                'tensor_count': len(keys),
                'tensor_names': keys
            }
        except Exception as e:
            comparison['formats']['npz'] = {
                'exists': True,
                'error': str(e)
            }
    else:
        comparison['formats']['npz'] = {'exists': False}
    
    # Check SafeTensors format
    if os.path.exists(safetensors_path):
        safetensors_size = get_file_size(safetensors_path)
        
        # Time SafeTensors loading
        start_time = time.time()
        try:
            tensors = load_file(safetensors_path)
            safetensors_load_time = time.time() - start_time
            
            comparison['formats']['safetensors'] = {
                'exists': True,
                'file_size': format_bytes(safetensors_size),
                'file_size_bytes': safetensors_size,
                'load_time_ms': round(safetensors_load_time * 1000, 2),
                'tensor_count': len(tensors),
                'tensor_names': list(tensors.keys()),
                'info': load_safetensors_info(safetensors_path)
            }
        except Exception as e:
            comparison['formats']['safetensors'] = {
                'exists': True,
                'error': str(e)
            }
    else:
        comparison['formats']['safetensors'] = {'exists': False}
    
    # Calculate differences
    if (comparison['formats']['npz'].get('exists') and 
        comparison['formats']['safetensors'].get('exists') and
        'error' not in comparison['formats']['npz'] and
        'error' not in comparison['formats']['safetensors']):
        
        npz_size = comparison['formats']['npz']['file_size_bytes']
        safetensors_size = comparison['formats']['safetensors']['file_size_bytes']
        npz_time = comparison['formats']['npz']['load_time_ms']
        safetensors_time = comparison['formats']['safetensors']['load_time_ms']
        
        comparison['differences'] = {
            'size_difference_bytes': safetensors_size - npz_size,
            'size_difference_percent': round(((safetensors_size - npz_size) / npz_size) * 100, 2) if npz_size > 0 else 0,
            'load_time_difference_ms': safetensors_time - npz_time,
            'load_time_improvement_percent': round(((npz_time - safetensors_time) / npz_time) * 100, 2) if npz_time > 0 else 0
        }
    
    return comparison


def scan_models_directory(models_dir: str = "models") -> Dict[str, Any]:
    """Scan models directory and return comprehensive info"""
    if not os.path.exists(models_dir):
        return {'error': f"Models directory '{models_dir}' does not exist"}
    
    scan_result = {
        'models_directory': models_dir,
        'scan_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'users': {},
        'summary': {
            'total_users': 0,
            'total_models': 0,
            'ngram_models': 0,
            'neural_models': 0,
            'safetensors_models': 0,
            'legacy_npz_models': 0
        }
    }
    
    for user_item in os.listdir(models_dir):
        user_path = os.path.join(models_dir, user_item)
        if os.path.isdir(user_path):
            scan_result['users'][user_item] = {
                'models': [],
                'model_count': 0
            }
            scan_result['summary']['total_users'] += 1
            
            for file_item in os.listdir(user_path):
                file_path = os.path.join(user_path, file_item)
                
                if file_item.endswith('_ngram.json'):
                    model_name = file_item.replace('_ngram.json', '')
                    model_info = {
                        'name': model_name,
                        'type': 'ngram',
                        'format': 'json',
                        'file_size': format_bytes(get_file_size(file_path))
                    }
                    scan_result['users'][user_item]['models'].append(model_info)
                    scan_result['summary']['ngram_models'] += 1
                    scan_result['summary']['total_models'] += 1
                
                elif file_item.endswith('_neural.safetensors'):
                    model_name = file_item.replace('_neural.safetensors', '')
                    model_info = {
                        'name': model_name,
                        'type': 'neural',
                        'format': 'safetensors',
                        'file_size': format_bytes(get_file_size(file_path)),
                        'safetensors_info': load_safetensors_info(file_path)
                    }
                    scan_result['users'][user_item]['models'].append(model_info)
                    scan_result['summary']['neural_models'] += 1
                    scan_result['summary']['safetensors_models'] += 1
                    scan_result['summary']['total_models'] += 1
                
                elif file_item.endswith('_neural.npz'):
                    model_name = file_item.replace('_neural.npz', '')
                    model_info = {
                        'name': model_name,
                        'type': 'neural',
                        'format': 'legacy_npz',
                        'file_size': format_bytes(get_file_size(file_path))
                    }
                    scan_result['users'][user_item]['models'].append(model_info)
                    scan_result['summary']['neural_models'] += 1
                    scan_result['summary']['legacy_npz_models'] += 1
                    scan_result['summary']['total_models'] += 1
            
            scan_result['users'][user_item]['model_count'] = len(scan_result['users'][user_item]['models'])
    
    return scan_result


def cleanup_backup_files(models_dir: str = "models", dry_run: bool = True) -> Dict[str, Any]:
    """Clean up .backup files from model directory"""
    cleanup_result = {
        'models_directory': models_dir,
        'dry_run': dry_run,
        'files_found': [],
        'files_removed': [],
        'errors': []
    }
    
    if not os.path.exists(models_dir):
        cleanup_result['errors'].append(f"Models directory '{models_dir}' does not exist")
        return cleanup_result
    
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.backup'):
                backup_path = os.path.join(root, file)
                cleanup_result['files_found'].append(backup_path)
                
                if not dry_run:
                    try:
                        os.remove(backup_path)
                        cleanup_result['files_removed'].append(backup_path)
                    except Exception as e:
                        cleanup_result['errors'].append(f"Failed to remove {backup_path}: {e}")
    
    if dry_run:
        cleanup_result['message'] = f"Found {len(cleanup_result['files_found'])} backup files. Use dry_run=False to remove them."
    else:
        cleanup_result['message'] = f"Removed {len(cleanup_result['files_removed'])} backup files."
    
    return cleanup_result


def validate_safetensors_integrity(filepath: str) -> Dict[str, Any]:
    """Validate SafeTensors file integrity"""
    validation_result = {
        'filepath': filepath,
        'valid': False,
        'checks': {}
    }
    
    if not os.path.exists(filepath):
        validation_result['checks']['file_exists'] = False
        validation_result['error'] = "File does not exist"
        return validation_result
    
    validation_result['checks']['file_exists'] = True
    
    try:
        # Try to load the file
        tensors = load_file(filepath)
        validation_result['checks']['loadable'] = True
        validation_result['checks']['tensor_count'] = len(tensors)
        
        # Check tensor properties
        all_valid = True
        tensor_checks = {}
        
        for name, tensor in tensors.items():
            tensor_info = {
                'shape_valid': len(tensor.shape) > 0,
                'dtype_valid': hasattr(tensor, 'dtype'),
                'data_valid': not np.any(np.isnan(tensor)) if tensor.dtype.kind == 'f' else True
            }
            tensor_checks[name] = tensor_info
            
            if not all(tensor_info.values()):
                all_valid = False
        
        validation_result['checks']['tensors'] = tensor_checks
        validation_result['checks']['all_tensors_valid'] = all_valid
        validation_result['valid'] = all_valid
        
    except Exception as e:
        validation_result['checks']['loadable'] = False
        validation_result['error'] = str(e)
    
    return validation_result


def export_model_report(models_dir: str = "models", output_file: str = "model_report.json") -> str:
    """Export comprehensive model report to JSON file"""
    report = {
        'report_info': {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models_directory': models_dir,
            'tinyllm_version': '1.0.0'  # You can make this dynamic
        },
        'directory_scan': scan_models_directory(models_dir),
        'format_comparisons': {},
        'validation_results': {}
    }
    
    # Add format comparisons for users with both NPZ and SafeTensors
    scan_result = report['directory_scan']
    if 'users' in scan_result:
        for user_id, user_info in scan_result['users'].items():
            models = user_info['models']
            model_names = set()
            
            for model in models:
                model_names.add(model['name'])
            
            for model_name in model_names:
                comparison = compare_model_formats(user_id, model_name, models_dir)
                if 'differences' in comparison:
                    report['format_comparisons'][f"{user_id}/{model_name}"] = comparison
    
    # Validate all SafeTensors files
    if 'users' in scan_result:
        for user_id, user_info in scan_result['users'].items():
            for model in user_info['models']:
                if model['format'] == 'safetensors':
                    safetensors_path = os.path.join(models_dir, user_id, f"{model['name']}_neural.safetensors")
                    validation = validate_safetensors_integrity(safetensors_path)
                    report['validation_results'][f"{user_id}/{model['name']}"] = validation
    
    # Write report to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Model report exported to: {output_file}")
    return output_file
