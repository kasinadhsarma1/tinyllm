"""
TinyLLM: A User-Based Language Model System

A lightweight, user-focused language model system supporting both N-gram and Neural models
with SafeTensors integration for secure and efficient model serialization.

Features:
- User-based model isolation
- Multiple model types (N-gram, Neural)
- SafeTensors for secure model storage
- Interactive chat interfaces
- Multi-personality conversations
"""

__version__ = "1.0.0"
__author__ = "TinyLLM Team"
__description__ = "User-Based Language Model System with SafeTensors"

# Core functionality
from .core.unified_llm import UnifiedLLM
from .core.manager import UserLLMManager
from .core.dataset import HelloWorldDataset

# Individual models
from .models.base_models import NGramModel, NeuralModel

# Chat interfaces
from .chat.personalities import ChatPersonality, MultiPersonalityChat, create_sample_personalities

# Utility functions
from .utils.helpers import (
    scan_models_directory,
    compare_model_formats,
    export_model_report,
    validate_safetensors_integrity
)

__all__ = [
    "UnifiedLLM",
    "UserLLMManager", 
    "HelloWorldDataset",
    "NGramModel",
    "NeuralModel",
    "ChatPersonality",
    "MultiPersonalityChat",
    "create_sample_personalities",
    "scan_models_directory",
    "compare_model_formats",
    "export_model_report",
    "validate_safetensors_integrity"
]
