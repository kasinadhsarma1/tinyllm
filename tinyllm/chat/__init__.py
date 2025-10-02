"""
Chat module for TinyLLM
"""

from .personalities import ChatPersonality, MultiPersonalityChat, create_sample_personalities

__all__ = [
    'ChatPersonality',
    'MultiPersonalityChat', 
    'create_sample_personalities'
]
