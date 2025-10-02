#!/usr/bin/env python3
"""
Simple interactive chat example for TinyLLM
"""

import sys
import os

# Add the package to Python path for development
sys.path.insert(0, '/home/kasinadhsarma/tinyllm')

from tinyllm.chat import create_sample_personalities


def main():
    """Run a simple interactive chat"""
    print("🎭 TinyLLM Interactive Chat")
    print("=" * 30)
    
    # Create chat system with pre-trained personalities
    print("🔄 Loading personalities...")
    chat_system = create_sample_personalities()
    
    print("\n✅ Available personalities:")
    personalities = chat_system.list_personalities()
    for p in personalities:
        print(f"  • {p['name']} ({p['model_type']})")
    
    print("\n💬 Starting interactive chat...")
    print("Commands: /switch <name>, /list, /quit")
    print("Current personality:", chat_system.get_current_personality_name())
    print()
    
    chat_system.interactive_chat()


if __name__ == "__main__":
    main()
