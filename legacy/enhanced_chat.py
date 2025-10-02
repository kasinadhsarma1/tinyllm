"""
Enhanced Interactive Chat Bot for Unified Hello World LLM
Multi-personality chat system with user-based models
"""

import numpy as np
import json
import os
from collections import defaultdict

# Import from the unified LLM file
try:
    from unified_llm import UnifiedLLM, UserLLMManager, HelloWorldDataset
except ImportError:
    print("Please ensure unified_llm.py is in the same directory!")
    exit(1)


class ChatPersonality:
    """Chat personality using UnifiedLLM"""
    
    def __init__(self, user_id, model_name="chat_model", model_type="ngram"):
        self.user_id = user_id
        self.model_name = model_name
        self.model_type = model_type
        self.conversation_history = []
        self.temperature = 0.8
        self.model = None
        self.manager = UserLLMManager()
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            # Try to load existing model
            self.model = self.manager.load_user_model(
                self.user_id, self.model_name, self.model_type
            )
            print(f"✓ Loaded existing {self.model_type} model for {self.user_id}")
            return True
        except:
            # Create new model
            print(f"Creating new {self.model_type} model for {self.user_id}...")
            if self.model_type == "ngram":
                self.model = self.manager.create_user_model(
                    self.user_id, "ngram", n=4
                )
            else:
                self.model = self.manager.create_user_model(
                    self.user_id, "neural", vocab_size=50
                )
            
            # Train with default dataset
            dataset = HelloWorldDataset()
            self.model.train(dataset.texts)
            self.model.save(self.model_name)
            print(f"✓ Created and trained new model for {self.user_id}")
            return True
    
    def respond(self, user_input):
        """Generate response to user input"""
        if self.model is None:
            return "Model not loaded!"
        
        # Use user input as context for generation
        prompt = user_input.lower()[:3] if user_input else ""
        
        # Generate response
        response = self.model.generate(
            prompt=prompt,
            max_length=25,
            temperature=self.temperature
        )
        
        # Clean response
        response = response.strip()
        if not response or response == prompt:
            response = "hello there!"
        
        # Store conversation
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'personality': self.user_id
        })
        
        return response
    
    def get_info(self):
        """Get personality info"""
        if self.model:
            model_info = self.model.get_model_info()
            return {
                'user_id': self.user_id,
                'model_type': self.model_type,
                'model_name': self.model_name,
                'temperature': self.temperature,
                'conversations': len(self.conversation_history),
                'model_info': model_info
            }
        return {'user_id': self.user_id, 'status': 'not loaded'}


class MultiPersonalityChat:
    """Chat system with multiple AI personalities"""
    
    def __init__(self):
        self.personalities = {}
        self.current_personality = None
        self.manager = UserLLMManager()
    
    def add_personality(self, user_id, model_type="ngram", model_name="chat_model"):
        """Add a new personality"""
        personality = ChatPersonality(user_id, model_name, model_type)
        if personality.load_or_create_model():
            self.personalities[user_id] = personality
            print(f"✓ Added personality: {user_id}")
            return True
        return False
    
    def switch_personality(self, user_id):
        """Switch to a different personality"""
        if user_id in self.personalities:
            self.current_personality = user_id
            print(f"✓ Switched to {user_id}")
            return True
        else:
            print(f"❌ Personality {user_id} not found")
            return False
    
    def chat_with_current(self, message):
        """Chat with current personality"""
        if self.current_personality and self.current_personality in self.personalities:
            personality = self.personalities[self.current_personality]
            return personality.respond(message)
        return "No personality selected!"
    
    def list_personalities(self):
        """List all personalities"""
        return list(self.personalities.keys())
    
    def get_personality_info(self, user_id=None):
        """Get info about personality"""
        if user_id is None:
            user_id = self.current_personality
        
        if user_id and user_id in self.personalities:
            return self.personalities[user_id].get_info()
        return None


def print_chat_header():
    """Print enhanced chat header"""
    print("\n" + "="*70)
    print("🤖 UNIFIED HELLO WORLD LLM - MULTI-PERSONALITY CHAT")
    print("="*70)
    print("Features:")
    print("  💬 Chat with different AI personalities")
    print("  🔄 Switch between N-gram and Neural models")
    print("  👥 Manage multiple user-specific models")
    print("  📊 View model information and statistics")
    print("\nCommands:")
    print("  'add <name> <type>' - Add personality (type: ngram/neural)")
    print("  'switch <name>' - Switch to personality")
    print("  'list' - List all personalities")
    print("  'info [name]' - Show personality info")
    print("  'temp <value>' - Set temperature (0.1-2.0)")
    print("  'history' - Show conversation history")
    print("  'help' - Show this help")
    print("  'quit' - Exit chat")
    print("="*70 + "\n")


def enhanced_interactive_chat():
    """Enhanced interactive chat with multiple personalities"""
    
    chat_system = MultiPersonalityChat()
    
    print_chat_header()
    
    # Setup default personalities
    print("Setting up default personalities...")
    chat_system.add_personality("alice", "ngram", "chat_model")
    chat_system.add_personality("bob", "neural", "smart_model")
    chat_system.switch_personality("alice")
    
    print(f"✓ Ready! Current personality: {chat_system.current_personality}")
    print("Type 'help' for commands or start chatting!\n")
    
    while True:
        try:
            # Show current personality in prompt
            current = chat_system.current_personality or "none"
            user_input = input(f"[{current}] You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for chatting!\n")
                break
            
            elif user_input.lower() == 'help':
                print_chat_header()
                continue
            
            elif user_input.lower().startswith('add '):
                parts = user_input.split()
                if len(parts) >= 3:
                    name = parts[1]
                    model_type = parts[2].lower()
                    if model_type in ['ngram', 'neural']:
                        chat_system.add_personality(name, model_type)
                    else:
                        print("❌ Model type must be 'ngram' or 'neural'")
                else:
                    print("❌ Usage: add <name> <type>")
                continue
            
            elif user_input.lower().startswith('switch '):
                name = user_input.split()[1] if len(user_input.split()) > 1 else ""
                chat_system.switch_personality(name)
                continue
            
            elif user_input.lower() == 'list':
                personalities = chat_system.list_personalities()
                print(f"🤖 Available personalities: {personalities}")
                print(f"Current: {chat_system.current_personality}")
                continue
            
            elif user_input.lower().startswith('info'):
                parts = user_input.split()
                name = parts[1] if len(parts) > 1 else None
                info = chat_system.get_personality_info(name)
                if info:
                    print("\n📊 Personality Info:")
                    print("-" * 50)
                    for key, value in info.items():
                        if key != 'model_info':
                            print(f"{key}: {value}")
                        else:
                            print("Model details:")
                            for k, v in value.items():
                                print(f"  {k}: {v}")
                    print("-" * 50)
                else:
                    print("❌ Personality not found")
                continue
            
            elif user_input.lower().startswith('temp '):
                try:
                    temp_value = float(user_input.split()[1])
                    if 0.1 <= temp_value <= 2.0:
                        if chat_system.current_personality:
                            personality = chat_system.personalities[chat_system.current_personality]
                            personality.temperature = temp_value
                            print(f"✓ Temperature set to {temp_value} for {chat_system.current_personality}")
                        else:
                            print("❌ No personality selected")
                    else:
                        print("⚠ Temperature must be between 0.1 and 2.0")
                except:
                    print("⚠ Invalid temperature value")
                continue
            
            elif user_input.lower() == 'history':
                if chat_system.current_personality:
                    personality = chat_system.personalities[chat_system.current_personality]
                    history = personality.conversation_history
                    print(f"\n📜 Conversation History for {chat_system.current_personality}:")
                    print("-" * 60)
                    if not history:
                        print("No conversation yet!")
                    else:
                        for i, exchange in enumerate(history[-10:], 1):  # Show last 10
                            print(f"{i}. You: {exchange['user']}")
                            print(f"   {exchange['personality']}: {exchange['bot']}")
                    print("-" * 60)
                else:
                    print("❌ No personality selected")
                continue
            
            # Generate response from current personality
            response = chat_system.chat_with_current(user_input)
            print(f"{chat_system.current_personality}: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"⚠ Error: {e}\n")


def simple_chat():
    """Simple single personality chat"""
    print("\n🤖 SIMPLE CHAT MODE")
    print("="*40)
    
    manager = UserLLMManager()
    dataset = HelloWorldDataset()
    
    user_id = input("Enter your name: ").strip() or "user"
    
    print("\nModel types:")
    print("1. N-gram (fast)")
    print("2. Neural (smart)")
    
    choice = input("Choose (1-2): ").strip()
    model_type = "ngram" if choice == "1" else "neural"
    
    # Create or load model
    try:
        model = manager.load_user_model(user_id, "simple_chat", model_type)
        print(f"✓ Loaded your {model_type} model!")
    except:
        print(f"Creating new {model_type} model...")
        if model_type == "ngram":
            model = manager.create_user_model(user_id, "ngram", n=3)
        else:
            model = manager.create_user_model(user_id, "neural", vocab_size=40)
        
        model.train(dataset.texts)
        model.save("simple_chat")
        print("✓ Model ready!")
    
    print(f"\n💬 Chat with your {model_type} model!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input(f"{user_id}: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print(f"\n👋 Goodbye {user_id}!")
            break
        
        if not user_input:
            continue
        
        response = model.generate(
            prompt=user_input[:3],
            max_length=20,
            temperature=0.8
        )
        
        print(f"Bot: {response}\n")


def personality_demo():
    """Demo different personalities side by side"""
    print("\n🎭 PERSONALITY COMPARISON DEMO")
    print("="*50)
    
    chat_system = MultiPersonalityChat()
    
    # Create personalities
    print("Creating different personalities...")
    chat_system.add_personality("creative", "ngram", "creative")
    chat_system.add_personality("logical", "neural", "logical")
    
    test_messages = ["hello", "how are you", "what's new", "goodbye"]
    
    print("\n💬 Comparing responses:\n")
    
    for msg in test_messages:
        print(f"You: {msg}")
        
        # Creative response
        chat_system.switch_personality("creative")
        creative_response = chat_system.chat_with_current(msg)
        print(f"  Creative: {creative_response}")
        
        # Logical response
        chat_system.switch_personality("logical")
        logical_response = chat_system.chat_with_current(msg)
        print(f"  Logical: {logical_response}")
        
        print()
    
    print("✓ Demo complete! Notice how different personalities respond differently!")


def main():
    print("\n🚀 Enhanced Hello World LLM Chat!")
    print("\nChoose mode:")
    print("1. Enhanced Multi-Personality Chat")
    print("2. Simple Single Chat")
    print("3. Personality Comparison Demo")
    print("4. Exit")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '1':
        enhanced_interactive_chat()
    elif choice == '2':
        simple_chat()
    elif choice == '3':
        personality_demo()
        
        if input("\nTry enhanced chat? (y/n): ").lower() == 'y':
            enhanced_interactive_chat()
    else:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
