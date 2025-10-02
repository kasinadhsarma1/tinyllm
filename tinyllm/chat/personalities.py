"""
Chat interfaces for TinyLLM
"""

import json
import random
from typing import Dict, List, Optional
from ..core.unified_llm import UnifiedLLM


class ChatPersonality:
    """A chat personality powered by TinyLLM"""
    
    def __init__(self, name: str, model_type: str = "ngram", 
                 personality_traits: Optional[Dict] = None):
        self.name = name
        self.model = UnifiedLLM(model_type=model_type, user_id=f"chat_{name.lower()}")
        self.personality_traits = personality_traits or {}
        self.conversation_history = []
        
        # Default personality settings
        self.greeting = self.personality_traits.get("greeting", f"Hello! I'm {name}.")
        self.style = self.personality_traits.get("style", "friendly")
        self.response_length = self.personality_traits.get("response_length", 30)
    
    def train_on_data(self, training_text: str) -> None:
        """Train the personality's model on text data"""
        print(f"🧠 Training {self.name}'s {self.model.model_type} model...")
        self.model.train(training_text)
        print(f"✓ {self.name} has been trained!")
    
    def respond(self, user_input: str) -> str:
        """Generate a response to user input"""
        # Add personality-specific prompt engineering
        if self.style == "pirate":
            prompt = "Arrr, "
        elif self.style == "formal":
            prompt = "I believe "
        elif self.style == "cheerful":
            prompt = "Oh wow! "
        else:
            prompt = ""
        
        # Generate response
        if user_input.lower() in ["hello", "hi", "hey"]:
            response = self.greeting
        else:
            # Use last few characters as prompt
            prompt_text = user_input[-3:] if len(user_input) >= 3 else user_input
            response = self.model.generate(prompt + prompt_text, self.response_length)
        
        # Store conversation
        self.conversation_history.append({
            "user": user_input,
            "response": response
        })
        
        return response
    
    def save_personality(self, model_name: Optional[str] = None) -> None:
        """Save the personality's model"""
        name = model_name or self.name.lower()
        self.model.save(name)
        
        # Save personality metadata
        personality_data = {
            "name": self.name,
            "model_type": self.model.model_type,
            "personality_traits": self.personality_traits,
            "conversation_history": self.conversation_history[-10:]  # Last 10 conversations
        }
        
        user_dir = f"models/chat_{self.name.lower()}"
        import os
        os.makedirs(user_dir, exist_ok=True)
        
        with open(f"{user_dir}/{name}_personality.json", 'w') as f:
            json.dump(personality_data, f, indent=2)
        
        print(f"✓ {self.name}'s personality saved!")
    
    @classmethod
    def load_personality(cls, name: str, model_name: Optional[str] = None) -> 'ChatPersonality':
        """Load a saved personality"""
        user_id = f"chat_{name.lower()}"
        model_name = model_name or name.lower()
        
        # Load personality metadata
        personality_file = f"models/{user_id}/{model_name}_personality.json"
        
        try:
            with open(personality_file, 'r') as f:
                personality_data = json.load(f)
            
            # Load the model
            model = UnifiedLLM.load(user_id, model_name)
            
            # Create personality instance
            personality = cls(
                name=personality_data["name"],
                model_type=model.model_type,
                personality_traits=personality_data["personality_traits"]
            )
            personality.model = model
            personality.conversation_history = personality_data.get("conversation_history", [])
            
            print(f"✓ {name} personality loaded!")
            return personality
            
        except FileNotFoundError:
            print(f"❌ Personality '{name}' not found. Creating new one...")
            return cls(name)
    
    def get_stats(self) -> Dict:
        """Get personality statistics"""
        return {
            "name": self.name,
            "model_type": self.model.model_type,
            "model_info": self.model.get_model_info(),
            "conversations": len(self.conversation_history),
            "personality_traits": self.personality_traits
        }


class MultiPersonalityChat:
    """Chat system with multiple personalities"""
    
    def __init__(self):
        self.personalities: Dict[str, ChatPersonality] = {}
        self.current_personality = None
    
    def add_personality(self, name: str, model_type: str = "ngram", 
                       personality_traits: Optional[Dict] = None) -> ChatPersonality:
        """Add a new chat personality"""
        personality = ChatPersonality(name, model_type, personality_traits)
        self.personalities[name.lower()] = personality
        
        if self.current_personality is None:
            self.current_personality = name.lower()
        
        print(f"✓ Added personality: {name} ({model_type})")
        return personality
    
    def load_personality(self, name: str, model_name: Optional[str] = None) -> None:
        """Load a saved personality"""
        personality = ChatPersonality.load_personality(name, model_name)
        self.personalities[name.lower()] = personality
        
        if self.current_personality is None:
            self.current_personality = name.lower()
    
    def switch_personality(self, name: str) -> bool:
        """Switch to a different personality"""
        if name.lower() in self.personalities:
            self.current_personality = name.lower()
            print(f"🔄 Switched to {name}")
            return True
        else:
            print(f"❌ Personality '{name}' not found")
            return False
    
    def train_personality(self, name: str, training_text: str) -> bool:
        """Train a specific personality"""
        if name.lower() in self.personalities:
            self.personalities[name.lower()].train_on_data(training_text)
            return True
        else:
            print(f"❌ Personality '{name}' not found")
            return False
    
    def chat(self, user_input: str, personality_name: Optional[str] = None) -> str:
        """Chat with current or specified personality"""
        target_personality = personality_name or self.current_personality
        
        if not target_personality or target_personality not in self.personalities:
            return "❌ No personality available. Please add one first!"
        
        personality = self.personalities[target_personality]
        response = personality.respond(user_input)
        
        return f"{personality.name}: {response}"
    
    def list_personalities(self) -> List[Dict]:
        """List all available personalities"""
        personalities_info = []
        for name, personality in self.personalities.items():
            personalities_info.append(personality.get_stats())
        return personalities_info
    
    def save_all_personalities(self) -> None:
        """Save all personalities"""
        for personality in self.personalities.values():
            personality.save_personality()
        print("✓ All personalities saved!")
    
    def get_current_personality_name(self) -> Optional[str]:
        """Get the name of the current personality"""
        if self.current_personality:
            return self.personalities[self.current_personality].name
        return None
    
    def interactive_chat(self) -> None:
        """Start an interactive chat session"""
        print("🎭 Multi-Personality Chat Started!")
        print("Commands:")
        print("  /switch <name> - Switch personality")
        print("  /list - List personalities") 
        print("  /train <name> <text> - Train personality")
        print("  /save - Save all personalities")
        print("  /quit - Exit chat")
        print()
        
        while True:
            try:
                current_name = self.get_current_personality_name() or "None"
                user_input = input(f"[{current_name}] You: ").strip()
                
                if user_input.lower() == '/quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == '/list':
                    personalities = self.list_personalities()
                    print("\n📋 Available Personalities:")
                    for p in personalities:
                        status = "⭐" if p["name"].lower() == self.current_personality else "  "
                        print(f"{status} {p['name']} ({p['model_type']}) - {p['conversations']} conversations")
                    print()
                elif user_input.lower() == '/save':
                    self.save_all_personalities()
                elif user_input.startswith('/switch '):
                    name = user_input[8:].strip()
                    self.switch_personality(name)
                elif user_input.startswith('/train '):
                    parts = user_input[7:].split(' ', 1)
                    if len(parts) == 2:
                        name, text = parts
                        self.train_personality(name, text)
                    else:
                        print("❌ Usage: /train <name> <text>")
                elif user_input:
                    response = self.chat(user_input)
                    print(f"{response}\n")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def create_sample_personalities() -> MultiPersonalityChat:
    """Create sample personalities for demonstration"""
    chat_system = MultiPersonalityChat()
    
    # Add different personalities
    chat_system.add_personality("Alice", "ngram", {
        "greeting": "Hi there! I'm Alice, nice to meet you!",
        "style": "friendly",
        "response_length": 25
    })
    
    chat_system.add_personality("Bob", "neural", {
        "greeting": "Greetings. I am Bob.",
        "style": "formal", 
        "response_length": 20
    })
    
    chat_system.add_personality("Pirate", "ngram", {
        "greeting": "Ahoy matey! Captain Pirate here!",
        "style": "pirate",
        "response_length": 30
    })
    
    # Train with sample data
    alice_training = "Hello how are you today? I hope you're doing well! It's such a beautiful day outside. I love chatting with people and making new friends. What do you like to do for fun?"
    bob_training = "Good morning. I trust you are well. I am here to assist with any inquiries you may have. Please let me know how I can help you today. I prefer to maintain professional discourse."
    pirate_training = "Ahoy there matey! Welcome aboard me ship! Arr, we be sailin' the seven seas in search of treasure! Shiver me timbers, what brings ye to these waters? Yo ho ho!"
    
    chat_system.train_personality("Alice", alice_training)
    chat_system.train_personality("Bob", bob_training) 
    chat_system.train_personality("Pirate", pirate_training)
    
    return chat_system
