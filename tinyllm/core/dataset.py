"""
Dataset module for TinyLLM
Contains sample datasets for training models
"""

class HelloWorldDataset:
    """Dataset for training with hello world examples"""
    
    def __init__(self):
        self.texts = [
            "hello world", "hello there", "hi world", "greetings world",
            "hello friend", "hello everyone", "hi there", "hey world",
            "hello universe", "good morning world", "hello beautiful world",
            "hi friend", "hello to the world", "greetings everyone",
            "hello hello world", "world hello", "the world says hello",
            "hello from the world", "world of hello", "in a world we say hello",
        ]
    
    def get_texts(self):
        """Get all training texts"""
        return self.texts.copy()
    
    def add_text(self, text):
        """Add custom text to dataset"""
        if text and text not in self.texts:
            self.texts.append(text)
    
    def size(self):
        """Get dataset size"""
        return len(self.texts)
