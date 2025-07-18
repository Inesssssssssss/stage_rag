import ollama

class LLMClient:
    def __init__(self, model_name: str = "llama3.1"):
        """
        Initialize the LLM client.
        
        Args:
            model_name: The name of the model to use (default: "llama3")
        """
        self.model_name = model_name
        self.messages = []
        self._load_model()
        
    def _load_model(self):
        """Explicitly load the model using Ollama."""
        try:
            # Check if model exists or pull it
            models = ollama.list()
            model_exists = any(model.model == self.model_name for model in models.models)
            
            if not model_exists:
                print(f"Model {self.model_name} not found locally. Pulling from Ollama...")
                ollama.pull(self.model_name)
            
            # quick gen to load the model
            _ = ollama.generate(
                model=self.model_name,
                prompt="Test",
                options={"num_predict": 1}
            )
            print(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise
        
    def reset_chat(self):
        """Reset the chat history."""
        self.messages = []
        self.messages.append(
            {
                'role': 'system',
                'content': 'You are a robot controller inside a simulator. Your ONLY job is to return a Python-style list with exactly one string that represents a robot primitive. Do not explain. Do not label. Do not add text before or after. Just return the list. Any extra text will cause the simulation to fail.'
            }
        )
        
    
    
    def generate(self, prompt: str, temperature: float = 0, max_tokens: int = 1024) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: The temperature parameter for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            The generated response as a string
        """
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        
        return response.get("response", "")
    
    def generate_im(self, prompt: str, image: str, temperature: float = 0, max_tokens: int = 1024) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: The temperature parameter for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            The generated response as a string
        """
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            images=[image],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        
        return response.get("response", "")
    
    def chat(self, text, format = None, temperature: float = 0, max_tokens: int = 4096) :
        self.messages.append(
            {
            'role': 'user',
            'content': text,
            }
        )
        
        response = ollama.chat(model=self.model_name, 
                               messages=self.messages, 
                               format=format,
                               options={
                                        "temperature": temperature,
                                        "num_predict": max_tokens
                                       })['message']['content']

        self.messages.append(
            {
            'role': 'assistant',
            'content': response,
            }
        )
        
        return response

        
    
