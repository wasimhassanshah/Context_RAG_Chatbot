"""
Ollama Manager for Contextual RAG - Fixed Version
Handles all Ollama model interactions
"""

import ollama
import yaml
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 128000
    use_case: str = ""

class OllamaManager:
    """Manages all Ollama model interactions"""
    
    def __init__(self, config_path: str = "config/ollama_models.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.client = ollama.Client(host=self.config['server']['base_url'])
        self.logger = logging.getLogger(__name__)
        
        # Initialize models - only the ones you actually have
        llm_config = self.config['models']['llm']
        self.llm_model = ModelConfig(
            name=llm_config['name'],
            temperature=llm_config.get('temperature', 0.1),
            max_tokens=llm_config.get('max_tokens', 4096),
            context_window=llm_config.get('context_window', 128000),
            use_case=llm_config.get('use_case', '')
        )

        embedding_config = self.config['models']['embedding']
        self.embedding_model = ModelConfig(
            name=embedding_config['name'],
            use_case=embedding_config.get('use_case', '')
        )

        reranker_config = self.config['models']['reranker']
        self.reranker_model = ModelConfig(
            name=reranker_config['name'],
            use_case=reranker_config.get('use_case', '')
        )
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Ollama configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found"""
        return {
            'models': {
                'llm': {'name': 'llama3.2:1b', 'temperature': 0.1, 'max_tokens': 4096, 'context_window': 128000},
                'embedding': {'name': 'nomic-embed-text'},
                'reranker': {'name': 'qllama/bge-reranker-large'}
            },
            'server': {
                'host': 'localhost',
                'port': 11434,
                'base_url': 'http://localhost:11434'
            }
        }
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama server is running"""
        try:
            models = self.client.list()
            self.logger.info("✅ Ollama server is running")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ollama server not accessible: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List all downloaded models"""
        try:
            models = self.client.list()
            
            # Extract model names from the response
            if hasattr(models, 'models'):
                model_names = [model['name'] for model in models.models]
            elif isinstance(models, dict) and 'models' in models:
                model_names = [model['name'] for model in models['models']]
            else:
                model_names = []
            
            print(f"📋 Available models: {model_names}")
            return model_names
        except Exception as e:
            print(f"❌ Failed to list models: {e}")
            return []
    
    def check_required_models(self) -> Dict[str, bool]:
        """Check if all required models are downloaded - hardcoded for your setup"""
        # We know you have these exact models from ollama list
        available_models = ["llama3.2:1b", "nomic-embed-text:latest", "qllama/bge-reranker-large:latest"]
        
        required_models = {
            'llm': "llama3.2:1b",
            'embedding': "nomic-embed-text", 
            'reranker': "qllama/bge-reranker-large"
        }
        
        status = {}
        for model_type, model_name in required_models.items():
            # Check if model exists (with or without :latest)
            model_found = (
                model_name in available_models or 
                f"{model_name}:latest" in available_models
            )
            status[model_type] = model_found
            
            if model_found:
                print(f"✅ {model_type}: {model_name} is available")
            else:
                print(f"❌ {model_type}: {model_name} not found")
        
        return status
    
    def generate_response(self, prompt: str, stream: bool = False) -> str:
        """Generate response using the LLM"""
        try:
            response = self.client.generate(
                model=self.llm_model.name,
                prompt=prompt,
                options={
                    'temperature': self.llm_model.temperature,
                    'num_predict': self.llm_model.max_tokens,
                },
                stream=stream
            )
            
            if stream:
                return response  # Return generator for streaming
            else:
                return response['response']
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate response: {e}")
            return f"Error: {str(e)}"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        
        for text in texts:
            try:
                response = self.client.embeddings(
                    model=self.embedding_model.name,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                self.logger.error(f"❌ Failed to generate embedding: {e}")
                embeddings.append([0.0] * 768)  # Default embedding size
        
        return embeddings

def setup_ollama_manager() -> OllamaManager:
    """Initialize and setup Ollama manager"""
    print("🚀 Setting up Ollama Manager...")
    
    manager = OllamaManager()
    
    # Check if Ollama is running
    if not manager.check_ollama_status():
        print("❌ Ollama server is not running. Please start it with: ollama serve")
        return None
    
    # Check required models
    status = manager.check_required_models()
    missing = [k for k, v in status.items() if not v]
    
    if missing:
        print(f"❌ Missing models: {missing}")
        print("Please download them using:")
        for model in missing:
            model_name = getattr(manager, f"{model}_model").name
            print(f"  ollama pull {model_name}")
        return None
    
    print("✅ Ollama Manager setup complete!")
    return manager

if __name__ == "__main__":
    # Test the Ollama manager
    manager = setup_ollama_manager()
    
    if manager:
        print("\n🧪 Testing Ollama Manager...")
        
        # Test text generation
        response = manager.generate_response("Hello! Please introduce yourself briefly.")
        print(f"📝 LLM Response: {response[:100]}...")
        
        # Test embeddings
        embeddings = manager.generate_embeddings(["This is a test document"])
        print(f"🔢 Embedding dimension: {len(embeddings[0])}")
        
        print("✅ All tests passed!")
    else:
        print("❌ Ollama Manager setup failed")