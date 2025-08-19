"""
RAGAs Evaluation Package for CrewAI RAG System
Location: src/contextual_rag/evaluation/__init__.py
"""

from .ragas_evaluator import OllamaRAGAsEvaluator

__all__ = ['OllamaRAGAsEvaluator']

__version__ = "1.0.0"
__description__ = "RAGAs evaluation using local Ollama models"
