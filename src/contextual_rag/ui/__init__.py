"""
Contextual RAG WebUI Module
"""

from .web_app import app, start_web_app
from .phoenix_manager import get_phoenix_manager

__all__ = ['app', 'start_web_app', 'get_phoenix_manager']
