"""
CrewAI Tools Package Initialization
Location: src/contextual_rag/agents/tools/__init__.py
"""

# Import all tools
try:
    from .vector_search_tool import VectorSearchTool
    VECTOR_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Vector search tool import failed: {e}")
    VECTOR_SEARCH_AVAILABLE = False

try:
    from .reranking_tool import RerankingTool
    RERANKING_AVAILABLE = True
except ImportError as e:
    print(f"Reranking tool import failed: {e}")
    RERANKING_AVAILABLE = False

try:
    from .llm_generation_tool import LLMGenerationTool
    LLM_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"LLM generation tool import failed: {e}")
    LLM_GENERATION_AVAILABLE = False

try:
    from .evaluation_tool import EvaluationTool
    EVALUATION_AVAILABLE = True
except ImportError as e:
    print(f"Evaluation tool import failed: {e}")
    EVALUATION_AVAILABLE = False

# Export available tools
__all__ = []
if VECTOR_SEARCH_AVAILABLE:
    __all__.append('VectorSearchTool')
if RERANKING_AVAILABLE:
    __all__.append('RerankingTool')
if LLM_GENERATION_AVAILABLE:
    __all__.append('LLMGenerationTool')
if EVALUATION_AVAILABLE:
    __all__.append('EvaluationTool')

# Version and metadata
__version__ = "1.0.0"
__author__ = "RAG Team"
__description__ = "CrewAI tools for contextual RAG system"

# Tool registry for easy access
AVAILABLE_TOOLS = {}
if VECTOR_SEARCH_AVAILABLE:
    AVAILABLE_TOOLS['vector_search'] = VectorSearchTool
if RERANKING_AVAILABLE:
    AVAILABLE_TOOLS['reranking'] = RerankingTool
if LLM_GENERATION_AVAILABLE:
    AVAILABLE_TOOLS['llm_generation'] = LLMGenerationTool
if EVALUATION_AVAILABLE:
    AVAILABLE_TOOLS['evaluation'] = EvaluationTool

def get_tool(tool_name: str):
    """Get tool class by name"""
    return AVAILABLE_TOOLS.get(tool_name)

def list_available_tools():
    """List all available tools"""
    return list(AVAILABLE_TOOLS.keys())

def create_all_tools():
    """Create instances of all tools"""
    return {name: tool_class() for name, tool_class in AVAILABLE_TOOLS.items()}

def get_system_status():
    """Get status of all tools"""
    return {
        'vector_search': VECTOR_SEARCH_AVAILABLE,
        'reranking': RERANKING_AVAILABLE,
        'llm_generation': LLM_GENERATION_AVAILABLE,
        'evaluation': EVALUATION_AVAILABLE,
        'total_available': len(AVAILABLE_TOOLS)
    }