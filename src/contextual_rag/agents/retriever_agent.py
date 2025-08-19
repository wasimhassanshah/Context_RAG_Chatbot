"""
Retriever Agent for CrewAI RAG System
Location: src/contextual_rag/agents/retriever_agent.py
"""

from crewai import Agent
from .tools.vector_search_tool import VectorSearchTool
from typing import Dict, Any, List
import logging

class RetrieverAgent:
    """
    Specialized agent for document retrieval using vector search
    Responsible for finding relevant documents from PostgreSQL vector store
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vector_tool = VectorSearchTool()
        
        # Create CrewAI agent
        self.agent = Agent(
            role="Document Retrieval Specialist",
            goal="Find the most relevant documents from the knowledge base to answer user queries",
            backstory="""You are an expert document retrieval specialist with deep knowledge of 
            vector search and semantic similarity. Your role is to find the most relevant 
            document chunks from a large corpus of organizational documents including 
            procurement manuals, HR policies, and security guidelines. You excel at 
            understanding query intent and retrieving comprehensive context.""",
            tools=[],  # Remove tools for now to avoid BaseTool issues
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def retrieve_documents(self, 
                          query: str, 
                          top_k: int = 10, 
                          filename_filter: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (default 10 for reranking)
            filename_filter: Optional filename filter for specific documents
        
        Returns:
            List of retrieved documents with metadata
        """
        try:
            self.logger.info(f"🔍 Retrieving documents for query: '{query[:50]}...'")
            
            if filename_filter:
                # Search within specific document
                documents = self.vector_tool.search_by_filename(query, filename_filter, top_k)
                self.logger.info(f"📄 Found {len(documents)} documents in {filename_filter}")
            else:
                # General search across all documents
                documents = self.vector_tool.run(query, top_k)
                self.logger.info(f"📄 Found {len(documents)} documents across all files")
            
            # Log retrieval results
            if documents and not any('error' in doc for doc in documents):
                self._log_retrieval_stats(documents)
                return documents
            else:
                self.logger.warning(f"⚠️ No relevant documents found for query: {query}")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Document retrieval failed: {e}")
            return []
    
    def _log_retrieval_stats(self, documents: List[Dict[str, Any]]):
        """Log retrieval statistics"""
        if not documents:
            return
        
        # Calculate statistics
        similarity_scores = [doc.get('similarity_score', 0) for doc in documents]
        filenames = [doc.get('filename', 'unknown') for doc in documents]
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        unique_files = len(set(filenames))
        
        self.logger.info(f"📊 Retrieval Stats:")
        self.logger.info(f"   • Average similarity: {avg_similarity:.3f}")
        self.logger.info(f"   • Unique source files: {unique_files}")
        self.logger.info(f"   • Top similarity: {max(similarity_scores):.3f}")
        self.logger.info(f"   • Lowest similarity: {min(similarity_scores):.3f}")
        
        # Log file distribution
        file_counts = {}
        for filename in filenames:
            file_counts[filename] = file_counts.get(filename, 0) + 1
        
        self.logger.info(f"📁 File distribution:")
        for filename, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"   • {filename}: {count} chunks")
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine retrieval strategy
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query analysis and retrieval recommendations
        """
        query_lower = query.lower()
        
        # Document type detection
        document_hints = {
            'procurement': ['procurement', 'purchase', 'vendor', 'supplier', 'contract', 'buying'],
            'hr': ['hr', 'human resources', 'employee', 'personnel', 'hiring', 'leave', 'policy'],
            'security': ['security', 'information security', 'data protection', 'access', 'password'],
            'general': []
        }
        
        detected_types = []
        for doc_type, keywords in document_hints.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_types.append(doc_type)
        
        if not detected_types:
            detected_types = ['general']
        
        # Query complexity analysis
        word_count = len(query.split())
        complexity = 'simple' if word_count <= 5 else 'medium' if word_count <= 10 else 'complex'
        
        # Recommended retrieval parameters
        if complexity == 'simple':
            recommended_k = 5
        elif complexity == 'medium':
            recommended_k = 8
        else:
            recommended_k = 12
        
        return {
            'query': query,
            'detected_document_types': detected_types,
            'query_complexity': complexity,
            'word_count': word_count,
            'recommended_retrieval_k': recommended_k,
            'should_use_filename_filter': len(detected_types) == 1 and detected_types[0] != 'general'
        }
    
    def retrieve_with_strategy(self, query: str) -> List[Dict[str, Any]]:
        """
        Intelligent retrieval using query analysis
        
        Args:
            query: User query
            
        Returns:
            Retrieved documents using optimized strategy
        """
        # Analyze query
        analysis = self.analyze_query_intent(query)
        
        self.logger.info(f"🧠 Query Analysis:")
        self.logger.info(f"   • Document types: {analysis['detected_document_types']}")
        self.logger.info(f"   • Complexity: {analysis['query_complexity']}")
        self.logger.info(f"   • Recommended K: {analysis['recommended_retrieval_k']}")
        
        # Use filename filter for specific document types
        filename_filter = None
        if analysis['should_use_filename_filter']:
            doc_type = analysis['detected_document_types'][0]
            filename_mappings = {
                'procurement': 'Procurement',
                'hr': 'HR',
                'security': 'Security'
            }
            filename_filter = filename_mappings.get(doc_type)
        
        # Retrieve documents
        documents = self.retrieve_documents(
            query=query,
            top_k=analysis['recommended_retrieval_k'],
            filename_filter=filename_filter
        )
        
        return documents
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get database and retrieval system status"""
        try:
            stats = self.vector_tool.get_document_stats()
            
            return {
                'status': 'operational',
                'database_stats': stats,
                'vector_search_available': True,
                'total_documents': stats.get('total_chunks', 0)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'vector_search_available': False
            }
    
    def test_retrieval_system(self) -> Dict[str, Any]:
        """Test the retrieval system"""
        try:
            # Test query
            test_query = "What are the procurement approval requirements?"
            
            # Test retrieval
            documents = self.retrieve_documents(test_query, top_k=3)
            
            # Test query analysis
            analysis = self.analyze_query_intent(test_query)
            
            return {
                'status': 'success',
                'test_query': test_query,
                'documents_retrieved': len(documents),
                'query_analysis': analysis,
                'sample_document': documents[0] if documents else None,
                'retrieval_working': len(documents) > 0
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'retrieval_working': False
            }

def create_retriever_agent() -> RetrieverAgent:
    """Factory function to create retriever agent"""
    return RetrieverAgent()

if __name__ == "__main__":
    # Test the retriever agent
    agent = create_retriever_agent()
    
    # Test system
    test_result = agent.test_retrieval_system()
    print(f"🧪 Retriever Agent Test: {test_result}")
    
    # Test retrieval
    if test_result.get('status') == 'success':
        documents = agent.retrieve_with_strategy("What are the information security policies?")
        print(f"📄 Retrieved {len(documents)} documents")
        
        if documents:
            print(f"🔍 Top result: {documents[0].get('filename', 'unknown')} (score: {documents[0].get('similarity_score', 0):.3f})")