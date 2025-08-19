"""
Reranker Agent for CrewAI RAG System
Location: src/contextual_rag/agents/reranker_agent.py
"""

from crewai import Agent
from .tools.reranking_tool import RerankingTool
from typing import Dict, Any, List
import logging

class RerankerAgent:
    """
    Specialized agent for document reranking using BGE reranker model
    Refines document relevance after initial vector retrieval
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reranking_tool = RerankingTool()
        
        # Create CrewAI agent
        self.agent = Agent(
            role="Document Relevance Specialist",
            goal="Rerank retrieved documents to optimize relevance and remove noise for better context quality",
            backstory="""You are an expert in document relevance assessment and ranking optimization. 
            Your specialty lies in understanding semantic relationships between queries and documents, 
            going beyond simple keyword matching. You use advanced reranking models to ensure that 
            only the most contextually relevant documents are passed forward for response generation. 
            Your work significantly improves the quality and accuracy of final responses by filtering 
            out tangentially related content.""",
            tools=[],  # Remove tools for now
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
    
    def rerank_documents(self, 
                        query: str, 
                        retrieved_documents: List[Dict[str, Any]], 
                        top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents for better relevance
        
        Args:
            query: Original search query
            retrieved_documents: Documents from retrieval stage
            top_k: Number of top documents to return after reranking
        
        Returns:
            List of reranked documents with relevance scores
        """
        try:
            if not retrieved_documents:
                self.logger.warning("⚠️ No documents provided for reranking")
                return []
            
            self.logger.info(f"🔄 Reranking {len(retrieved_documents)} documents for query: '{query[:50]}...'")
            
            # Perform reranking
            reranked_docs = self.reranking_tool.run(query, retrieved_documents, top_k)
            
            if reranked_docs:
                self._log_reranking_results(query, retrieved_documents, reranked_docs)
                return reranked_docs
            else:
                self.logger.warning("⚠️ Reranking failed, returning original documents")
                return retrieved_documents[:top_k]
                
        except Exception as e:
            self.logger.error(f"❌ Reranking failed: {e}")
            # Fallback to original documents
            return retrieved_documents[:top_k]
    
    def _log_reranking_results(self, 
                              query: str, 
                              original_docs: List[Dict[str, Any]], 
                              reranked_docs: List[Dict[str, Any]]):
        """Log reranking performance and changes"""
        
        self.logger.info(f"📊 Reranking Results:")
        self.logger.info(f"   • Input documents: {len(original_docs)}")
        self.logger.info(f"   • Output documents: {len(reranked_docs)}")
        
        # Compare original vs reranked order
        rank_changes = 0
        for i, reranked_doc in enumerate(reranked_docs):
            original_rank = reranked_doc.get('original_rank', i + 1)
            new_rank = i + 1
            
            if original_rank != new_rank:
                rank_changes += 1
            
            rerank_score = reranked_doc.get('rerank_score', 0)
            original_similarity = reranked_doc.get('similarity_score', 0)
            filename = reranked_doc.get('filename', 'unknown')
            
            self.logger.info(f"   • Rank {new_rank}: {filename} "
                           f"(was rank {original_rank}, rerank: {rerank_score:.3f}, "
                           f"similarity: {original_similarity:.3f})")
        
        self.logger.info(f"🔀 Rank changes made: {rank_changes}/{len(reranked_docs)}")
        
        # Calculate score improvements
        avg_rerank_score = sum(doc.get('rerank_score', 0) for doc in reranked_docs) / len(reranked_docs)
        avg_similarity_score = sum(doc.get('similarity_score', 0) for doc in reranked_docs) / len(reranked_docs)
        
        self.logger.info(f"📈 Score comparison:")
        self.logger.info(f"   • Average rerank score: {avg_rerank_score:.3f}")
        self.logger.info(f"   • Average similarity score: {avg_similarity_score:.3f}")
    
    def analyze_relevance_distribution(self, 
                                     query: str, 
                                     documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze relevance distribution of documents
        
        Args:
            query: Search query
            documents: List of documents to analyze
        
        Returns:
            Analysis of relevance distribution and quality
        """
        try:
            if not documents:
                return {'analysis': 'no_documents'}
            
            # Get rerank scores for all documents
            scored_docs = self.reranking_tool._run(query, documents, len(documents))
            
            if not scored_docs:
                return {'analysis': 'reranking_failed'}
            
            # Extract scores
            rerank_scores = [doc.get('rerank_score', 0) for doc in scored_docs]
            similarity_scores = [doc.get('similarity_score', 0) for doc in scored_docs]
            
            # Calculate statistics
            analysis = {
                'total_documents': len(documents),
                'rerank_scores': {
                    'mean': sum(rerank_scores) / len(rerank_scores),
                    'max': max(rerank_scores),
                    'min': min(rerank_scores),
                    'range': max(rerank_scores) - min(rerank_scores)
                },
                'similarity_scores': {
                    'mean': sum(similarity_scores) / len(similarity_scores),
                    'max': max(similarity_scores),
                    'min': min(similarity_scores),
                    'range': max(similarity_scores) - min(similarity_scores)
                },
                'quality_assessment': self._assess_document_quality(rerank_scores),
                'score_distribution': self._get_score_distribution(rerank_scores),
                'top_documents': scored_docs[:3]  # Top 3 for inspection
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Relevance analysis failed: {e}")
            return {'analysis': 'error', 'error': str(e)}
    
    def _assess_document_quality(self, scores: List[float]) -> str:
        """Assess overall quality of document set"""
        if not scores:
            return 'no_data'
        
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        score_range = max(scores) - min(scores)
        
        if avg_score >= 0.7 and min_score >= 0.5:
            return 'excellent'
        elif avg_score >= 0.6 and min_score >= 0.3:
            return 'good'
        elif avg_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get distribution of scores across quality bands"""
        distribution = {
            'excellent': 0,    # 0.8+
            'good': 0,        # 0.6-0.8
            'fair': 0,        # 0.4-0.6
            'poor': 0         # <0.4
        }
        
        for score in scores:
            if score >= 0.8:
                distribution['excellent'] += 1
            elif score >= 0.6:
                distribution['good'] += 1
            elif score >= 0.4:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def adaptive_rerank(self, 
                       query: str, 
                       documents: List[Dict[str, Any]], 
                       quality_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Adaptive reranking that adjusts top_k based on quality
        
        Args:
            query: Search query
            documents: Retrieved documents
            quality_threshold: Minimum quality threshold for inclusion
        
        Returns:
            Adaptively filtered and reranked documents
        """
        try:
            # First pass: rerank all documents
            all_reranked = self.reranking_tool._run(query, documents, len(documents))
            
            if not all_reranked:
                return documents[:3]  # Fallback
            
            # Filter by quality threshold
            quality_docs = [
                doc for doc in all_reranked 
                if doc.get('rerank_score', 0) >= quality_threshold
            ]
            
            # Adaptive top_k selection
            if len(quality_docs) >= 3:
                # If we have enough quality documents, use them
                selected_docs = quality_docs[:5]  # Take top 5 quality docs
            elif len(quality_docs) >= 1:
                # If we have some quality docs, supplement with best of remainder
                remaining_docs = [
                    doc for doc in all_reranked 
                    if doc.get('rerank_score', 0) < quality_threshold
                ]
                needed = 3 - len(quality_docs)
                selected_docs = quality_docs + remaining_docs[:needed]
            else:
                # If no docs meet threshold, take top 3 anyway
                selected_docs = all_reranked[:3]
            
            self.logger.info(f"🎯 Adaptive reranking:")
            self.logger.info(f"   • Quality threshold: {quality_threshold}")
            self.logger.info(f"   • Quality documents: {len(quality_docs)}")
            self.logger.info(f"   • Final selection: {len(selected_docs)}")
            
            return selected_docs
            
        except Exception as e:
            self.logger.error(f"❌ Adaptive reranking failed: {e}")
            return documents[:3]
    
    def test_reranker_system(self) -> Dict[str, Any]:
        """Test the reranker system"""
        try:
            # Test reranker model
            model_test = self.reranking_tool.test_reranker_model()
            
            # Test with sample documents
            test_query = "What are the procurement approval requirements?"
            test_docs = [
                {
                    'content': 'Procurement approval requires proper documentation and manager authorization for purchases above $1000.',
                    'filename': 'procurement_manual.pdf',
                    'similarity_score': 0.85,
                    'rank': 1
                },
                {
                    'content': 'HR policies include leave management and employee onboarding procedures.',
                    'filename': 'hr_policies.pdf',
                    'similarity_score': 0.75,
                    'rank': 2
                },
                {
                    'content': 'Procurement processes must follow established approval workflows and vendor management guidelines.',
                    'filename': 'procurement_process.pdf',
                    'similarity_score': 0.70,
                    'rank': 3
                }
            ]
            
            # Test reranking
            reranked = self.rerank_documents(test_query, test_docs, top_k=2)
            
            return {
                'status': 'success',
                'model_test': model_test,
                'test_query': test_query,
                'original_docs': len(test_docs),
                'reranked_docs': len(reranked),
                'reranking_working': len(reranked) > 0,
                'top_result': reranked[0] if reranked else None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'reranking_working': False
            }

def create_reranker_agent() -> RerankerAgent:
    """Factory function to create reranker agent"""
    return RerankerAgent()

if __name__ == "__main__":
    # Test the reranker agent
    agent = create_reranker_agent()
    
    # Test system
    test_result = agent.test_reranker_system()
    print(f"🧪 Reranker Agent Test: {test_result}")
    
    if test_result.get('status') == 'success':
        print(f"✅ Reranker system working correctly")
        top_result = test_result.get('top_result')
        if top_result:
            print(f"🥇 Top ranked document: {top_result.get('filename')} (score: {top_result.get('rerank_score', 0):.3f})")
    else:
        print(f"❌ Reranker system test failed: {test_result.get('error', 'Unknown error')}")