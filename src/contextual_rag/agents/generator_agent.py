"""
Generator Agent for CrewAI RAG System
Location: src/contextual_rag/agents/generator_agent.py
"""

from crewai import Agent
from .tools.llm_generation_tool import LLMGenerationTool
from typing import Dict, Any, List, Optional
import logging

class GeneratorAgent:
    """
    Specialized agent for response generation using Llama 3.2 1B model
    Creates contextual, accurate responses based on reranked documents
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.generation_tool = LLMGenerationTool()
        
        # Create CrewAI agent
        self.agent = Agent(
            role="Response Generation Specialist",
            goal="Generate accurate, contextual, and helpful responses based on retrieved documents while maintaining factual accuracy",
            backstory="""You are an expert content generator and communication specialist with 
            extensive experience in synthesizing information from multiple sources. Your strength 
            lies in creating clear, accurate, and well-structured responses that directly address 
            user questions while staying faithful to source documents. You excel at balancing 
            comprehensiveness with clarity, ensuring responses are both informative and accessible. 
            You have deep expertise in organizational policies, procedures, and documentation.""",
            tools=[],  # Remove tools for now
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
    
    def generate_response(self, 
                         query: str, 
                         context_documents: List[Dict[str, Any]], 
                         response_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Generate contextual response using reranked documents
        
        Args:
            query: User question
            context_documents: Reranked documents with relevance scores
            response_type: Type of response (comprehensive, concise, detailed)
        
        Returns:
            Generated response with metadata and source information
        """
        try:
            if not context_documents:
                self.logger.warning("⚠️ No context documents provided for generation")
                return self._generate_no_context_response(query)
            
            self.logger.info(f"🤖 Generating {response_type} response for query: '{query[:50]}...'")
            self.logger.info(f"📚 Using {len(context_documents)} context documents")
            
            # Generate response using LLM tool
            response_data = self.generation_tool.run(query, context_documents, response_type)
            
            # Enhance response with generation metadata
            enhanced_response = self._enhance_response_metadata(response_data, query, context_documents)
            
            self._log_generation_results(enhanced_response)
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"❌ Response generation failed: {e}")
            return self._create_error_response(query, str(e))
    
    def generate_with_citations(self, 
                               query: str, 
                               context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response with explicit source citations
        
        Args:
            query: User question
            context_documents: Reranked documents
        
        Returns:
            Response with numbered citations
        """
        try:
            self.logger.info(f"📝 Generating response with citations for: '{query[:50]}...'")
            
            # Use citation-enabled generation
            response_data = self.generation_tool.generate_with_citations(query, context_documents)
            
            # Enhance with metadata
            enhanced_response = self._enhance_response_metadata(response_data, query, context_documents)
            enhanced_response['has_citations'] = True
            
            self.logger.info(f"✅ Citation-enabled response generated")
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"❌ Citation generation failed: {e}")
            return self.generate_response(query, context_documents, "comprehensive")
    
    def adaptive_generate(self, 
                         query: str, 
                         context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adaptive generation that chooses response type based on query and context
        
        Args:
            query: User question
            context_documents: Available context
        
        Returns:
            Optimally generated response
        """
        try:
            # Analyze query to determine optimal response type
            query_analysis = self._analyze_query_for_generation(query)
            context_analysis = self._analyze_context_quality(context_documents)
            
            # Determine response strategy
            response_strategy = self._determine_response_strategy(query_analysis, context_analysis)
            
            self.logger.info(f"🧠 Adaptive generation strategy:")
            self.logger.info(f"   • Query type: {query_analysis['query_type']}")
            self.logger.info(f"   • Context quality: {context_analysis['quality_level']}")
            self.logger.info(f"   • Response strategy: {response_strategy['type']}")
            
            # Generate response using determined strategy
            if response_strategy['use_citations']:
                response = self.generate_with_citations(query, context_documents)
            else:
                response = self.generate_response(
                    query, 
                    context_documents, 
                    response_strategy['response_type']
                )
            
            # Add strategy metadata
            response['generation_strategy'] = response_strategy
            response['query_analysis'] = query_analysis
            response['context_analysis'] = context_analysis
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Adaptive generation failed: {e}")
            return self.generate_response(query, context_documents, "comprehensive")
    
    def _analyze_query_for_generation(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine generation approach"""
        query_lower = query.lower()
        
        # Query type classification
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            query_type = 'definition'
        elif any(word in query_lower for word in ['how', 'steps', 'process', 'procedure']):
            query_type = 'procedural'
        elif any(word in query_lower for word in ['why', 'reason', 'because']):
            query_type = 'explanatory'
        elif any(word in query_lower for word in ['list', 'examples', 'types']):
            query_type = 'listing'
        elif '?' in query:
            query_type = 'question'
        else:
            query_type = 'general'
        
        # Complexity assessment
        word_count = len(query.split())
        complexity = 'simple' if word_count <= 5 else 'medium' if word_count <= 10 else 'complex'
        
        # Specificity assessment
        specific_terms = ['approval', 'requirement', 'policy', 'procedure', 'guideline', 'standard']
        specificity = 'high' if any(term in query_lower for term in specific_terms) else 'medium'
        
        return {
            'query_type': query_type,
            'complexity': complexity,
            'specificity': specificity,
            'word_count': word_count,
            'requires_examples': query_type in ['listing', 'procedural'],
            'requires_detail': query_type in ['explanatory', 'procedural']
        }
    
    def _analyze_context_quality(self, context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality and characteristics of context documents"""
        if not context_documents:
            return {'quality_level': 'none', 'document_count': 0}
        
        # Extract relevance scores
        relevance_scores = [
            doc.get('rerank_score', doc.get('similarity_score', 0)) 
            for doc in context_documents
        ]
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        min_relevance = min(relevance_scores)
        
        # Determine quality level
        if avg_relevance >= 0.7 and min_relevance >= 0.5:
            quality_level = 'high'
        elif avg_relevance >= 0.5 and min_relevance >= 0.3:
            quality_level = 'medium'
        else:
            quality_level = 'low'
        
        # Check document diversity
        unique_files = len(set(doc.get('filename', '') for doc in context_documents))
        diversity = 'high' if unique_files >= 3 else 'medium' if unique_files >= 2 else 'low'
        
        return {
            'quality_level': quality_level,
            'document_count': len(context_documents),
            'avg_relevance': avg_relevance,
            'min_relevance': min_relevance,
            'diversity': diversity,
            'unique_sources': unique_files
        }
    
    def _determine_response_strategy(self, 
                                   query_analysis: Dict[str, Any], 
                                   context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal response generation strategy"""
        
        # Base response type on query analysis
        if query_analysis['query_type'] in ['definition', 'question']:
            base_response_type = 'concise'
        elif query_analysis['requires_detail']:
            base_response_type = 'detailed'
        else:
            base_response_type = 'comprehensive'
        
        # Adjust based on context quality
        if context_analysis['quality_level'] == 'low':
            response_type = 'concise'  # Don't over-elaborate with poor context
            use_citations = False
        elif context_analysis['quality_level'] == 'high':
            response_type = base_response_type
            use_citations = context_analysis['unique_sources'] >= 2
        else:
            response_type = base_response_type
            use_citations = False
        
        return {
            'type': response_type,
            'response_type': response_type,
            'use_citations': use_citations,
            'reasoning': f"Query: {query_analysis['query_type']}, Context: {context_analysis['quality_level']}"
        }
    
    def _enhance_response_metadata(self, 
                                  response_data: Dict[str, Any], 
                                  query: str, 
                                  context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance response with additional metadata"""
        
        # Calculate response metrics
        answer_length = len(response_data.get('answer', ''))
        word_count = len(response_data.get('answer', '').split())
        
        # Enhance with metrics
        enhanced = response_data.copy()
        enhanced.update({
            'response_metrics': {
                'character_count': answer_length,
                'word_count': word_count,
                'sentence_count': response_data.get('answer', '').count('.') + response_data.get('answer', '').count('!') + response_data.get('answer', '').count('?'),
                'avg_words_per_sentence': word_count / max(response_data.get('answer', '').count('.') + 1, 1)
            },
            'context_utilization': {
                'documents_provided': len(context_documents),
                'sources_cited': len(response_data.get('sources_used', [])),
                'utilization_rate': len(response_data.get('sources_used', [])) / max(len(context_documents), 1)
            },
            'generation_timestamp': response_data.get('generation_timestamp', ''),
            'quality_indicators': {
                'has_specific_details': 'requirement' in response_data.get('answer', '').lower() or 'policy' in response_data.get('answer', '').lower(),
                'references_sources': len(response_data.get('source_files', [])) > 0,
                'appropriate_length': 50 <= word_count <= 300
            }
        })
        
        return enhanced
    
    def _generate_no_context_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no context is available"""
        return self.generation_tool._generate_no_context_response(query)
    
    def _create_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'answer': f"I apologize, but I encountered an error while generating a response to your question: '{query}'. Please try rephrasing your question or contact support if the issue persists.",
            'query': query,
            'error': error,
            'generation_method': 'error_fallback',
            'context_sources': 0
        }
    
    def _log_generation_results(self, response_data: Dict[str, Any]):
        """Log generation results and metrics"""
        metrics = response_data.get('response_metrics', {})
        context_util = response_data.get('context_utilization', {})
        
        self.logger.info(f"📊 Generation Results:")
        self.logger.info(f"   • Response length: {metrics.get('word_count', 0)} words")
        self.logger.info(f"   • Sources used: {context_util.get('sources_cited', 0)}/{context_util.get('documents_provided', 0)}")
        self.logger.info(f"   • Utilization rate: {context_util.get('utilization_rate', 0):.2%}")
        self.logger.info(f"   • Generation method: {response_data.get('generation_method', 'unknown')}")
    
    def test_generator_system(self) -> Dict[str, Any]:
        """Test the generator system"""
        try:
            # Test LLM model
            model_test = self.generation_tool.test_llm_model()
            
            # Test with sample context
            test_query = "What are the procurement approval requirements?"
            test_context = [
                {
                    'content': 'Procurement approval requires proper documentation and manager authorization for purchases above $1000. All purchases must follow the established approval workflow.',
                    'filename': 'procurement_manual.pdf',
                    'rerank_score': 0.85,
                    'final_rank': 1
                }
            ]
            
            # Test generation
            response = self.generate_response(test_query, test_context, "comprehensive")
            
            return {
                'status': 'success',
                'model_test': model_test,
                'test_query': test_query,
                'response_generated': bool(response.get('answer')),
                'response_length': len(response.get('answer', '')),
                'sources_used': len(response.get('sources_used', [])),
                'generation_working': bool(response.get('answer')) and len(response.get('answer', '')) > 10
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'generation_working': False
            }

def create_generator_agent() -> GeneratorAgent:
    """Factory function to create generator agent"""
    return GeneratorAgent()

if __name__ == "__main__":
    # Test the generator agent
    agent = create_generator_agent()
    
    # Test system
    test_result = agent.test_generator_system()
    print(f"🧪 Generator Agent Test: {test_result}")
    
    if test_result.get('status') == 'success':
        print(f"✅ Generator system working correctly")
        print(f"📝 Response length: {test_result.get('response_length', 0)} characters")
        print(f"📚 Sources used: {test_result.get('sources_used', 0)}")
    else:
        print(f"❌ Generator system test failed: {test_result.get('error', 'Unknown error')}")