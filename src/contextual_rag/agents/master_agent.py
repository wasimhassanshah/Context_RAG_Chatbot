"""
Master Agent for CrewAI RAG System - Main Orchestrator
Location: src/contextual_rag/agents/master_agent.py
"""

from crewai import Agent, Task, Crew
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Import specialized agents
from .retriever_agent import RetrieverAgent
from .reranker_agent import RerankerAgent
from .generator_agent import GeneratorAgent
from .evaluator_agent import EvaluatorAgent

class MasterAgent:
    """
    Master orchestrator agent that coordinates the entire RAG pipeline
    Routes queries through retrieval, reranking, generation, and evaluation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized agents
        self.retriever_agent = RetrieverAgent()
        self.reranker_agent = RerankerAgent()
        self.generator_agent = GeneratorAgent()
        self.evaluator_agent = EvaluatorAgent()
        
        # Create master CrewAI agent
        self.agent = Agent(
            role="RAG System Orchestrator",
            goal="Orchestrate the complete RAG pipeline to provide accurate, relevant, and high-quality responses to user queries",
            backstory="""You are the master orchestrator of an advanced RAG system with deep 
            expertise in information retrieval, document analysis, and response generation. 
            Your role is to coordinate a team of specialized agents to deliver the best possible 
            answers to user questions. You understand when to use different strategies based on 
            query complexity, manage the flow of information between agents, and ensure quality 
            at every step. You have extensive knowledge of organizational documents and can 
            route queries appropriately to get the most relevant and accurate responses.""",
            verbose=True,
            allow_delegation=True,
            max_iter=5
        )
        
        # Pipeline configuration
        self.pipeline_config = {
            'retrieval_k': 10,      # Initial retrieval count
            'reranking_k': 3,       # Final reranked documents
            'enable_evaluation': True,
            'adaptive_generation': True,
            'quality_threshold': 0.6
        }
    
    def process_query(self, 
                     query: str, 
                     response_type: str = "adaptive",
                     enable_citations: bool = False,
                     custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main query processing pipeline
        
        Args:
            query: User question
            response_type: Type of response (adaptive, comprehensive, concise, detailed)
            enable_citations: Whether to include citations
            custom_config: Custom pipeline configuration
        
        Returns:
            Complete response with metadata from all pipeline stages
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🎯 Processing query: '{query[:100]}...'")
            
            # Use custom config if provided
            config = {**self.pipeline_config, **(custom_config or {})}
            
            # Stage 1: Document Retrieval
            self.logger.info("📚 Stage 1: Document Retrieval")
            retrieval_result = self._execute_retrieval_stage(query, config)
            
            if not retrieval_result.get('documents'):
                return self._handle_no_results(query, start_time)
            
            # Stage 2: Document Reranking
            self.logger.info("🔄 Stage 2: Document Reranking")
            reranking_result = self._execute_reranking_stage(
                query, retrieval_result['documents'], config
            )
            
            # Stage 3: Response Generation
            self.logger.info("🤖 Stage 3: Response Generation")
            generation_result = self._execute_generation_stage(
                query, reranking_result['documents'], response_type, enable_citations
            )
            
            # Stage 4: Response Evaluation
            evaluation_result = None
            if config.get('enable_evaluation', True):
                self.logger.info("📊 Stage 4: Response Evaluation")
                evaluation_result = self._execute_evaluation_stage(
                    query, generation_result['answer'], reranking_result['documents']
                )
            
            # Compile final response
            final_response = self._compile_final_response(
                query, retrieval_result, reranking_result, 
                generation_result, evaluation_result, start_time
            )
            
            self._log_pipeline_completion(final_response)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            return self._create_error_response(query, str(e), start_time)
    
    def _execute_retrieval_stage(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document retrieval stage"""
        try:
            # Use intelligent retrieval strategy
            documents = self.retriever_agent.retrieve_with_strategy(query)
            
            # Limit to configured top_k
            top_k = config.get('retrieval_k', 10)
            documents = documents[:top_k]
            
            return {
                'documents': documents,
                'retrieval_count': len(documents),
                'strategy': 'intelligent',
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Retrieval stage failed: {e}")
            return {
                'documents': [],
                'retrieval_count': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_reranking_stage(self, 
                                query: str, 
                                documents: List[Dict[str, Any]], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document reranking stage"""
        try:
            if not documents:
                return {'documents': [], 'reranking_count': 0, 'status': 'no_input'}
            
            # Rerank with adaptive strategy or fixed top_k
            if config.get('adaptive_reranking', False):
                reranked_docs = self.reranker_agent.adaptive_rerank(
                    query, documents, config.get('quality_threshold', 0.6)
                )
            else:
                reranked_docs = self.reranker_agent.rerank_documents(
                    query, documents, config.get('reranking_k', 3)
                )
            
            return {
                'documents': reranked_docs,
                'reranking_count': len(reranked_docs),
                'original_count': len(documents),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Reranking stage failed: {e}")
            # Fallback to original documents
            top_k = config.get('reranking_k', 3)
            return {
                'documents': documents[:top_k],
                'reranking_count': len(documents[:top_k]),
                'status': 'fallback',
                'error': str(e)
            }
    
    def _execute_generation_stage(self, 
                                 query: str, 
                                 documents: List[Dict[str, Any]], 
                                 response_type: str,
                                 enable_citations: bool) -> Dict[str, Any]:
        """Execute response generation stage"""
        try:
            if not documents:
                return self.generator_agent._generate_no_context_response(query)
            
            # Choose generation method
            if enable_citations:
                response = self.generator_agent.generate_with_citations(query, documents)
            elif response_type == "adaptive":
                response = self.generator_agent.adaptive_generate(query, documents)
            else:
                response = self.generator_agent.generate_response(query, documents, response_type)
            
            response['status'] = 'success'
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Generation stage failed: {e}")
            return {
                'answer': f"I apologize, but I encountered an error generating a response: {str(e)}",
                'query': query,
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_evaluation_stage(self, 
                                 query: str, 
                                 response: str, 
                                 documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute response evaluation stage"""
        try:
            evaluation = self.evaluator_agent.evaluate_response(
                query, response, documents
            )
            evaluation['status'] = 'success'
            return evaluation
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation stage failed: {e}")
            return {
                'overall_score': 0.0,
                'detailed_scores': {},
                'status': 'error',
                'error': str(e),
                'recommendations': ['Evaluation failed - check system configuration']
            }
    
    def _compile_final_response(self, 
                               query: str,
                               retrieval_result: Dict[str, Any],
                               reranking_result: Dict[str, Any],
                               generation_result: Dict[str, Any],
                               evaluation_result: Optional[Dict[str, Any]],
                               start_time: datetime) -> Dict[str, Any]:
        """Compile final response with all pipeline metadata and individual RAGAs metrics"""
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # FIXED: Extract individual RAGAs metrics properly - NO ANSWER SIMILARITY
        individual_metrics = {}
        detailed_scores = {}
        
        if evaluation_result and 'detailed_scores' in evaluation_result:
            detailed_scores = evaluation_result['detailed_scores']
            individual_metrics = {
                'answer_relevancy': detailed_scores.get('answer_relevancy', 0),
                'context_precision': detailed_scores.get('context_precision', 0),
                'context_recall': detailed_scores.get('context_recall', 0),
                'answer_correctness': detailed_scores.get('answer_correctness', 0)
                # REMOVED: answer_similarity
            }
        
        # Log individual metrics for debugging
        self.logger.info(f"🔍 Individual metrics in master agent: {individual_metrics}")
        
        # FIXED: Calculate proper processing time including evaluation
        evaluation_time = evaluation_result.get('processing_time', 0) if evaluation_result else 0
        if evaluation_time > 0:
            processing_time = max(processing_time, evaluation_time)
        
        final_response = {
            # Main response data
            'answer': generation_result.get('answer', ''),
            'query': query,
            'processing_time_seconds': processing_time,  # FIXED: Proper processing time
            'timestamp': end_time.isoformat(),
            
            # Pipeline stage results
            'pipeline_stages': {
                'retrieval': {
                    'documents_found': retrieval_result.get('retrieval_count', 0),
                    'status': retrieval_result.get('status', 'unknown')
                },
                'reranking': {
                    'documents_reranked': reranking_result.get('reranking_count', 0),
                    'original_count': reranking_result.get('original_count', 0),
                    'status': reranking_result.get('status', 'unknown')
                },
                'generation': {
                    'response_length': len(generation_result.get('answer', '')),
                    'generation_method': generation_result.get('generation_method', 'unknown'),
                    'status': generation_result.get('status', 'unknown')
                },
                'evaluation': {
                    'overall_score': evaluation_result.get('overall_score', 0) if evaluation_result else None,
                    'detailed_scores': detailed_scores,  # Add detailed scores here
                    'individual_metrics': individual_metrics,  # Add individual metrics here
                    'status': evaluation_result.get('status', 'skipped') if evaluation_result else 'skipped'
                }
            },
            
            # Source information
            'sources': generation_result.get('sources_used', []),
            'source_files': generation_result.get('source_files', []),
            'context_utilization': generation_result.get('context_utilization', {}),
            
            # Quality metrics with individual scores
            'quality_metrics': {
                'evaluation_score': evaluation_result.get('overall_score', 0) if evaluation_result else None,
                'detailed_scores': detailed_scores,  # Add here too for easy access
                'individual_metrics': individual_metrics,  # Add here too for easy access
                'has_ground_truth': evaluation_result.get('has_ground_truth', False) if evaluation_result else False,
                'recommendations': evaluation_result.get('recommendations', []) if evaluation_result else []
            },
            
            # System metadata
            'system_info': {
                'pipeline_version': '1.0',
                'agents_used': ['retriever', 'reranker', 'generator', 'evaluator'],
                'configuration': self.pipeline_config
            }
        }
        
        # Add citations if available
        if generation_result.get('has_citations'):
            final_response['has_citations'] = True
        
        # Add any errors from stages
        errors = []
        for stage_name, stage_result in [
            ('retrieval', retrieval_result),
            ('reranking', reranking_result), 
            ('generation', generation_result),
            ('evaluation', evaluation_result or {})
        ]:
            if stage_result.get('error'):
                errors.append(f"{stage_name}: {stage_result['error']}")
        
        if errors:
            final_response['pipeline_errors'] = errors
        
        return final_response
    
    def _handle_no_results(self, query: str, start_time: datetime) -> Dict[str, Any]:
        """Handle case when no documents are retrieved"""
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            'answer': "I apologize, but I couldn't find relevant information in the available documents to answer your question. Please try rephrasing your question or contact support for assistance.",
            'query': query,
            'processing_time_seconds': processing_time,
            'timestamp': end_time.isoformat(),
            'pipeline_stages': {
                'retrieval': {'documents_found': 0, 'status': 'no_results'},
                'reranking': {'status': 'skipped'},
                'generation': {'status': 'no_context_fallback'},
                'evaluation': {'status': 'skipped'}
            },
            'sources': [],
            'source_files': [],
            'quality_metrics': {
                'evaluation_score': 0.0,
                'detailed_scores': {},
                'individual_metrics': {},
                'recommendations': ['No relevant documents found - consider expanding the knowledge base']
            }
        }
    
    def _create_error_response(self, query: str, error: str, start_time: datetime) -> Dict[str, Any]:
        """Create error response"""
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            'answer': f"I apologize, but I encountered a system error while processing your question. Please try again or contact support if the issue persists. Error: {error}",
            'query': query,
            'processing_time_seconds': processing_time,
            'timestamp': end_time.isoformat(),
            'error': error,
            'pipeline_stages': {
                'retrieval': {'status': 'error'},
                'reranking': {'status': 'error'},
                'generation': {'status': 'error'},
                'evaluation': {'status': 'error'}
            },
            'quality_metrics': {
                'evaluation_score': 0.0,
                'detailed_scores': {},
                'individual_metrics': {},
                'recommendations': ['System error occurred - check logs and system status']
            }
        }
    
    def _log_pipeline_completion(self, response: Dict[str, Any]):
        """Log pipeline completion statistics"""
        
        processing_time = response.get('processing_time_seconds', 0)
        stages = response.get('pipeline_stages', {})
        
        self.logger.info(f"🎉 Pipeline completed in {processing_time:.2f}s")
        self.logger.info(f"   • Documents retrieved: {stages.get('retrieval', {}).get('documents_found', 0)}")
        self.logger.info(f"   • Documents reranked: {stages.get('reranking', {}).get('documents_reranked', 0)}")
        self.logger.info(f"   • Response length: {stages.get('generation', {}).get('response_length', 0)} chars")
        
        evaluation_score = stages.get('evaluation', {}).get('overall_score')
        if evaluation_score is not None:
            self.logger.info(f"   • Evaluation score: {evaluation_score:.3f}")
    
    def batch_process_queries(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        
        self.logger.info(f"📋 Processing batch of {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            self.logger.info(f"🔄 Processing query {i+1}/{len(queries)}")
            result = self.process_query(query, **kwargs)
            results.append(result)
        
        # Add batch statistics
        batch_stats = self._calculate_batch_statistics(results)
        
        return {
            'individual_results': results,
            'batch_statistics': batch_stats,
            'total_queries': len(queries)
        }
    
    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for batch processing"""
        
        if not results:
            return {}
        
        processing_times = [r.get('processing_time_seconds', 0) for r in results]
        evaluation_scores = [
            r.get('quality_metrics', {}).get('evaluation_score', 0) 
            for r in results if r.get('quality_metrics', {}).get('evaluation_score') is not None
        ]
        
        return {
            'average_processing_time': sum(processing_times) / len(processing_times),
            'total_processing_time': sum(processing_times),
            'average_evaluation_score': sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0,
            'successful_responses': len([r for r in results if not r.get('error')]),
            'failed_responses': len([r for r in results if r.get('error')])
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Test all agents
            retriever_status = self.retriever_agent.test_retrieval_system()
            reranker_status = self.reranker_agent.test_reranker_system()
            generator_status = self.generator_agent.test_generator_system()
            evaluator_status = self.evaluator_agent.test_evaluator_system()
            
            # Overall system health
            all_working = all([
                retriever_status.get('status') == 'success',
                reranker_status.get('status') == 'success',
                generator_status.get('status') == 'success',
                evaluator_status.get('status') == 'success'
            ])
            
            return {
                'overall_status': 'operational' if all_working else 'degraded',
                'agent_status': {
                    'retriever': retriever_status,
                    'reranker': reranker_status,
                    'generator': generator_status,
                    'evaluator': evaluator_status
                },
                'pipeline_config': self.pipeline_config,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def create_master_agent() -> MasterAgent:
    """Factory function to create master agent"""
    return MasterAgent()

if __name__ == "__main__":
    # Test the master agent
    master = create_master_agent()
    
    # Test system status
    status = master.get_system_status()
    print(f"🧪 Master Agent System Status: {status['overall_status']}")
    
    # Test full pipeline if system is operational
    if status['overall_status'] == 'operational':
        print("\n🚀 Testing full RAG pipeline...")
        
        test_query = "What are the procurement approval requirements?"
        response = master.process_query(
            query=test_query,
            response_type="comprehensive",
            enable_citations=True
        )
        
        print(f"✅ Pipeline test completed!")
        print(f"📝 Query: {test_query}")
        print(f"🎯 Answer: {response['answer'][:200]}...")
        print(f"⏱️  Processing time: {response['processing_time_seconds']:.2f}s")
        print(f"📊 Evaluation score: {response.get('quality_metrics', {}).get('evaluation_score', 'N/A')}")
        print(f"📚 Sources used: {len(response.get('sources', []))}")
        
        # Show pipeline stages
        stages = response.get('pipeline_stages', {})
        print(f"\n🔄 Pipeline Stages:")
        for stage, info in stages.items():
            print(f"   • {stage.title()}: {info.get('status', 'unknown')}")
    
    else:
        print(f"❌ System not operational. Status: {status}")
        print("🔧 Agent status details:")
        for agent, agent_status in status.get('agent_status', {}).items():
            print(f"   • {agent}: {agent_status.get('status', 'unknown')}")