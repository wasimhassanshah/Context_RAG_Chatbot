"""
CrewAI Main Orchestrator with Phoenix Integration
Location: src/contextual_rag/agents/crew_orchestrator.py
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# CrewAI imports
from crewai import Agent, Task, Crew, Process

# Phoenix integration imports
try:
    import phoenix as px
    from phoenix.trace import trace
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry import trace as trace_api
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    # Create dummy trace decorator when Phoenix is not available
    def trace(name):
        def decorator(func):
            return func
        return decorator

# Local imports
from .master_agent import MasterAgent
from .retriever_agent import RetrieverAgent
from .reranker_agent import RerankerAgent
from .generator_agent import GeneratorAgent
from .evaluator_agent import EvaluatorAgent

class CrewRAGOrchestrator:
    """
    Main orchestrator for CrewAI RAG system with Phoenix monitoring
    Manages the complete 4-agent pipeline with observability
    """
    
    def __init__(self, enable_phoenix: bool = True, phoenix_port: int = 6006):
        self.logger = logging.getLogger(__name__)
        self.enable_phoenix = enable_phoenix and PHOENIX_AVAILABLE
        self.phoenix_port = phoenix_port
        self.phoenix_session = None
        
        # Initialize Phoenix if available
        if self.enable_phoenix:
            self._setup_phoenix_monitoring()
        
        # Initialize master agent
        self.master_agent = MasterAgent()
        
        # Initialize individual agents
        self.retriever_agent = RetrieverAgent()
        self.reranker_agent = RerankerAgent()
        self.generator_agent = GeneratorAgent()
        self.evaluator_agent = EvaluatorAgent()
        
        # Create CrewAI crew
        self.crew = self._create_crew()
        
        # Session tracking
        self.session_stats = {
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'average_evaluation_score': 0.0,
            'session_start': datetime.now()
        }
        
        self.logger.info("🚀 CrewAI RAG Orchestrator initialized successfully")
        if self.enable_phoenix:
            self.logger.info(f"📊 Phoenix monitoring enabled on port {phoenix_port}")
    
    def _setup_phoenix_monitoring(self):
        """Setup Phoenix monitoring and tracing"""
        try:
            # Set environment variable for Phoenix port
            import os
            os.environ['PHOENIX_PORT'] = str(self.phoenix_port)
            
            # Launch Phoenix session (updated for Phoenix 11.x)
            self.phoenix_session = px.launch_app()
            
            self.logger.info(f"✅ Phoenix monitoring setup complete on port {self.phoenix_port}")
            self.logger.info(f"📊 Phoenix dashboard: http://localhost:{self.phoenix_port}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Phoenix setup failed: {e}")
            self.enable_phoenix = False
    
    def _create_crew(self):
        """Create CrewAI crew with all agents"""
        
        # Define tasks for the crew
        retrieval_task = Task(
            description="Retrieve relevant documents from the knowledge base based on the user query",
            agent=self.retriever_agent.agent,
            expected_output="List of relevant documents with similarity scores"
        )
        
        reranking_task = Task(
            description="Rerank retrieved documents to optimize relevance using BGE reranker",
            agent=self.reranker_agent.agent,
            expected_output="Top 3-5 reranked documents with relevance scores"
        )
        
        generation_task = Task(
            description="Generate accurate and contextual response based on reranked documents",
            agent=self.generator_agent.agent,
            expected_output="Complete response with source information"
        )
        
        evaluation_task = Task(
            description="Evaluate response quality using RAGAs metrics and provide improvement recommendations",
            agent=self.evaluator_agent.agent,
            expected_output="Quality scores and improvement recommendations"
        )
        
        # Create crew
        crew = Crew(
            agents=[
                self.retriever_agent.agent,
                self.reranker_agent.agent, 
                self.generator_agent.agent,
                self.evaluator_agent.agent
            ],
            tasks=[retrieval_task, reranking_task, generation_task, evaluation_task],
            process=Process.sequential,
            verbose=True
        )
        
        return crew
    
    @trace("rag_query_processing")
    def process_query(self, 
                     query: str, 
                     response_type: str = "adaptive",
                     enable_citations: bool = False,
                     use_crew: bool = False) -> Dict[str, Any]:
        """
        Process query using either master agent or CrewAI crew
        
        Args:
            query: User question
            response_type: Response type (adaptive, comprehensive, concise, detailed)
            enable_citations: Whether to include citations
            use_crew: Whether to use CrewAI crew (experimental) or master agent
        
        Returns:
            Complete response with tracing and monitoring
        """
        start_time = datetime.now()
        
        try:
            # Add Phoenix span attributes if enabled
            if self.enable_phoenix:
                try:
                    current_span = trace_api.get_current_span()
                    current_span.set_attribute("query", query)
                    current_span.set_attribute("response_type", response_type)
                    current_span.set_attribute("enable_citations", enable_citations)
                except:
                    pass  # Ignore tracing errors
            
            # Process query using master agent (recommended)
            if not use_crew:
                response = self.master_agent.process_query(
                    query=query,
                    response_type=response_type,
                    enable_citations=enable_citations
                )
            else:
                # Experimental: Use CrewAI crew directly
                response = self._process_with_crew(query, response_type, enable_citations)
            
            # Update session statistics
            self._update_session_stats(response)
            
            # Log to Phoenix if enabled
            if self.enable_phoenix:
                self._log_to_phoenix(query, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Query processing failed: {e}")
            
            error_response = {
                'answer': f"I apologize, but I encountered an error: {str(e)}",
                'query': query,
                'error': str(e),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
            
            return error_response
    
    def _process_with_crew(self, 
                          query: str, 
                          response_type: str, 
                          enable_citations: bool) -> Dict[str, Any]:
        """Process query using CrewAI crew (experimental)"""
        
        # Create context for the crew
        crew_context = {
            'query': query,
            'response_type': response_type,
            'enable_citations': enable_citations
        }
        
        # Execute crew tasks
        result = self.crew.kickoff(inputs=crew_context)
        
        # Convert crew result to standard format
        # Note: This is simplified - in practice, you'd need to extract
        # results from each agent and compile them
        return {
            'answer': str(result),
            'query': query,
            'generation_method': 'crewai_crew',
            'processing_time_seconds': 0,  # Would need to track this
            'sources': [],
            'quality_metrics': {}
        }
    
    def _update_session_stats(self, response: Dict[str, Any]):
        """Update session statistics"""
        
        self.session_stats['queries_processed'] += 1
        self.session_stats['total_processing_time'] += response.get('processing_time_seconds', 0)
        
        # Update average evaluation score
        eval_score = response.get('quality_metrics', {}).get('evaluation_score')
        if eval_score is not None:
            current_avg = self.session_stats['average_evaluation_score']
            count = self.session_stats['queries_processed']
            
            # Calculate running average
            new_avg = ((current_avg * (count - 1)) + eval_score) / count
            self.session_stats['average_evaluation_score'] = new_avg
    
    def _log_to_phoenix(self, query: str, response: Dict[str, Any]):
        """Log interaction to Phoenix for monitoring"""
        
        if not self.enable_phoenix:
            return
            
        try:
            # Create Phoenix log entry
            phoenix_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'answer': response.get('answer', ''),
                'processing_time': response.get('processing_time_seconds', 0),
                'evaluation_score': response.get('quality_metrics', {}).get('evaluation_score'),
                'sources_count': len(response.get('sources', [])),
                'pipeline_stages': response.get('pipeline_stages', {}),
                'session_id': id(self)  # Simple session identifier
            }
            
            # Add span attributes if available
            try:
                current_span = trace_api.get_current_span()
                current_span.set_attribute("answer_length", len(response.get('answer', '')))
                current_span.set_attribute("sources_used", len(response.get('sources', [])))
                if response.get('quality_metrics', {}).get('evaluation_score'):
                    current_span.set_attribute("evaluation_score", response['quality_metrics']['evaluation_score'])
            except:
                pass  # Ignore tracing errors
            
            self.logger.debug(f"📊 Logged interaction to Phoenix: {query[:50]}...")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Phoenix logging failed: {e}")
    
    def interactive_session(self):
        """Run interactive Q&A session with Phoenix monitoring"""
        
        print("🧪 CrewAI RAG System - Interactive Session")
        print("=" * 60)
        print("📚 System ready with 4-agent pipeline:")
        print("   🔍 Retriever Agent - Document search and retrieval")
        print("   🔄 Reranker Agent - all-minilm similarity optimization") 
        print("   🤖 Generator Agent - Llama 3.2 response generation")
        print("   📊 Evaluator Agent - RAGAs quality assessment")
        
        if self.enable_phoenix:
            print(f"📊 Phoenix monitoring: http://localhost:{self.phoenix_port}")
        
        print("\nType your questions below (or 'quit' to exit):")
        print("Commands: 'status' for system status, 'stats' for session statistics")
        
        while True:
            try:
                user_input = input("\n🔍 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self._show_session_summary()
                    break
                
                if user_input.lower() == 'status':
                    self._show_system_status()
                    continue
                
                if user_input.lower() == 'stats':
                    self._show_session_stats()
                    continue
                
                if not user_input:
                    print("⚠️ Please enter a question.")
                    continue
                
                print(f"\n--- Processing Query #{self.session_stats['queries_processed'] + 1} ---")
                
                # Process the question with tracing
                response = self.process_query(
                    query=user_input,
                    response_type="adaptive",
                    enable_citations=True
                )
                
                # Display response
                self._display_response(response)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Session interrupted.")
                self._show_session_summary()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    def _display_response(self, response: Dict[str, Any]):
        """Display formatted response"""
        
        print(f"💡 Answer: {response['answer']}")
        
        # Show sources if available
        sources = response.get('sources', [])
        if sources:
            print(f"\n📚 Sources ({len(sources)}):")
            for i, source in enumerate(sources[:3], 1):
                filename = source.get('filename', 'unknown')
                relevance = source.get('relevance_score', 0)
                print(f"   {i}. {filename} (relevance: {relevance:.3f})")
        
        # Show quality metrics
        quality = response.get('quality_metrics', {})
        eval_score = quality.get('evaluation_score')
        if eval_score is not None:
            print(f"📊 Quality Score: {eval_score:.3f}")
        
        # Show processing time
        proc_time = response.get('processing_time_seconds', 0)
        print(f"⏱️  Processing Time: {proc_time:.2f}s")
        
        # Show any recommendations
        recommendations = quality.get('recommendations', [])
        if recommendations and len(recommendations) > 0:
            print(f"💡 Suggestion: {recommendations[0]}")
    
    def _show_system_status(self):
        """Show system status"""
        
        status = self.master_agent.get_system_status()
        print(f"\n🔧 System Status: {status['overall_status'].upper()}")
        
        agent_status = status.get('agent_status', {})
        for agent_name, agent_info in agent_status.items():
            status_icon = "✅" if agent_info.get('status') == 'success' else "❌"
            print(f"   {status_icon} {agent_name.title()} Agent: {agent_info.get('status', 'unknown')}")
        
        if self.enable_phoenix:
            print(f"📊 Phoenix Monitoring: Active on port {self.phoenix_port}")
        else:
            print("📊 Phoenix Monitoring: Disabled")
    
    def _show_session_stats(self):
        """Show session statistics"""
        
        stats = self.session_stats
        session_duration = (datetime.now() - stats['session_start']).total_seconds()
        
        print(f"\n📈 Session Statistics:")
        print(f"   • Queries processed: {stats['queries_processed']}")
        print(f"   • Session duration: {session_duration:.1f}s")
        print(f"   • Total processing time: {stats['total_processing_time']:.2f}s")
        print(f"   • Average evaluation score: {stats['average_evaluation_score']:.3f}")
        
        if stats['queries_processed'] > 0:
            avg_response_time = stats['total_processing_time'] / stats['queries_processed']
            print(f"   • Average response time: {avg_response_time:.2f}s")
    
    def _show_session_summary(self):
        """Show session summary"""
        
        print(f"\n👋 Session Summary:")
        self._show_session_stats()
        
        if self.enable_phoenix:
            print(f"\n📊 View detailed analytics at: http://localhost:{self.phoenix_port}")
        
        print("Thank you for using CrewAI RAG System!")
    
    def batch_evaluate_performance(self, test_queries: List[str]) -> Dict[str, Any]:
        """Run batch evaluation for performance testing"""
        
        print(f"🧪 Running batch evaluation with {len(test_queries)} queries...")
        
        results = []
        for i, query in enumerate(test_queries):
            print(f"Processing {i+1}/{len(test_queries)}: {query[:50]}...")
            
            response = self.process_query(query, response_type="comprehensive")
            results.append(response)
        
        # Calculate batch statistics
        evaluation_scores = [
            r.get('quality_metrics', {}).get('evaluation_score', 0) 
            for r in results if r.get('quality_metrics', {}).get('evaluation_score') is not None
        ]
        
        processing_times = [r.get('processing_time_seconds', 0) for r in results]
        
        batch_stats = {
            'total_queries': len(test_queries),
            'successful_responses': len([r for r in results if not r.get('error')]),
            'average_evaluation_score': sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0,
            'average_processing_time': sum(processing_times) / len(processing_times),
            'total_processing_time': sum(processing_times),
            'score_distribution': {
                'excellent': len([s for s in evaluation_scores if s >= 0.8]),
                'good': len([s for s in evaluation_scores if 0.6 <= s < 0.8]),
                'fair': len([s for s in evaluation_scores if 0.4 <= s < 0.6]),
                'poor': len([s for s in evaluation_scores if s < 0.4])
            }
        }
        
        return {
            'individual_results': results,
            'batch_statistics': batch_stats,
            'phoenix_enabled': self.enable_phoenix
        }
    
    def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        
        self.logger.info("🔄 Shutting down CrewAI RAG Orchestrator...")
        
        if self.enable_phoenix and self.phoenix_session:
            try:
                # Phoenix sessions typically auto-cleanup
                self.logger.info("📊 Phoenix session cleanup complete")
            except Exception as e:
                self.logger.warning(f"⚠️ Phoenix cleanup warning: {e}")
        
        self.logger.info("✅ Shutdown complete")

def create_crew_orchestrator(enable_phoenix: bool = True, phoenix_port: int = 6006) -> CrewRAGOrchestrator:
    """Factory function to create CrewAI orchestrator"""
    return CrewRAGOrchestrator(enable_phoenix=enable_phoenix, phoenix_port=phoenix_port)

if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = create_crew_orchestrator(enable_phoenix=True)
    
    try:
        # Run interactive session
        orchestrator.interactive_session()
    finally:
        orchestrator.shutdown()