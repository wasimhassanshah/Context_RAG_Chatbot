"""
Simplified Evaluator Agent - Single User Query Only
Location: src/contextual_rag/agents/evaluator_agent.py
"""

from crewai import Agent
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from contextual_rag.evaluation.ragas_evaluator import create_ragas_evaluator
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False

class EvaluatorAgent:
    """Simple evaluator agent for single user query evaluation - Pure RAGAs only"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAs required. Install: pip install ragas langchain-groq")
        
        # Initialize RAGAs evaluator
        try:
            self.ragas_evaluator = create_ragas_evaluator()
            self.logger.info("✅ RAGAs evaluator initialized")
        except Exception as e:
            self.logger.error(f"❌ RAGAs initialization failed: {e}")
            raise Exception(f"RAGAs initialization failed: {e}")
        
        # Create CrewAI agent
        self.agent = Agent(
            role="Response Quality Evaluator",
            goal="Evaluate user query responses using RAGAs metrics",
            backstory="Expert in evaluating AI responses using RAGAs metrics for quality assessment.",
            tools=[],
            verbose=True,
            allow_delegation=False,
            max_iter=1
        )
    
    def evaluate_response(self, 
                         query: str, 
                         generated_response: str, 
                         context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate single user query response - CORE FUNCTION
        
        Args:
            query: User question
            generated_response: Generated answer
            context_documents: Context used for generation
        
        Returns:
            Evaluation results with RAGAs scores
        """
        try:
            self.logger.info(f"🔍 Evaluating: '{query[:50]}...'")
            
            # Extract contexts
            contexts = []
            for doc in context_documents:
                content = doc.get('content', '')
                if content:
                    contexts.append(content)
            
            if not contexts:
                contexts = ["No context available"]
            
            # Run RAGAs evaluation
            evaluation_result = self.ragas_evaluator.evaluate_query(
                query=query,
                answer=generated_response,
                contexts=contexts
            )
            
            # Generate simple recommendations
            recommendations = self._generate_recommendations(evaluation_result)
            
            # Create final result
            result = {
                'query': query,
                'answer': generated_response,
                'overall_score': evaluation_result.get('overall_score', 0),
                'detailed_scores': evaluation_result.get('detailed_scores', {}),
                'evaluation_method': 'ragas',
                'has_ground_truth': evaluation_result.get('has_ground_truth', False),
                'recommendations': recommendations,
                'context_count': len(contexts),
                'source_files': list(set([doc.get('filename', 'unknown') for doc in context_documents])),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            self._log_results(result)
            return result
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
    
    def _generate_recommendations(self, evaluation_result: Dict[str, Any]) -> List[str]:
        """Generate simple recommendations"""
        recommendations = []
        detailed_scores = evaluation_result.get('detailed_scores', {})
        overall_score = evaluation_result.get('overall_score', 0)
        
        # Overall assessment
        if overall_score >= 0.8:
            recommendations.append("EXCELLENT: Strong performance across RAGAs metrics")
        elif overall_score >= 0.6:
            recommendations.append("GOOD: Solid performance with room for improvement")
        else:
            recommendations.append("NEEDS IMPROVEMENT: Significant optimization required")
        
        # Specific metric recommendations
        for metric, score in detailed_scores.items():
            if score < 0.6:
                if metric == 'answer_relevancy':
                    recommendations.append("Improve answer relevancy - better align responses with questions")
                elif metric == 'context_precision':
                    recommendations.append("Improve context precision - retrieve more relevant documents")
                elif metric == 'context_recall':
                    recommendations.append("Improve context recall - ensure complete information retrieval")
                elif metric in ['answer_similarity', 'answer_correctness']:
                    recommendations.append("Improve answer accuracy - verify factual correctness")
        
        # Ground truth note
        if evaluation_result.get('has_ground_truth'):
            recommendations.append("✅ Evaluated against ground truth benchmark")
        
        return recommendations
    
    def _log_results(self, evaluation: Dict[str, Any]):
        """Log evaluation results"""
        self.logger.info(f"📊 Evaluation Results:")
        self.logger.info(f"   • Overall score: {evaluation.get('overall_score', 0):.3f}")
        self.logger.info(f"   • Ground truth used: {evaluation.get('has_ground_truth', False)}")
        self.logger.info(f"   • Evaluation method: {evaluation.get('evaluation_method', 'unknown')}")
        
        # Log metric scores
        for metric, score in evaluation.get('detailed_scores', {}).items():
            self.logger.info(f"   • {metric}: {score:.3f}")
        
        # Log top recommendation
        recommendations = evaluation.get('recommendations', [])
        if recommendations:
            self.logger.info(f"   • Recommendation: {recommendations[0]}")

def create_evaluator_agent() -> EvaluatorAgent:
    """Create simple evaluator agent"""
    return EvaluatorAgent()

if __name__ == "__main__":
    # Simple test
    try:
        agent = create_evaluator_agent()
        print("✅ Evaluator agent ready for single query evaluation")
    except Exception as e:
        print(f"❌ Setup failed: {e}")