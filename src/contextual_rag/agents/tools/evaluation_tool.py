"""
RAGAs Evaluation Tool for CrewAI Agents - PURE RAGAS (No Fallback)
Location: src/contextual_rag/agents/tools/evaluation_tool.py
"""

from typing import List, Dict, Any, Optional
import logging
import json
from dataclasses import dataclass
from datetime import datetime
import sys
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.contextual_rag.evaluation.ragas_evaluator import create_ragas_evaluator

@dataclass
class EvaluationResult:
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    scores: Dict[str, float]
    timestamp: datetime
    evaluation_method: str

class EvaluationTool:
    """PURE RAGAs evaluation tool using local Ollama models"""
    
    def __init__(self):
        self.name = "RAGAs Evaluation Tool"
        self.description = "Evaluate RAG responses using RAGAs metrics with local Ollama models"
        self.logger = logging.getLogger(__name__)
        self.evaluation_history = []
        
        # Initialize RAGAs evaluator
        try:
            self.ragas_evaluator = create_ragas_evaluator()
            self.logger.info("✅ RAGAs evaluator initialized with Ollama models")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RAGAs evaluator: {e}")
            raise Exception(f"RAGAs evaluator initialization failed: {e}")
    
    def run(self, 
            query: str, 
            generated_answer: str, 
            context_documents: List[Dict[str, Any]], 
            ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate RAG response using PURE RAGAs metrics (NO FALLBACK)
        
        Args:
            query: Original user query
            generated_answer: Generated response from LLM
            context_documents: Retrieved context documents
            ground_truth: Optional ground truth answer for comparison
        
        Returns:
            Dictionary with RAGAs evaluation scores and metrics
        """
        try:
            self.logger.info(f"📊 Running PURE RAGAs evaluation for: '{query[:50]}...'")
            
            # Extract contexts from documents
            contexts = []
            for doc in context_documents:
                content = doc.get('content', '')
                if content:
                    contexts.append(content)
            
            if not contexts:
                self.logger.warning("⚠️ No contexts available for evaluation")
                contexts = ["No context available"]
            
            # Run RAGAs evaluation
            evaluation_result = self.ragas_evaluator.evaluate_single_response(
                query=query,
                answer=generated_answer,
                contexts=contexts,
                ground_truth=ground_truth
            )
            
            # Check if evaluation was successful
            if evaluation_result.get('evaluation_method') != 'ragas_ollama':
                error_msg = f"RAGAs evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
                self.logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            # Extract scores
            detailed_scores = evaluation_result.get('detailed_scores', {})
            overall_score = evaluation_result.get('overall_score', 0)
            
            # Generate recommendations based on RAGAs scores
            recommendations = self._generate_ragas_recommendations(detailed_scores)
            
            # Calculate composite quality score
            composite_score = self._calculate_composite_score(detailed_scores)
            
            # Create final result
            result = {
                'query': query,
                'answer': generated_answer,
                'overall_score': overall_score,
                'detailed_scores': detailed_scores,
                'composite_quality_score': composite_score,
                'context_count': len(contexts),
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluation_method': 'ragas_ollama',
                'source_files': list(set([doc.get('filename', 'unknown') for doc in context_documents])),
                'recommendations': recommendations,
                'has_ground_truth': ground_truth is not None,
                'metrics_evaluated': list(detailed_scores.keys())
            }
            
            # Store in history
            eval_result = EvaluationResult(
                query=query,
                answer=generated_answer,
                contexts=contexts,
                ground_truth=ground_truth,
                scores=detailed_scores,
                timestamp=datetime.now(),
                evaluation_method='ragas_ollama'
            )
            self.evaluation_history.append(eval_result)
            
            self.logger.info(f"✅ RAGAs evaluation complete - Overall: {overall_score:.3f}")
            return result
            
        except Exception as e:
            error_msg = f"RAGAs evaluation failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            
            # NO FALLBACK - Raise the error
            raise Exception(error_msg)
    
    def _generate_ragas_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on RAGAs scores"""
        
        recommendations = []
        
        for metric, score in scores.items():
            if score < 0.6:  # Low score threshold
                if metric == 'faithfulness':
                    recommendations.append("Improve faithfulness: Ensure answers are better grounded in retrieved context")
                elif metric == 'answer_relevancy':
                    recommendations.append("Improve answer relevancy: Make responses more directly related to the question")
                elif metric == 'context_precision':
                    recommendations.append("Improve context precision: Retrieve more relevant and focused documents")
                elif metric == 'context_recall':
                    recommendations.append("Improve context recall: Ensure all relevant information is retrieved")
            elif score < 0.8:  # Medium score threshold
                if metric == 'faithfulness':
                    recommendations.append("Good faithfulness: Minor improvements in grounding answers to context")
                elif metric == 'answer_relevancy':
                    recommendations.append("Good relevancy: Fine-tune answer focus for better alignment")
        
        # Overall assessment
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        if avg_score >= 0.8:
            recommendations.insert(0, "Excellent RAGAs performance across all metrics")
        elif avg_score >= 0.6:
            recommendations.insert(0, "Good RAGAs performance with room for improvement")
        else:
            recommendations.insert(0, "RAGAs scores indicate significant areas for improvement")
        
        return recommendations if recommendations else ["RAGAs evaluation completed successfully"]
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite score from RAGAs metrics"""
        
        if not scores:
            return 0.0
        
        # Weights for different RAGAs metrics
        weights = {
            'faithfulness': 0.3,      # Very important - answer must be grounded
            'answer_relevancy': 0.3,  # Very important - answer must be relevant
            'context_precision': 0.2, # Important - context quality matters
            'context_recall': 0.2     # Important - completeness matters
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 0.1)  # Default weight for unknown metrics
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate multiple responses in batch using RAGAs"""
        
        self.logger.info(f"📊 Running batch RAGAs evaluation on {len(evaluations)} items")
        
        # Use RAGAs evaluator's batch method
        try:
            batch_results = self.ragas_evaluator.evaluate_batch(evaluations)
            return batch_results
        except Exception as e:
            self.logger.error(f"❌ Batch RAGAs evaluation failed: {e}")
            raise Exception(f"Batch evaluation failed: {e}")
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history"""
        return [
            {
                'query': eval_result.query,
                'answer': eval_result.answer[:100] + "...",
                'overall_score': sum(eval_result.scores.values()) / len(eval_result.scores) if eval_result.scores else 0,
                'timestamp': eval_result.timestamp.isoformat(),
                'evaluation_method': eval_result.evaluation_method,
                'metrics': list(eval_result.scores.keys())
            }
            for eval_result in self.evaluation_history
        ]
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics over time"""
        
        if not self.evaluation_history:
            return {'message': 'No RAGAs evaluation history available'}
        
        # Calculate overall scores
        overall_scores = []
        for eval_result in self.evaluation_history:
            if eval_result.scores:
                overall_score = sum(eval_result.scores.values()) / len(eval_result.scores)
                overall_scores.append(overall_score)
        
        # Metric-wise analysis
        metric_stats = {}
        for eval_result in self.evaluation_history:
            for metric, score in eval_result.scores.items():
                if metric not in metric_stats:
                    metric_stats[metric] = []
                metric_stats[metric].append(score)
        
        metric_averages = {
            metric: sum(scores) / len(scores)
            for metric, scores in metric_stats.items()
        }
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_overall_score': sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            'score_trend': overall_scores[-10:],  # Last 10 scores
            'metric_averages': metric_averages,
            'best_score': max(overall_scores) if overall_scores else 0,
            'worst_score': min(overall_scores) if overall_scores else 0,
            'evaluation_method': 'ragas_ollama',
            'metrics_tracked': list(metric_averages.keys())
        }
    
    def test_evaluation_system(self) -> Dict[str, Any]:
        """Test the RAGAs evaluation system"""
        
        try:
            # Test RAGAs setup
            test_result = self.ragas_evaluator.test_ragas_setup()
            
            if test_result['status'] == 'success':
                # Run a full evaluation test
                test_data = {
                    'query': 'What are the procurement approval requirements?',
                    'answer': 'Procurement approval requires proper documentation and authorization from designated authorities.',
                    'context_documents': [
                        {
                            'content': 'Procurement requires approval from designated authorities with proper documentation and workflow compliance.',
                            'filename': 'procurement_manual.pdf'
                        }
                    ]
                }
                
                evaluation_result = self.run(**test_data)
                
                return {
                    'status': 'success',
                    'ragas_setup': test_result,
                    'evaluation_test': {
                        'overall_score': evaluation_result.get('overall_score', 0),
                        'metrics_count': len(evaluation_result.get('detailed_scores', {})),
                        'evaluation_method': evaluation_result.get('evaluation_method'),
                        'recommendations_count': len(evaluation_result.get('recommendations', []))
                    },
                    'system_working': True
                }
            else:
                return {
                    'status': 'failed',
                    'ragas_setup': test_result,
                    'error': 'RAGAs setup test failed',
                    'system_working': False
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'system_working': False
            }

def create_evaluation_tool() -> EvaluationTool:
    """Factory function to create evaluation tool"""
    return EvaluationTool()

if __name__ == "__main__":
    # Test the evaluation tool
    tool = create_evaluation_tool()
    
    # Run test
    test_result = tool.test_evaluation_system()
    print(f"🧪 RAGAs Evaluation Tool Test: {test_result}")