"""
Pure RAGAs Evaluator - Fixed Interface & No Fallback
Location: src/contextual_rag/evaluation/ragas_evaluator.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    )
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

class OllamaEmbeddingsWrapper:
    """Wrapper for Ollama embeddings"""
    
    def __init__(self, ollama_client, model_name: str = "nomic-embed-text"):
        self.client = ollama_client
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    def embed_query(self, text: str):
        try:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            return response.get('embedding', [0.0] * 768)
        except Exception as e:
            self.logger.error(f"❌ Embedding failed: {e}")
            return [0.0] * 768
    
    def embed_documents(self, texts: List[str]):
        return [self.embed_query(text) for text in texts]
    
    async def aembed_documents(self, texts: List[str]):
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str):
        return self.embed_query(text)

class PureRAGAsEvaluator:
    """Pure RAGAs evaluation - No fallback mechanisms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ground_truth_file = Path(__file__).parent / "testing" / "ground_truth_dataset.json"
        self.ground_truth_data = None
        
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAs not available. Install: pip install ragas langchain-groq")
        
        self.setup_models()
        self.load_ground_truth()
        
        # Pure RAGAs metrics
        self.metrics = [
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ]
    
    def setup_models(self):
        """Setup pure Groq LLM and Ollama embeddings for RAGAs"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        # FIXED: Use standard ChatGroq without custom wrappers
        self.groq_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.1,
            max_retries=1,
            request_timeout=15.0
        )
        
        import ollama
        ollama_client = ollama.Client(host="http://localhost:11434")
        self.ollama_embeddings = OllamaEmbeddingsWrapper(ollama_client, "nomic-embed-text")
        
        # Standard RAGAs wrappers
        self.ragas_llm = LangchainLLMWrapper(self.groq_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.ollama_embeddings)
    
    def load_ground_truth(self):
        """Load ground truth dataset"""
        try:
            if self.ground_truth_file.exists():
                with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                    self.ground_truth_data = json.load(f)
                self.logger.info(f"📂 Ground truth loaded: {len(self.ground_truth_data.get('data', []))} questions")
            else:
                self.ground_truth_data = None
                self.logger.warning("⚠️ No ground truth found")
        except Exception as e:
            self.logger.error(f"❌ Error loading ground truth: {e}")
            self.ground_truth_data = None
    
    def find_ground_truth_for_query(self, query: str) -> Optional[str]:
        """Find ground truth answer for query"""
        if not self.ground_truth_data:
            return None
        
        for item in self.ground_truth_data.get('data', []):
            gt_question = item.get('question', '').lower().strip()
            user_question = query.lower().strip()
            
            if gt_question == user_question:
                return item.get('answer', '')
            
            # Simple similarity check
            if len(gt_question) > 0 and len(user_question) > 0:
                gt_words = set(gt_question.split())
                user_words = set(user_question.split())
                overlap = len(gt_words.intersection(user_words))
                similarity = overlap / max(len(gt_words), len(user_words))
                if similarity >= 0.85:
                    return item.get('answer', '')
        
        return None
    
    def evaluate_query(self, query: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Pure RAGAs evaluation - single query
        
        Args:
            query: User question
            answer: Generated response
            contexts: Retrieved context documents
        
        Returns:
            Pure RAGAs evaluation results
        """
        try:
            self.logger.info(f"📊 Pure RAGAs evaluation: '{query[:50]}...'")
            
            # Auto-load ground truth
            ground_truth = self.find_ground_truth_for_query(query)
            
            # Prepare single data point for evaluation
            eval_data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth] if ground_truth else [answer]
            }
            
            dataset = Dataset.from_dict(eval_data)
            
            # FIXED: Configure metrics properly for RAGAs
            metrics_to_use = []
            for metric in self.metrics:
                # Create fresh instances to avoid conflicts
                if metric.name == 'answer_relevancy':
                    metric_instance = answer_relevancy
                elif metric.name == 'context_precision':
                    metric_instance = context_precision
                elif metric.name == 'context_recall':
                    metric_instance = context_recall
                elif metric.name == 'answer_correctness':
                    metric_instance = answer_correctness
                else:
                    continue
                
                # Configure with RAGAs wrappers
                if hasattr(metric_instance, 'llm'):
                    metric_instance.llm = self.ragas_llm
                if hasattr(metric_instance, 'embeddings'):
                    metric_instance.embeddings = self.ragas_embeddings
                
                # Skip answer_correctness if no ground truth
                if ground_truth is None and metric_instance.name == 'answer_correctness':
                    self.logger.info(f"⏭️ Skipping {metric_instance.name} - no ground truth")
                    continue
                
                metrics_to_use.append(metric_instance)
            
            if not metrics_to_use:
                raise ValueError("No metrics available for evaluation")
            
            self.logger.info(f"🔧 Using pure RAGAs metrics: {[m.name for m in metrics_to_use]}")
            
            # PURE RAGAs evaluation - no modifications
            start_time = time.time()
            
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                raise_exceptions=True  # Let RAGAs handle errors properly
            )
            
            eval_time = time.time() - start_time
            self.logger.info(f"⚡ Pure RAGAs evaluation completed in {eval_time:.1f}s")
            
            # FIXED: Extract actual scores from RAGAs results
            results_df = results.to_pandas()
            detailed_scores = {}
            
            for metric in metrics_to_use:
                if metric.name in results_df.columns:
                    score_value = results_df[metric.name].iloc[0]
                    if score_value is not None and str(score_value) != 'nan':
                        detailed_scores[metric.name] = float(score_value)
                        self.logger.info(f"✅ {metric.name}: {float(score_value):.3f}")
                    else:
                        self.logger.warning(f"⚠️ Invalid score for {metric.name}: {score_value}")
                        # Don't add invalid scores
                else:
                    self.logger.warning(f"⚠️ Missing column for {metric.name}")
            
            if not detailed_scores:
                raise ValueError("No valid scores extracted from RAGAs evaluation")
            
            overall_score = sum(detailed_scores.values()) / len(detailed_scores)
            
            # Log pure RAGAs results
            self.logger.info(f"📊 Pure RAGAs Results:")
            self.logger.info(f"   • Overall score: {overall_score:.3f}")
            self.logger.info(f"   • Ground truth used: {ground_truth is not None}")
            self.logger.info(f"   • Evaluation method: pure_ragas")
            
            for metric_name, score in detailed_scores.items():
                self.logger.info(f"   • {metric_name}: {score:.3f}")
            
            return {
                'query': query,
                'answer': answer,
                'overall_score': overall_score,
                'detailed_scores': detailed_scores,
                'evaluation_method': 'pure_ragas',
                'has_ground_truth': ground_truth is not None,
                'metrics_used': list(detailed_scores.keys()),
                'evaluation_timestamp': datetime.now().isoformat(),
                'processing_time': eval_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pure RAGAs evaluation failed: {e}")
            # Re-raise the exception - no fallback
            raise Exception(f"Pure RAGAs evaluation failed: {e}")

def create_ragas_evaluator() -> PureRAGAsEvaluator:
    """Create pure RAGAs evaluator"""
    return PureRAGAsEvaluator()

# Backward compatibility alias
SimpleRAGAsEvaluator = PureRAGAsEvaluator
OllamaRAGAsEvaluator = PureRAGAsEvaluator