"""
Reranking Tool for CrewAI Agents using BGE Reranker
Location: src/contextual_rag/agents/tools/reranking_tool.py
"""

import ollama
from typing import List, Dict, Any, Tuple
import logging

class RerankingTool:
    """Rerank retrieved documents using BGE reranker model for better relevance"""
    
    def __init__(self):
        self.name = "Document Reranking Tool"
        self.description = "Rerank retrieved documents using BGE reranker model for better relevance"
        self.client = ollama.Client(host="http://localhost:11434")
        self.reranker_model = "llama3.2:1b"
        self.logger = logging.getLogger(__name__)
    
    def run(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank documents using BGE reranker model
        
        Args:
            query: The search query
            documents: List of documents from vector search
            top_k: Number of top documents to return after reranking
        
        Returns:
            List of reranked documents with relevance scores
        """
        try:
            if not documents:
                return []
            
            # Prepare documents for reranking
            doc_texts = []
            for doc in documents:
                # Use content preview or full content based on length
                content = doc.get('content', '')
                if len(content) > 512:  # BGE reranker works better with shorter texts
                    content = content[:512] + "..."
                doc_texts.append(content)
            
            # Rerank using BGE model
            reranked_results = []
            
            for i, (doc, doc_text) in enumerate(zip(documents, doc_texts)):
                try:
                    # Create prompt for reranking
                    rerank_prompt = f"""Query: {query}
Document: {doc_text}

Rate the relevance of this document to the query on a scale of 0.0 to 1.0.
Only return a number between 0.0 and 1.0."""
                    
                    # Get relevance score from reranker
                    response = self.client.generate(
                        model=self.reranker_model,
                        prompt=rerank_prompt,
                        options={
                            'temperature': 0.1,
                            'num_predict': 10  # Short response expected
                        }
                    )
                    
                    # Extract relevance score
                    relevance_text = response['response'].strip()
                    try:
                        relevance_score = float(relevance_text)
                        # Ensure score is between 0 and 1
                        relevance_score = max(0.0, min(1.0, relevance_score))
                    except ValueError:
                        # Fallback: use original similarity score
                        relevance_score = doc.get('similarity_score', 0.5)
                        self.logger.warning(f"Could not parse relevance score: {relevance_text}")
                    
                    # Create reranked document
                    reranked_doc = doc.copy()
                    reranked_doc.update({
                        'rerank_score': relevance_score,
                        'original_rank': doc.get('rank', i + 1),
                        'reranking_method': 'bge-reranker-large'
                    })
                    
                    reranked_results.append(reranked_doc)
                    
                except Exception as e:
                    self.logger.error(f"Reranking failed for document {i}: {e}")
                    # Keep original document with lower score
                    fallback_doc = doc.copy()
                    fallback_doc.update({
                        'rerank_score': doc.get('similarity_score', 0.3) * 0.5,  # Penalty for failed reranking
                        'original_rank': doc.get('rank', i + 1),
                        'reranking_method': 'fallback'
                    })
                    reranked_results.append(fallback_doc)
            
            # Sort by rerank score (descending)
            reranked_results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            # Update ranks and return top_k
            final_results = []
            for i, doc in enumerate(reranked_results[:top_k]):
                doc['final_rank'] = i + 1
                final_results.append(doc)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Reranking tool failed: {e}")
            # Return original documents as fallback
            return documents[:top_k]
    
    def _advanced_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Advanced reranking with batch processing for better performance
        """
        try:
            if not documents:
                return []
            
            # Batch reranking for efficiency
            batch_size = 5
            all_scores = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_scores = self._rerank_batch(query, batch)
                all_scores.extend(batch_scores)
            
            # Combine documents with scores
            reranked_docs = []
            for doc, score in zip(documents, all_scores):
                reranked_doc = doc.copy()
                reranked_doc.update({
                    'rerank_score': score,
                    'original_rank': doc.get('rank', 0),
                    'reranking_method': 'bge-batch-reranker'
                })
                reranked_docs.append(reranked_doc)
            
            # Sort and return top_k
            reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            for i, doc in enumerate(reranked_docs[:top_k]):
                doc['final_rank'] = i + 1
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            self.logger.error(f"Advanced reranking failed: {e}")
            return documents[:top_k]
    
    def _rerank_batch(self, query: str, batch: List[Dict[str, Any]]) -> List[float]:
        """Rerank a batch of documents"""
        scores = []
        
        for doc in batch:
            try:
                content = doc.get('content', '')[:512]  # Truncate for efficiency
                
                rerank_prompt = f"""Given the query and document below, rate their relevance from 0.0 to 1.0.

Query: {query}
Document: {content}

Relevance score (0.0-1.0):"""
                
                response = self.client.generate(
                    model=self.reranker_model,
                    prompt=rerank_prompt,
                    options={'temperature': 0.1, 'num_predict': 5}
                )
                
                try:
                    score = float(response['response'].strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    score = doc.get('similarity_score', 0.5)
                
                scores.append(score)
                
            except Exception:
                scores.append(doc.get('similarity_score', 0.3))
        
        return scores
    
    def test_reranker_model(self) -> Dict[str, Any]:
        """Test if the reranker model is working"""
        try:
            test_prompt = "Query: What is procurement?\nDocument: Procurement involves buying goods and services.\n\nRelevance (0.0-1.0):"
            
            response = self.client.generate(
                model=self.reranker_model,
                prompt=test_prompt,
                options={'temperature': 0.1, 'num_predict': 5}
            )
            
            return {
                'status': 'success',
                'model': self.reranker_model,
                'test_response': response['response'].strip()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'model': self.reranker_model,
                'error': str(e)
            }