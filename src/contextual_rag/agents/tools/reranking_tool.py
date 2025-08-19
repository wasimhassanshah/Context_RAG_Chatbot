"""
Similarity-Based Reranking Tool using all-minilm
Location: src/contextual_rag/agents/tools/reranking_tool.py
"""

import ollama
import numpy as np
from typing import List, Dict, Any
import logging

class RerankingTool:
    """Rerank retrieved documents using all-minilm similarity-based approach"""
    
    def __init__(self):
        self.name = "Document Similarity Reranking Tool"
        self.description = "Rerank retrieved documents using all-minilm embeddings for better relevance"
        self.client = ollama.Client(host="http://localhost:11434")
        self.reranker_model = "all-minilm"
        self.embedding_dim = 384  # all-minilm embedding dimension
        self.logger = logging.getLogger(__name__)
    
    def run(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Similarity-based reranking using all-minilm embeddings
        No text generation - pure vector similarity scoring
        
        Args:
            query: The search query
            documents: List of documents from vector search
            top_k: Number of top documents to return after reranking
        
        Returns:
            List of reranked documents with similarity scores
        """
        try:
            if not documents:
                self.logger.warning("⚠️ No documents provided for reranking")
                return []
            
            self.logger.info(f"🔄 Similarity reranking {len(documents)} documents using {self.reranker_model}")
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                self.logger.error("❌ Failed to get query embedding, using original order")
                return documents[:top_k]
            
            # Get embeddings for all documents and calculate similarities
            scored_docs = []
            successful_embeddings = 0
            
            for i, doc in enumerate(documents):
                try:
                    # Use document content, truncate for efficiency
                    content = doc.get('content', '')
                    if len(content) > 512:  # Truncate long documents
                        content = content[:512] + "..."
                    
                    # Get document embedding
                    doc_embedding = self._get_embedding(content)
                    
                    if doc_embedding is not None:
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(query_embedding, doc_embedding)
                        successful_embeddings += 1
                    else:
                        # Fallback to original similarity score if embedding fails
                        similarity = doc.get('similarity_score', 0.5)
                    
                    # Create reranked document
                    reranked_doc = doc.copy()
                    reranked_doc.update({
                        'rerank_score': float(similarity),
                        'original_rank': doc.get('rank', i + 1),
                        'reranking_method': 'all-minilm-similarity',
                        'content_length': len(doc.get('content', ''))
                    })
                    
                    scored_docs.append(reranked_doc)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to rerank document {i}: {e}")
                    # Keep document with original score
                    fallback_doc = doc.copy()
                    fallback_doc.update({
                        'rerank_score': doc.get('similarity_score', 0.3),
                        'original_rank': doc.get('rank', i + 1),
                        'reranking_method': 'fallback'
                    })
                    scored_docs.append(fallback_doc)
            
            # Sort by similarity score (descending)
            scored_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            # Return top k with updated ranks
            final_results = []
            for i, doc in enumerate(scored_docs[:top_k]):
                doc['final_rank'] = i + 1
                final_results.append(doc)
            
            # Log results
            if final_results:
                top_score = final_results[0]['rerank_score']
                self.logger.info(f"✅ Similarity reranking complete!")
                self.logger.info(f"   • Successful embeddings: {successful_embeddings}/{len(documents)}")
                self.logger.info(f"   • Top similarity score: {top_score:.3f}")
                self.logger.info(f"   • Returned {len(final_results)} documents")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Similarity reranking failed: {e}")
            # Fallback to original documents
            return documents[:top_k]
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from all-minilm via Ollama embed API"""
        try:
            if not text or not text.strip():
                return None
            
            response = self.client.embeddings(
                model=self.reranker_model,
                prompt=text.strip()
            )
            
            embedding = response.get('embedding')
            if embedding and len(embedding) > 0:
                return embedding
            else:
                self.logger.warning(f"⚠️ Empty embedding received for text: {text[:50]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1, dtype=np.float32)
            vec2 = np.array(vec2, dtype=np.float32)
            
            # Handle edge cases
            if len(vec1) == 0 or len(vec2) == 0:
                return 0.0
            
            if len(vec1) != len(vec2):
                self.logger.warning(f"⚠️ Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure similarity is in valid range [0, 1]
            # Cosine similarity can be [-1, 1], but we want [0, 1] for consistency
            normalized_similarity = (similarity + 1) / 2
            
            return float(np.clip(normalized_similarity, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"❌ Cosine similarity calculation failed: {e}")
            return 0.0
    
    def analyze_similarity_distribution(self, 
                                      query: str, 
                                      documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze similarity distribution of documents (debugging tool)
        
        Args:
            query: Search query
            documents: List of documents to analyze
        
        Returns:
            Analysis of similarity distribution and quality
        """
        try:
            if not documents:
                return {'analysis': 'no_documents'}
            
            # Get similarities for all documents
            scored_docs = self.run(query, documents, len(documents))
            
            if not scored_docs:
                return {'analysis': 'reranking_failed'}
            
            # Extract scores
            similarities = [doc.get('rerank_score', 0) for doc in scored_docs]
            original_similarities = [doc.get('similarity_score', 0) for doc in scored_docs]
            
            # Calculate statistics
            analysis = {
                'total_documents': len(documents),
                'rerank_similarities': {
                    'mean': np.mean(similarities),
                    'max': np.max(similarities),
                    'min': np.min(similarities),
                    'std': np.std(similarities),
                    'range': np.max(similarities) - np.min(similarities)
                },
                'original_similarities': {
                    'mean': np.mean(original_similarities),
                    'max': np.max(original_similarities),
                    'min': np.min(original_similarities)
                },
                'quality_assessment': self._assess_similarity_quality(similarities),
                'score_distribution': self._get_similarity_distribution(similarities),
                'top_documents': scored_docs[:3]  # Top 3 for inspection
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Similarity analysis failed: {e}")
            return {'analysis': 'error', 'error': str(e)}
    
    def _assess_similarity_quality(self, scores: List[float]) -> str:
        """Assess overall quality of similarity scores"""
        if not scores:
            return 'no_data'
        
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        score_range = np.max(scores) - np.min(scores)
        
        if mean_score >= 0.8 and min_score >= 0.6:
            return 'excellent'
        elif mean_score >= 0.6 and min_score >= 0.4:
            return 'good'
        elif mean_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _get_similarity_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get distribution of similarity scores across quality bands"""
        distribution = {
            'high': 0,        # 0.8+
            'medium': 0,      # 0.6-0.8
            'low': 0,         # 0.4-0.6
            'very_low': 0     # <0.4
        }
        
        for score in scores:
            if score >= 0.8:
                distribution['high'] += 1
            elif score >= 0.6:
                distribution['medium'] += 1
            elif score >= 0.4:
                distribution['low'] += 1
            else:
                distribution['very_low'] += 1
        
        return distribution
    
    def test_reranker_model(self) -> Dict[str, Any]:
        """Test if the all-minilm model is working for embeddings"""
        try:
            test_text = "This is a test document about procurement policies."
            
            embedding = self._get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                return {
                    'status': 'success',
                    'model': self.reranker_model,
                    'embedding_dimension': len(embedding),
                    'sample_embedding_values': embedding[:5] if len(embedding) >= 5 else embedding
                }
            else:
                return {
                    'status': 'error',
                    'model': self.reranker_model,
                    'error': 'No embedding generated'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'model': self.reranker_model,
                'error': str(e)
            }
    
    def batch_rerank(self, 
                    query: str, 
                    document_batches: List[List[Dict[str, Any]]], 
                    top_k_per_batch: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank multiple batches of documents and combine results
        Useful for processing large document sets efficiently
        """
        all_reranked = []
        
        for i, batch in enumerate(document_batches):
            self.logger.info(f"🔄 Processing batch {i+1}/{len(document_batches)}")
            batch_results = self.run(query, batch, top_k_per_batch)
            all_reranked.extend(batch_results)
        
        # Final reranking of combined results
        if len(all_reranked) > top_k_per_batch:
            # Re-sort all results by similarity score
            all_reranked.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            final_results = all_reranked[:top_k_per_batch]
            
            # Update final ranks
            for i, doc in enumerate(final_results):
                doc['final_rank'] = i + 1
                doc['batch_processed'] = True
            
            return final_results
        
        return all_reranked