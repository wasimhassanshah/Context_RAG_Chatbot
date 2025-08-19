"""
LLM Generation Tool for CrewAI Agents
Location: src/contextual_rag/agents/tools/llm_generation_tool.py
"""

import ollama
from typing import List, Dict, Any, Optional
import logging
import json

class LLMGenerationTool:
    """Generate contextual responses using Llama 3.2 1B model with retrieved documents"""
    
    def __init__(self):
        self.name = "LLM Response Generation Tool"
        self.description = "Generate contextual responses using Llama 3.2 1B model with retrieved documents"
        self.client = ollama.Client(host="http://localhost:11434")
        self.llm_model = "llama3.2:1b"
        self.logger = logging.getLogger(__name__)
    
    def run(self, 
             query: str, 
             context_documents: List[Dict[str, Any]], 
             response_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Generate response using LLM with context documents
        
        Args:
            query: User question
            context_documents: Reranked documents with content
            response_type: Type of response (comprehensive, concise, detailed)
        
        Returns:
            Dictionary with generated response and metadata
        """
        try:
            if not context_documents:
                return self._generate_no_context_response(query)
            
            # Prepare context from documents
            context = self._format_context(context_documents)
            
            # Create prompt based on response type
            prompt = self._create_prompt(query, context, response_type)
            
            # Generate response
            response = self.client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 1000,  # Allow longer responses
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            generated_text = response['response'].strip()
            
            # Prepare response metadata
            response_data = {
                'answer': generated_text,
                'query': query,
                'response_type': response_type,
                'context_sources': len(context_documents),
                'source_files': list(set([doc.get('filename', 'unknown') for doc in context_documents])),
                'generation_method': 'llama3.2-1b',
                'context_length': len(context),
                'sources_used': []
            }
            
            # Add source information
            for i, doc in enumerate(context_documents):
                source_info = {
                    'rank': doc.get('final_rank', i + 1),
                    'filename': doc.get('filename', 'unknown'),
                    'chunk_id': doc.get('chunk_id', 'unknown'),
                    'relevance_score': doc.get('rerank_score', doc.get('similarity_score', 0)),
                    'content_preview': doc.get('content', '')[:100] + "..."
                }
                response_data['sources_used'].append(source_info)
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return {
                'answer': f"I apologize, but I encountered an error while generating a response: {str(e)}",
                'query': query,
                'error': str(e),
                'generation_method': 'error_fallback'
            }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format context documents for the prompt"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            filename = doc.get('filename', 'Unknown Document')
            content = doc.get('content', '')
            relevance = doc.get('rerank_score', doc.get('similarity_score', 0))
            
            context_part = f"""Document {i+1}: {filename} (Relevance: {relevance:.3f})
{content}
"""
            context_parts.append(context_part)
        
        return "\n" + "="*50 + "\n".join(context_parts) + "="*50 + "\n"
    
    def _create_prompt(self, query: str, context: str, response_type: str) -> str:
        """Create appropriate prompt based on response type"""
        
        base_instructions = {
            'comprehensive': "Provide a detailed, comprehensive answer that fully addresses the question.",
            'concise': "Provide a brief, direct answer that gets straight to the point.",
            'detailed': "Provide a thorough answer with explanations, examples, and relevant details."
        }
        
        instruction = base_instructions.get(response_type, base_instructions['comprehensive'])
        
        prompt = f"""You are an expert assistant helping users understand organizational documents. Based on the provided context documents, answer the user's question accurately and helpfully.

CONTEXT DOCUMENTS:
{context}

INSTRUCTION: {instruction}

QUESTION: {query}

GUIDELINES:
1. Base your answer primarily on the provided context documents
2. If the context doesn't contain sufficient information, clearly state this
3. Cite specific document names when referencing information
4. Be accurate and avoid making assumptions beyond what's in the context
5. Structure your response clearly and logically

ANSWER:"""
        
        return prompt
    
    def _generate_no_context_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no relevant context is found"""
        
        prompt = f"""A user asked the following question, but no relevant documents were found in the knowledge base:

Question: {query}

Please provide a helpful response explaining that no specific information was found in the available documents, and suggest what type of information might be needed to answer their question properly.

Response:"""
        
        try:
            response = self.client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 300
                }
            )
            
            return {
                'answer': response['response'].strip(),
                'query': query,
                'context_sources': 0,
                'generation_method': 'no-context-fallback',
                'source_files': [],
                'sources_used': []
            }
            
        except Exception as e:
            return {
                'answer': "I apologize, but I couldn't find relevant information in the available documents to answer your question.",
                'query': query,
                'error': str(e),
                'context_sources': 0,
                'generation_method': 'error-fallback'
            }
    
    def generate_with_citations(self, 
                               query: str, 
                               context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response with explicit citations"""
        
        try:
            context = self._format_context_with_citations(context_documents)
            
            prompt = f"""Based on the numbered documents below, answer the question and include citations in square brackets [1], [2], etc.

DOCUMENTS:
{context}

QUESTION: {query}

Provide your answer with citations. Use [1], [2], etc. to reference specific documents.

ANSWER:"""
            
            response = self.client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 800
                }
            )
            
            return {
                'answer': response['response'].strip(),
                'query': query,
                'has_citations': True,
                'context_sources': len(context_documents),
                'generation_method': 'citation-enabled',
                'sources_used': [
                    {
                        'citation_number': i + 1,
                        'filename': doc.get('filename', 'unknown'),
                        'chunk_id': doc.get('chunk_id', 'unknown'),
                        'relevance_score': doc.get('rerank_score', 0)
                    }
                    for i, doc in enumerate(context_documents)
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Citation generation failed: {e}")
            return self._run(query, context_documents, "comprehensive")
    
    def _format_context_with_citations(self, documents: List[Dict[str, Any]]) -> str:
        """Format context with citation numbers"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            filename = doc.get('filename', 'Unknown Document')
            content = doc.get('content', '')
            
            context_part = f"[{i+1}] {filename}\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def test_llm_model(self) -> Dict[str, Any]:
        """Test if the LLM model is working"""
        try:
            test_prompt = "Please respond with 'LLM model is working correctly' if you can understand this message."
            
            response = self.client.generate(
                model=self.llm_model,
                prompt=test_prompt,
                options={'temperature': 0.1, 'num_predict': 20}
            )
            
            return {
                'status': 'success',
                'model': self.llm_model,
                'test_response': response['response'].strip()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'model': self.llm_model,
                'error': str(e)
            }