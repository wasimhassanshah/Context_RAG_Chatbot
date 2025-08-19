"""
Vector Search Tool for CrewAI Agents
Location: src/contextual_rag/agents/tools/vector_search_tool.py
"""

import psycopg2
import json
from typing import List, Dict, Any, Optional
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

class VectorSearchTool:
    """Search for relevant documents using vector similarity in PostgreSQL database"""
    
    def __init__(self):
        self.name = "Vector Search Tool"
        self.description = "Search for relevant documents using vector similarity in PostgreSQL database"
        self.password = os.getenv('POSTGRES_PASSWORD', 'RagUser2024')
        self.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text", 
            base_url="http://localhost:11434"
        )
    
    def run(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector embeddings
        
        Args:
            query: Search query text
            top_k: Number of top results to return (default 10 for reranking)
        
        Returns:
            List of dictionaries containing document content and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # Connect to database
            conn = psycopg2.connect(
                host='localhost', 
                database='contextual_rag', 
                user='rag_user', 
                password=self.password
            )
            cursor = conn.cursor()
            
            # Vector similarity search
            cursor.execute("""
                SELECT 
                    content, 
                    metadata, 
                    filename, 
                    chunk_id,
                    1 - (embedding <=> %s::vector) as similarity_score
                FROM custom_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Format results
            formatted_results = []
            for i, (content, metadata_json, filename, chunk_id, similarity) in enumerate(results):
                
                # Parse metadata if it's JSON string
                try:
                    metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                except:
                    metadata = {}
                
                result = {
                    'rank': i + 1,
                    'content': content,
                    'filename': filename,
                    'chunk_id': chunk_id,
                    'similarity_score': float(similarity),
                    'metadata': metadata,
                    'content_preview': content[:200] + "..." if len(content) > 200 else content
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            return [{"error": f"Vector search failed: {str(e)}"}]
    
    def search_by_filename(self, query: str, filename: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within specific document file"""
        try:
            query_embedding = self.embed_model.get_text_embedding(query)
            
            conn = psycopg2.connect(
                host='localhost', 
                database='contextual_rag', 
                user='rag_user', 
                password=self.password
            )
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    content, 
                    metadata, 
                    filename, 
                    chunk_id,
                    1 - (embedding <=> %s::vector) as similarity_score
                FROM custom_embeddings
                WHERE filename ILIKE %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, f"%{filename}%", query_embedding, top_k))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            formatted_results = []
            for i, (content, metadata_json, filename, chunk_id, similarity) in enumerate(results):
                try:
                    metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                except:
                    metadata = {}
                
                result = {
                    'rank': i + 1,
                    'content': content,
                    'filename': filename,
                    'chunk_id': chunk_id,
                    'similarity_score': float(similarity),
                    'metadata': metadata
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            return [{"error": f"Filename search failed: {str(e)}"}]
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get database statistics for debugging"""
        try:
            conn = psycopg2.connect(
                host='localhost', 
                database='contextual_rag', 
                user='rag_user', 
                password=self.password
            )
            cursor = conn.cursor()
            
            # Total documents
            cursor.execute("SELECT COUNT(*) FROM custom_embeddings")
            total_docs = cursor.fetchone()[0]
            
            # Documents by filename
            cursor.execute("""
                SELECT filename, COUNT(*) 
                FROM custom_embeddings 
                GROUP BY filename 
                ORDER BY COUNT(*) DESC
            """)
            doc_counts = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'total_chunks': total_docs,
                'documents': dict(doc_counts)
            }
            
        except Exception as e:
            return {"error": f"Stats query failed: {str(e)}"}