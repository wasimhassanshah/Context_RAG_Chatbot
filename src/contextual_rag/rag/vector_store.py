"""
LlamaIndex Vector Store Implementation with PostgreSQL + PGVector
Location: src/contextual_rag/rag/vector_store.py
"""

import json
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

# Database imports
import psycopg2
from sqlalchemy import create_engine, text
import asyncpg

# Local imports
from ..agents.ollama_manager import OllamaManager

from dotenv import load_dotenv
load_dotenv()

class VectorStoreManager:
    """Manages LlamaIndex vector store with PostgreSQL + PGVector"""
    
    def __init__(self, 
                 processed_data_path: str = "data/processed",
                 embedding_dim: int = 768):
        
        self.processed_data_path = Path(processed_data_path)
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'contextual_rag'),
            'user': os.getenv('POSTGRES_USER', 'rag_user'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
        
        # Initialize components
        self.ollama_manager = None
        self.embedding_model = None
        self.llm_model = None
        self.vector_store = None
        self.index = None
        
        # Setup components
        self._setup_ollama()
        self._setup_llamaindex_settings()
        self._setup_vector_store()
    
    def _setup_ollama(self):
        """Setup Ollama manager and models"""
        self.logger.info("🔧 Setting up Ollama integration...")
        
        try:
            self.ollama_manager = OllamaManager()
            
            # Configure embedding model
            self.embedding_model = OllamaEmbedding(
                model_name="nomic-embed-text",
                base_url="http://localhost:11434",
                ollama_additional_kwargs={"mirostat": 0}
            )
            
            # Configure LLM model
            self.llm_model = Ollama(
                model="llama3.2:1b",
                base_url="http://localhost:11434",
                temperature=0.1,
                request_timeout=120.0
            )
            
            self.logger.info("✅ Ollama models configured")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Ollama: {e}")
            raise
    
    def _setup_llamaindex_settings(self):
        """Configure global LlamaIndex settings"""
        self.logger.info("⚙️ Configuring LlamaIndex settings...")
        
        try:
            # Set global settings
            Settings.embed_model = self.embedding_model
            Settings.llm = self.llm_model
            Settings.chunk_size = 1000
            Settings.chunk_overlap = 200
            
            self.logger.info("✅ LlamaIndex settings configured")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to configure LlamaIndex: {e}")
            raise
    
    def _setup_vector_store(self):
        """Setup PostgreSQL vector store"""
        self.logger.info("🗄️ Setting up PostgreSQL vector store...")
        
        try:
            # Create connection string with password
            connection_string = (
                f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )

            self.vector_store = PGVectorStore(
                connection_string=connection_string,
                table_name="llamaindex_embeddings",
                embed_dim=self.embedding_dim
            )
            
            self.logger.info("✅ Vector store configured")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup vector store: {e}")
            raise
    
    def load_processed_documents(self) -> List[Document]:
        """Load all processed documents from JSON files"""
        self.logger.info("📚 Loading processed documents...")
        
        documents = []
        json_files = list(self.processed_data_path.glob("*.json"))
        
        if not json_files:
            self.logger.warning("⚠️ No processed documents found")
            return documents
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                # Convert chunks to LlamaIndex Documents
                for chunk_data in doc_data.get('chunks', []):
                    
                    # Create metadata
                    metadata = {
                        'document_id': doc_data['id'],
                        'filename': doc_data['filename'],
                        'title': doc_data['title'],
                        'chunk_id': chunk_data['id'],
                        'chunk_index': chunk_data['chunk_index'],
                        'file_type': doc_data['file_type'],
                        'page_number': chunk_data.get('page_number'),
                        'processing_timestamp': doc_data['processing_timestamp']
                    }
                    
                    # Add chunk metadata
                    if 'metadata' in chunk_data:
                        metadata.update(chunk_data['metadata'])
                    
                    # Create LlamaIndex Document
                    document = Document(
                        text=chunk_data['content'],
                        metadata=metadata,
                        id_=chunk_data['id']
                    )
                    
                    documents.append(document)
                
                self.logger.info(f"📄 Loaded {len(doc_data.get('chunks', []))} chunks from {doc_data['filename']}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load {json_file.name}: {e}")
        
        self.logger.info(f"✅ Total documents loaded: {len(documents)}")
        return documents
    
    def create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Create vector index from documents"""
        self.logger.info("🔍 Creating vector index...")
        
        try:
            # Create index with vector store
            self.index = VectorStoreIndex.from_documents(
                documents,
                vector_store=self.vector_store,
                show_progress=True
            )
            
            self.logger.info("✅ Vector index created successfully")
            return self.index
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create index: {e}")
            raise
    
    def test_embedding_generation(self, test_text: str = "This is a test document about procurement policies.") -> bool:
        """Test embedding generation"""
        self.logger.info("🧪 Testing embedding generation...")
        
        try:
            # Generate embedding
            embedding = self.embedding_model.get_text_embedding(test_text)
            
            if embedding and len(embedding) == self.embedding_dim:
                self.logger.info(f"✅ Embedding generated: {len(embedding)} dimensions")
                return True
            else:
                self.logger.error(f"❌ Invalid embedding: {len(embedding) if embedding else 0} dimensions")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return False
    
    def test_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Test document retrieval"""
        self.logger.info(f"🔍 Testing retrieval for query: '{query[:50]}...'")
        
        if not self.index:
            self.logger.error("❌ Index not created")
            return []
        
        try:
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )
            
            # Retrieve documents
            retrieved_nodes = retriever.retrieve(query)
            
            results = []
            for i, node in enumerate(retrieved_nodes):
                result = {
                    'rank': i + 1,
                    'score': node.score,
                    'content': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'metadata': node.metadata,
                    'chunk_id': node.metadata.get('chunk_id', 'unknown')
                }
                results.append(result)
            
            self.logger.info(f"✅ Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Retrieval failed: {e}")
            return []
    
    def test_llm_response(self, query: str, context_chunks: List[str]) -> str:
        """Test LLM response generation with context"""
        self.logger.info("🤖 Testing LLM response generation...")
        
        try:
            # Prepare context
            context = "\n\n".join(context_chunks)
            
            # Create prompt
            prompt = f"""Based on the following context from the documents, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.llm_model.complete(prompt)
            
            self.logger.info("✅ LLM response generated")
            return response.text
            
        except Exception as e:
            self.logger.error(f"❌ LLM response generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Use direct psycopg2 connection with explicit password
            import psycopg2
            
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'llamaindex_embeddings'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                # Get row count
                cursor.execute("SELECT COUNT(*) FROM llamaindex_embeddings")
                row_count = cursor.fetchone()[0]
                
                result = {
                    'table_exists': True,
                    'total_embeddings': row_count,
                    'embedding_dimension': self.embedding_dim
                }
            else:
                result = {
                    'table_exists': False,
                    'total_embeddings': 0,
                    'embedding_dimension': self.embedding_dim
                }
            
            cursor.close()
            conn.close()
            return result
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to get database stats: {e}")
            return {'error': str(e)}
    
    def run_complete_setup(self) -> bool:
        """Run complete RAG setup process"""
        self.logger.info("🚀 Starting complete RAG setup...")
        
        try:
            # Step 1: Load documents
            documents = self.load_processed_documents()
            if not documents:
                self.logger.error("❌ No documents to process")
                return False
            
            # Step 2: Test embedding
            if not self.test_embedding_generation():
                return False
            
            # Step 3: Create index
            self.create_index(documents)
            
            # Step 4: Test retrieval
            test_results = self.test_retrieval("What are the procurement approval requirements?")
            if not test_results:
                self.logger.warning("⚠️ Retrieval test returned no results")
            
            self.logger.info("✅ Complete RAG setup successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Complete setup failed: {e}")
            return False

def setup_vector_store_manager() -> VectorStoreManager:
    """Initialize and setup vector store manager"""
    print("🚀 Setting up Vector Store Manager...")
    
    try:
        manager = VectorStoreManager()
        print("✅ Vector store manager initialized")
        return manager
    except Exception as e:
        print(f"❌ Failed to initialize vector store manager: {e}")
        return None

if __name__ == "__main__":
    # Test the vector store manager
    manager = setup_vector_store_manager()
    
    if manager:
        success = manager.run_complete_setup()
        if success:
            print("🎉 RAG system setup complete!")
        else:
            print("❌ RAG system setup failed")
    else:
        print("❌ Failed to initialize manager")