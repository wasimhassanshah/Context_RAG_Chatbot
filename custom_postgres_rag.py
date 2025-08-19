import psycopg2
import numpy as np
import json
import os
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class CustomPostgresRAG:
    """Custom RAG implementation that actually saves to PostgreSQL"""
    
    def __init__(self):
        self.password = os.getenv('POSTGRES_PASSWORD', 'RagUser2024')
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
        self.llm = Ollama(model="llama3.2:1b", base_url="http://localhost:11434")
        self.setup_database()
    
    def setup_database(self):
        """Create our custom table"""
        conn = psycopg2.connect(host='localhost', database='contextual_rag', user='rag_user', password=self.password)
        cursor = conn.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS custom_embeddings CASCADE")
        cursor.execute("""
        CREATE TABLE custom_embeddings (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding VECTOR(768),
            filename VARCHAR(255),
            chunk_id VARCHAR(255)
        );
        CREATE INDEX ON custom_embeddings USING ivfflat (embedding vector_cosine_ops);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Custom table created")
    
    def add_documents(self, documents, batch_size=None):
        """Add documents with embeddings - CUSTOM IMPLEMENTATION"""
        if batch_size is None:
            batch_size = len(documents)  # Process all documents
        
        conn = psycopg2.connect(host='localhost', database='contextual_rag', user='rag_user', password=self.password)
        cursor = conn.cursor()
        
        for i, doc in enumerate(documents[:batch_size]):
            print(f"Processing document {i+1}/{batch_size}: {doc.metadata.get('filename', 'unknown')}")
            
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(doc.text)
            
            # Insert into database
            cursor.execute("""
                INSERT INTO custom_embeddings (content, metadata, embedding, filename, chunk_id)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                doc.text,
                json.dumps(doc.metadata),
                embedding,
                doc.metadata.get('filename', 'unknown'),
                doc.id_
            ))
            
            # Commit each insert to ensure it's saved
            conn.commit()
            print(f"   ✅ Saved embedding for: {doc.metadata.get('filename', 'unknown')}")
        
        cursor.close()
        conn.close()
        
        # Verify saves
        self.verify_saves()
    
    def verify_saves(self):
        """Verify embeddings are actually saved"""
        conn = psycopg2.connect(host='localhost', database='contextual_rag', user='rag_user', password=self.password)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM custom_embeddings")
        count = cursor.fetchone()[0]
        print(f"🎉 VERIFIED: {count} embeddings saved in database!")
        
        if count > 0:
            cursor.execute("SELECT filename, LEFT(content, 50) FROM custom_embeddings LIMIT 3")
            samples = cursor.fetchall()
            print("📄 Sample entries:")
            for sample in samples:
                print(f"   • {sample[0]}: {sample[1]}...")
        
        cursor.close()
        conn.close()
        return count
    
    def search(self, query, top_k=3):
        """Search for similar documents"""
        query_embedding = self.embed_model.get_text_embedding(query)
        
        conn = psycopg2.connect(host='localhost', database='contextual_rag', user='rag_user', password=self.password)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content, metadata, filename,
                   1 - (embedding <=> %s::vector) as similarity
            FROM custom_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return results
    
    def answer_question(self, question):
        """Full RAG pipeline"""
        print(f"🔍 Searching for: {question}")
        
        # Retrieve relevant documents
        results = self.search(question, top_k=3)
        
        if not results:
            return "No relevant documents found."
        
        # Prepare context
        context_parts = []
        print(f"📄 Found {len(results)} relevant chunks:")
        for i, (content, metadata, filename, similarity) in enumerate(results):
            context_parts.append(content)
            print(f"   {i+1}. {filename} (similarity: {similarity:.3f})")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Based on the following context from documents, answer the question:

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.complete(prompt)
        return response.text

# Test the custom implementation
if __name__ == "__main__":
    print("🚀 Testing Custom PostgreSQL RAG Implementation")
    
    # Initialize
    rag = CustomPostgresRAG()
    
    # Load documents
    from src.contextual_rag.rag.vector_store import VectorStoreManager
    manager = VectorStoreManager()
    documents = manager.load_processed_documents()
    
    print(f"📚 Loaded {len(documents)} documents")
    
    # Add ALL documents
    rag.add_documents(documents)  # Process all 2,303 documents
    
    # Test search and answer
    question = "What are the procurement approval requirements?"
    answer = rag.answer_question(question)
    
    print(f"\n📝 Question: {question}")
    print(f"📝 Answer: {answer}")