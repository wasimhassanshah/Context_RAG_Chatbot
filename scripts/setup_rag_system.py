"""
Setup Script for LlamaIndex RAG System
Location: scripts/setup_rag_system.py
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if LlamaIndex dependencies are installed"""
    print("🔧 Checking LlamaIndex dependencies...")
    
    required_packages = [
        ('llama_index', 'llama-index'),
        ('llama_index.core', 'llama-index-core'),
        ('llama_index.embeddings.ollama', 'llama-index-embeddings-ollama'),
        ('llama_index.llms.ollama', 'llama-index-llms-ollama'),
        ('llama_index.vector_stores.postgres', 'llama-index-vector-stores-postgres'),
        ('sqlalchemy', 'sqlalchemy'),
        ('asyncpg', 'asyncpg')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}: Available")
        except ImportError:
            print(f"   ❌ {package_name}: Missing")
            missing_packages.append(package_name)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing dependencies"""
    if not missing_packages:
        return True
    
    print(f"\n📦 Installing {len(missing_packages)} missing packages...")
    
    for package in missing_packages:
        print(f"   Installing {package}...")
        try:
            result = subprocess.run([
                "uv", "add", package
            ], capture_output=True, text=True, check=True)
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
        except FileNotFoundError:
            # Fallback to pip
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, check=True)
                print(f"   ✅ {package} installed successfully (via pip)")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e}")
                return False
    
    return True

def check_ollama_status():
    """Check if Ollama is running and models are available"""
    print("\n🤖 Checking Ollama status...")
    
    try:
        # Add the current directory to Python path
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        
        # Import after path fix
        from src.contextual_rag.agents.ollama_manager import OllamaManager
        
        manager = OllamaManager()
        
        # Check server status
        if not manager.check_ollama_status():
            print("   ❌ Ollama server not running")
            return False
        
        # Check required models
        required_models = manager.check_required_models()
        missing_models = [name for name, available in required_models.items() if not available]
        
        if missing_models:
            print(f"   ❌ Missing models: {missing_models}")
            return False
        
        print("   ✅ Ollama server running")
        print("   ✅ All required models available")
        return True
        
    except Exception as e:
        print(f"   ❌ Ollama check failed: {e}")
        return False

def check_database_connection():
    """Check PostgreSQL database connection"""
    print("\n🗄️ Checking database connection...")
    
    try:
        import psycopg2
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Database configuration
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'contextual_rag'),
            'user': os.getenv('POSTGRES_USER', 'rag_user'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
        
        # Test connection
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Check for vector extension
        cursor.execute("SELECT 1 WHERE EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        has_vector = cursor.fetchone() is not None
        
        cursor.close()
        conn.close()
        
        if has_vector:
            print("   ✅ Database connection successful")
            print("   ✅ PGVector extension available")
            return True
        else:
            print("   ❌ PGVector extension not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return False

def check_processed_documents():
    """Check if processed documents exist"""
    print("\n📄 Checking processed documents...")
    
    processed_path = Path("data/processed")
    json_files = list(processed_path.glob("*.json"))
    
    if json_files:
        print(f"   ✅ Found {len(json_files)} processed documents")
        
        # Show document summary
        total_chunks = 0
        for json_file in json_files:
            try:
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                chunks = len(data.get('chunks', []))
                total_chunks += chunks
                print(f"      📄 {data.get('filename', json_file.name)}: {chunks} chunks")
            except:
                print(f"      ⚠️ {json_file.name}: Could not read")
        
        print(f"   📊 Total chunks available: {total_chunks}")
        return total_chunks > 0
    else:
        print("   ❌ No processed documents found")
        print("   💡 Run document processing first: python tests/test_docling_processor.py")
        return False

def create_required_directories():
    """Create required directories"""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        "src/contextual_rag/rag",
        "tests"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_path}")
        else:
            print(f"   📁 Exists: {dir_path}")
    
    # Create __init__.py files
    init_files = [
        "src/contextual_rag/rag/__init__.py"
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        if not path.exists():
            path.touch()
            print(f"   ✅ Created: {init_file}")
    
    return True

def run_quick_test():
    """Run a quick test of the RAG system setup"""
    print("\n🧪 Running quick setup test...")
    
    try:
        # Test LlamaIndex imports
        from llama_index.core import Settings
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        print("   ✅ LlamaIndex imports successful")
        
        # Test Ollama embedding model
        embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        print("   ✅ Ollama embedding model configured")
        
        # Test Ollama LLM
        llm_model = Ollama(
            model="llama3.1:8b",
            base_url="http://localhost:11434"
        )
        print("   ✅ Ollama LLM configured")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 LLAMAINDEX RAG SYSTEM SETUP")
    print("=" * 60)
    
    # Step 1: Create directories
    if not create_required_directories():
        print("❌ Directory creation failed")
        return False
    
    # Step 2: Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        if not install_dependencies(missing_packages):
            print("❌ Dependency installation failed")
            return False
    
    # Step 3: Check Ollama
    if not check_ollama_status():
        print("❌ Ollama check failed")
        print("💡 Make sure Ollama is running: ollama serve")
        return False
    
    # Step 4: Check database
    if not check_database_connection():
        print("❌ Database check failed")
        print("💡 Check PostgreSQL is running and configured")
        return False
    
    # Step 5: Check processed documents
    if not check_processed_documents():
        print("❌ No processed documents available")
        print("💡 Run document processing first")
        return False
    
    # Step 6: Quick test
    if not run_quick_test():
        print("❌ Quick test failed")
        return False
    
    print("\n🎉 RAG SYSTEM SETUP COMPLETE!")
    print("=" * 40)
    print("✅ Dependencies installed")
    print("✅ Ollama models ready")
    print("✅ Database connection verified")
    print("✅ Processed documents available")
    print("✅ LlamaIndex configuration tested")
    
    print("\n🚀 Next steps:")
    print("1. Run: python tests/test_vector_store.py")
    print("2. If tests pass, RAG system will be fully operational")
    
    return True

if __name__ == "__main__":
    main()