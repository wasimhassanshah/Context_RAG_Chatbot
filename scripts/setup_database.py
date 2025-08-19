"""
PostgreSQL + PGVector Setup Script
Automates database creation and configuration for the RAG project
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class PostgreSQLSetup:
    """Handles PostgreSQL and PGVector setup"""
    
    def __init__(self):
        self.admin_user = "postgres"
        self.admin_password = os.getenv("POSTGRES_ADMIN_PASSWORD", "")
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        
        # Project database settings
        self.db_name = os.getenv("POSTGRES_DB", "contextual_rag")
        self.db_user = os.getenv("POSTGRES_USER", "rag_user")
        self.db_password = os.getenv("POSTGRES_PASSWORD", "")
        
    def test_admin_connection(self):
        """Test connection as postgres admin user"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.admin_user,
                password=self.admin_password,
                dbname="postgres"
            )
            conn.close()
            print("✅ Admin connection successful")
            return True
        except Exception as e:
            print(f"❌ Admin connection failed: {e}")
            return False
    
    def create_database_and_user(self):
        """Create project database and user"""
        try:
            # Connect as admin
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.admin_user,
                password=self.admin_password,
                dbname="postgres"
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (self.db_name,)
            )
            
            if cursor.fetchone():
                print(f"📋 Database '{self.db_name}' already exists")
            else:
                # Create database
                cursor.execute(f'CREATE DATABASE "{self.db_name}"')
                print(f"✅ Created database '{self.db_name}'")
            
            # Check if user exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_user WHERE usename = %s",
                (self.db_user,)
            )
            
            if cursor.fetchone():
                print(f"📋 User '{self.db_user}' already exists")
            else:
                # Create user
                cursor.execute(
                    f"CREATE USER {self.db_user} WITH PASSWORD %s",
                    (self.db_password,)
                )
                print(f"✅ Created user '{self.db_user}'")
            
            # Grant privileges
            cursor.execute(f'GRANT ALL PRIVILEGES ON DATABASE "{self.db_name}" TO {self.db_user}')
            print(f"✅ Granted privileges to '{self.db_user}'")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to create database/user: {e}")
            return False
    
    def setup_pgvector_extension(self):
        """Install and configure PGVector extension"""
        try:
            # Connect to project database as admin
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.admin_user,
                password=self.admin_password,
                dbname=self.db_name
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create vector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("✅ PGVector extension installed")
            
            # Grant schema privileges to project user
            cursor.execute(f"GRANT USAGE ON SCHEMA public TO {self.db_user}")
            cursor.execute(f"GRANT CREATE ON SCHEMA public TO {self.db_user}")
            print(f"✅ Schema privileges granted to '{self.db_user}'")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup PGVector: {e}")
            print("💡 You may need to install PGVector manually")
            return False
    
    def test_project_connection(self):
        """Test connection with project credentials"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.db_user,
                password=self.db_password,
                dbname=self.db_name
            )
            
            cursor = conn.cursor()
            
            # Test vector extension
            cursor.execute("SELECT vector_dims('[1,2,3]'::vector)")
            result = cursor.fetchone()
            
            if result and result[0] == 3:
                print("✅ Project database connection successful")
                print("✅ PGVector extension working")
                cursor.close()
                conn.close()
                return True
            else:
                print("❌ PGVector test failed")
                cursor.close()
                conn.close()
                return False
                
        except Exception as e:
            print(f"❌ Project connection failed: {e}")
            return False
    
    def create_tables(self):
        """Create initial tables for the RAG system"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.db_user,
                password=self.db_password,
                dbname=self.db_name
            )
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(10) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create document chunks table with vector embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(768),  -- 768 dimensions for nomic-embed-text
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for vector similarity search
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            # Create index for document lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_filename 
                ON documents(filename)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
                ON document_chunks(document_id)
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("✅ Database tables created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create tables: {e}")
            return False
    
    def run_complete_setup(self):
        """Run complete database setup"""
        print("🚀 Starting PostgreSQL + PGVector Setup")
        print("=" * 60)
        
        # Check if we have admin password
        if not self.admin_password:
            self.admin_password = input("Enter PostgreSQL admin (postgres) password: ")
        
        if not self.db_password:
            self.db_password = input(f"Enter password for new user '{self.db_user}': ")
            
            # Update .env file
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, "a") as f:
                    f.write(f"\nPOSTGRES_PASSWORD={self.db_password}\n")
        
        # Run setup steps
        steps = [
            ("Testing admin connection", self.test_admin_connection),
            ("Creating database and user", self.create_database_and_user),
            ("Setting up PGVector extension", self.setup_pgvector_extension),
            ("Testing project connection", self.test_project_connection),
            ("Creating database tables", self.create_tables)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        print("\n🎉 PostgreSQL + PGVector setup complete!")
        print("✅ Database ready for RAG system")
        
        # Print connection details
        print(f"\n📊 Database Configuration:")
        print(f"   🔹 Host: {self.host}:{self.port}")
        print(f"   🔹 Database: {self.db_name}")
        print(f"   🔹 User: {self.db_user}")
        print(f"   🔹 Connection URL: postgresql://{self.db_user}:***@{self.host}:{self.port}/{self.db_name}")
        
        return True

def main():
    """Main setup function"""
    setup = PostgreSQLSetup()
    
    if setup.run_complete_setup():
        print("\n🚀 Ready to proceed with Docling document processing!")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()