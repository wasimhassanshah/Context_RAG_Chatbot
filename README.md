 # 🚀 Advanced Contextual RAG Chatbot

> Enterprise-grade Retrieval-Augmented Generation system with CrewAI 4-agent pipeline, Arize Phoenix monitoring, and RAGAs evaluation for Abu Dhabi government procurement standards, HR policies, and security guidelines.


## 🌟 Key Features

- **4-Agent CrewAI Pipeline**: Specialized agents for retrieval, reranking, generation, and evaluation
- **Local AI Models**: Cost-effective Ollama integration with privacy-focused local inference
- **Enterprise Document Processing**: Docling pipeline for high-quality PDF/DOCX processing
- **Production Database**: PostgreSQL + PGVector for scalable vector operations
- **Pure RAGAs Evaluation**: Industry-standard ML evaluation with ground truth benchmarking
- **Phoenix Monitoring**: Real-time observability with Arize Phoenix dashboard
- **Multi-Interface**: FastAPI web app with Phoenix monitoring integration

---
## Command to Run APP

**python start_phoenix.py(TERMINAL 1)**

and

**ppython src/contextual_rag/ui/web_app.py(TERMINAL 2)**


## 🧠 System Architecture

### **4-Agent CrewAI Pipeline**
```
User Query → Retriever Agent → Reranker Agent → Generator Agent → Evaluator Agent → Response
```

| Agent | Description | Responsibility |
|-------|-------------|----------------|
| **Retriever Agent** | Vector search using PostgreSQL + PGVector | Finds relevant document chunks from 2,303 processed pieces |
| **Reranker Agent** | Document relevance optimization | Reranks retrieved documents using all-minilm model |
| **Generator Agent** | Response generation using Llama 3.2:1b | Creates contextual responses with source citations |
| **Evaluator Agent** | Pure RAGAs evaluation system | Validates response quality with 5 ML metrics |

### **Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Document Processing** | Docling Data Pipeline | Enterprise-grade PDF/DOCX processing |
| **Database** | PostgreSQL + PGVector | Scalable vector storage and retrieval |
| **AI Models** | Ollama (Local) | Privacy-focused, cost-effective inference |
| **Agent Framework** | CrewAI | Multi-agent orchestration and workflows |
| **Evaluation** | RAGAs | Scientific performance evaluation |
| **Monitoring** | Arize Phoenix | Real-time observability and debugging |
| **Interface** | Custom FastAPI | Web application with Phoenix integration |

---

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd C_RAG

# Setup Python environment (requires Python 3.12+)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies using UV (recommended)
pip install uv
uv pip install -r requirements.txt
```

### 2. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
```

**Required Environment Variables:**
```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=contextual_rag
POSTGRES_USER=your_postgres_username
POSTGRES_PASSWORD=your_postgres_password

# RAGAs Evaluation
GROQ_API_KEY=your_groq_api_key_here

# Phoenix Monitoring (optional)
PHOENIX_HOST=localhost
PHOENIX_PORT=6006
```

### 3. **Database Setup**
```bash
# Install PostgreSQL 16 with PGVector extension
# Run automated setup
python scripts/setup_database.py

# Verify database connection
python scripts/setup_rag_system.py
```

### 4. **Local AI Models Setup**
```bash
# Install Ollama from https://ollama.ai
# Pull required models
ollama pull llama3.2:1b          # Main LLM (1B parameters)
ollama pull nomic-embed-text     # Embeddings (768-dim)
ollama pull all-minilm           # Reranking model

# Verify models are available
ollama list
```

### 5. **Document Processing**
```bash
# Add your PDF/DOCX files to data/raw/ folder
# Process documents with Docling
python src/contextual_rag/document_processing/docling_processor.py

# This will create processed JSON files in data/processed/
# Processing time: ~90+ minutes for 6 documents (2,303 chunks)
```

### 6. **Launch the Application**

#### **FastAPI Web Interface** (Primary)
```bash
# Start the web application
python src/contextual_rag/main.py

# Access interfaces:
# FastAPI App: http://localhost:8000
# Phoenix Dashboard: http://localhost:6006
# Interactive Docs: http://localhost:8000/docs
```

#### **Phoenix Interactive Chat**
```bash
# Terminal-based chat with monitoring
python src/contextual_rag/ui/phoenix_chat.py
```

#### **Direct RAG Testing**
```bash
# Test core RAG system directly
python src/contextual_rag/main.py --test-mode
```

---

## 🔧 Configuration

### **Local AI Models** (config/ollama_models.yaml)
```yaml
models:
  llm:
    name: "llama3.2:1b"
    temperature: 0.1
    max_tokens: 4096
    context_window: 128000
    use_case: "Main reasoning and chat responses"
    
  embedding:
    name: "nomic-embed-text"
    use_case: "Document embeddings and similarity search"
    
  reranker:
    name: "all-minilm"
    use_case: "Lightweight document reranking"

server:
  host: "localhost"
  port: 11434
  base_url: "http://localhost:11434"
```

### **CrewAI Agent Configuration**
- **Retriever**: Searches 2,303 document chunks using vector similarity
- **Reranker**: Optimizes top 10 results to select best 3 documents
- **Generator**: Creates responses using selected context
- **Evaluator**: Validates quality using RAGAs metrics

---

## 📊 Performance Metrics

### **System Specifications**
- **Documents Processed**: 6 files → 2,303 high-quality chunks
- **Vector Dimensions**: 768 (nomic-embed-text)
- **Database**: PostgreSQL 16 + PGVector
- **Processing Environment**: Windows 11, Python 3.12.4

### **Performance Results**
- **Document Processing**: 0.7 docs/second (Docling pipeline)
- **Embedding Generation**: ~1.3 seconds per embedding
- **Vector Search**: 1-2 second response time
- **Full Pipeline**: 180-220 seconds (including RAGAs evaluation)
- **RAGAs Evaluation**: 30-60 seconds (using Groq API)

### **Quality Metrics** (RAGAs Evaluation)
```
✅ Overall Score: 0.702
📊 Detailed Metrics:
   • Answer Relevancy: 0.846
   • Context Precision: 1.000
   • Context Recall: 1.000  
   • Semantic Similarity: 0.913
   • Answer Correctness: 0.453
```

### **Document Breakdown**
| Document | Size | Chunks | Processing Time |
|----------|------|--------|-----------------|
| Information Security | 6.08 MB | 843 | ~25 minutes |
| Procurement Manual (Ariba) | 7.11 MB | 668 | ~20 minutes |
| Procurement Manual (Business) | 2.51 MB | 316 | ~12 minutes |
| HR Bylaws | 1.12 MB | 268 | ~10 minutes |
| Abu Dhabi Procurement | 1.38 MB | 203 | ~8 minutes |
| Q&A Questions | 0.03 MB | 5 | ~1 minute |

---

## 🧪 Testing & Evaluation

### **Comprehensive Test Suite**
```bash
# Run all tests
python -m pytest tests/ -v

# Individual test modules
python tests/test_ollama_llm.py      # LLM functionality (5/5 passed)
python tests/test_docling_processor.py  # Document processing (6/6 passed)
python tests/test_vector_store.py   # Vector operations (all passed)
```

### **RAGAs Evaluation System**
- **Pure RAGAs Mode**: No fallback mechanisms, ML-only evaluation
- **Ground Truth Integration**: Automatic comparison against established answers
- **5 Metrics Evaluated**:
  - Answer Relevancy
  - Context Precision
  - Context Recall
  - Answer Similarity
  - Answer Correctness
- **LLM Backend**: Groq (llama-3.1-8b-instant) for evaluation
- **Embeddings**: Ollama (nomic-embed-text) for local processing


---

## 🖥️ User Interfaces

### **1. FastAPI Web Application** (Primary Interface)
- **URL**: http://localhost:8000
- **Features**:
  - ChatGPT-style conversation interface
  - Real-time RAGAs evaluation display
  - Document source citations
  - Conversation history
  - Phoenix monitoring integration
  - Interactive API documentation at `/docs`

### **2. Arize Phoenix Dashboard** (Monitoring)
- **URL**: http://localhost:6006
- **Features**:
  - Real-time pipeline tracing
  - Performance analytics
  - Token usage monitoring
  - Query/response inspection
  - Agent interaction visualization

---

## 🐳 Docker Deployment

```bash
# Build container
docker build -t contextual-rag-chatbot .

# Run with all interfaces
docker run -p 8000:8000 -p 6006:6006 -p 11434:11434 contextual-rag-chatbot

# Access interfaces
# FastAPI: http://localhost:8000
# Phoenix: http://localhost:6006
# Ollama: http://localhost:11434
```

**Docker Compose Setup:**
```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: contextual_rag
      POSTGRES_USER: your_postgres_id
      POSTGRES_PASSWORD: your_postgres_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  contextual-rag:
    build: .
    ports:
      - "8000:8000"
      - "6006:6006"
    depends_on:
      - postgres
      - ollama
    environment:
      - POSTGRES_HOST=postgres
      - OLLAMA_BASE_URL=http://ollama:11434

volumes:
  postgres_data:
  ollama_data:
```