# 🚀 Advanced Contextual RAG Chatbot

> Enterprise-grade Retrieval-Augmented Generation system with CrewAI 4-agent pipeline, Arize Phoenix monitoring, and RAGAs evaluation for Abu Dhabi government procurement standards, HR policies, and security guidelines.

## 🎥 Demo Video

**Please watch the demo video located in:** `Demo-C-RAG-Video/`

## 🌟 Key Features

- **4-Agent CrewAI Pipeline**: Specialized agents for retrieval, reranking, generation, and evaluation
- **Local AI Models**: Cost-effective Ollama integration with privacy-focused local inference
- **Enterprise Document Processing**: Docling pipeline for high-quality PDF/DOCX processing
- **Production Database**: PostgreSQL + PGVector for scalable vector operations
- **Pure RAGAs Evaluation**: Industry-standard ML evaluation with ground truth benchmarking
- **Phoenix Monitoring**: Real-time observability with Arize Phoenix dashboard
- **Multi-Interface**: FastAPI web app with Phoenix monitoring integration

---

## 📁 Project Structure

```
D:\C_RAG/
├── .env                           # Environment configuration
├── .gitignore                     # Git ignore patterns  
├── pyproject.toml                 # Dependencies & config
├── README.md                      # This file
├── .venv/                         # Virtual environment

├── data/
│   ├── raw/                       # Original documents (6 files)
│   │   ├── Abu Dhabi Procurement Standards.PDF (1.38 MB)
│   │   ├── HR Bylaws.pdf (1.12 MB)
│   │   ├── Information Security.pdf (6.08 MB)  
│   │   ├── Procurement Manual (Ariba Aligned).PDF (7.11 MB)
│   │   ├── Procurement Manual (Business Process).PDF (2.51 MB)
│   │   └── Document Q&A Questions.docx (0.03 MB)
│   │
│   └── processed/                 # Docling processed chunks
│       ├── doc_*.json             # 2,303 processed chunks

├── src/
│   └── contextual_rag/
│       ├── main.py                # Main application entry
│       │
│       ├── agents/                # CrewAI 4-Agent System
│       │   ├── master_agent.py           # Pipeline orchestrator
│       │   ├── retriever_agent.py        # Document retrieval
│       │   ├── reranker_agent.py         # Document reranking  
│       │   ├── generator_agent.py        # Response generation
│       │   ├── evaluator_agent.py        # RAGAs evaluation
│       │   ├── crew_orchestrator.py      # CrewAI orchestration
│       │   └── tools/                    # Agent tools
│       │
│       ├── document_processing/   # Docling Processing
│       │   └── docling_processor.py
│       │
│       ├── rag/                   # LlamaIndex RAG
│       │   └── vector_store.py
│       │
│       ├── evaluation/            # RAGAs Evaluation
│       │   ├── ragas_evaluator.py        # Pure RAGAs implementation
│       │   ├── results/                  # Evaluation results
│       │   └── testing/
│       │       ├── test_questions.py
│       │       ├── ground_truth_generator.py
│       │       └── ground_truth_dataset.json
│       │
│       └── ui/                    # FastAPI Web Interface
│           └── (FastAPI application)

├── tests/                         # Comprehensive test suite
│   ├── test_ollama_llm.py         # LLM functionality tests
│   ├── test_docling_processor.py  # Document processing tests
│   └── test_vector_store.py       # Vector database tests

├── scripts/                       # Setup automation
│   ├── setup_database.py          # PostgreSQL automation
│   ├── setup_docling.py           # Docling configuration
│   └── setup_rag_system.py        # RAG system verification

└── config/                        # Configuration files
    └── ollama_models.yaml         # Model specifications
```

---

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
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=RagUser2024

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

### **Custom Evaluation**
```bash
# Run RAGAs evaluation on test dataset
python src/contextual_rag/evaluation/ragas_evaluator.py

# Generate ground truth for new questions
python src/contextual_rag/evaluation/testing/ground_truth_generator.py
```

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

### **3. Terminal Interface** (Development)
- **Command**: `python src/contextual_rag/ui/phoenix_chat.py`
- **Features**:
  - Direct command-line interaction
  - Debug output visibility
  - Quick testing capabilities

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
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: RagUser2024
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

---

## ☁️ AWS Cloud Deployment

### **Prerequisites**
- Download AWS CLI from: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

### **1. IAM User Setup**

#### **Create IAM User**
- Search for IAM in AWS Console
- Click on Users → Create user
- Enter the user name (e.g., `rag-deployment-user`)

#### **Attach Policies**
Attach the following policies to the user:
- `AdministratorAccess`
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

#### **Generate Access Keys**
- Go to the created user → Security credentials tab
- Click Create access key → Select Use Case: Local code
- Click Next and Create access key

#### **Configure AWS CLI**
```bash
aws configure
AWS Access Key ID [****************ZPUP]: your-access-key-id
AWS Secret Access Key [****************P184]: your-secret-access-key
Default region name [us-east-1]: us-east-1
Default output format [None]: json
```

### **2. S3 Bucket Setup**
- Search for S3 in AWS Console
- Click Create bucket
- Enter a globally unique bucket name (e.g., `your-company-rag-artifacts-2024`)
- Keep all other settings as default
- Click Create bucket

### **3. ECR Repository Setup**
- Search for ECR in AWS Console
- Click Create repository
- Enter Repository name (e.g., `abu-dhabi-rag-system`)
- Keep settings as default
- Click Create
- **Note**: Copy the repository URI for later use in GitHub secrets

### **4. EC2 Instance Setup**

#### **Launch Instance**
- Search for EC2 in AWS Console
- Click Launch instance
- Configure the following:
  - **Name**: `rag-system-server`
  - **OS Image**: Ubuntu Server 22.04 LTS
  - **Instance Type**: t3.medium (or as per requirement)
  - **Key pair**: Create new or select existing
  - **Security Group**: Configure as below

#### **Security Group Configuration**
Select the following options:
- ✅ Allow SSH traffic from → Anywhere (0.0.0.0/0)
- ✅ Allow HTTPS traffic from the internet
- ✅ Allow HTTP traffic from the internet

#### **Configure Additional Ports**
After instance creation:
1. Select your Instance ID → Go to Security tab
2. Click on the Security group → Click Edit inbound rules
3. Add rules for:
   - **Type**: Custom TCP, **Port**: 8501 (Streamlit), **Source**: 0.0.0.0/0
   - **Type**: Custom TCP, **Port**: 6006 (Phoenix), **Source**: 0.0.0.0/0

### **5. GitHub Secrets Configuration**
Navigate to your repository **Settings → Secrets and variables → Actions**

Add the following secrets:

#### **AWS Credentials**
- `AWS_ACCESS_KEY_ID`: Your IAM user access key
- `AWS_SECRET_ACCESS_KEY`: Your IAM user secret key
- `AWS_DEFAULT_REGION`: us-east-1 (or your preferred region)

#### **AWS Resources**
- `AWS_ECR_LOGIN_URI`: Your ECR repository URI
- `ECR_REPOSITORY_NAME`: Your ECR repository name

#### **API Keys**
- `GROQ_API_KEY`: Your Groq API key
- `POSTGRES_HOST`: Your PostgreSQL host
- `POSTGRES_USER`: Your PostgreSQL username
- `POSTGRES_PASSWORD`: Your PostgreSQL password

### **6. Install Docker on EC2**
SSH into your EC2 instance and run:

```bash
# Update System
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

# Verify installation
docker --version
```

### **7. GitHub Actions Self-Hosted Runner Setup**

#### **Configure Runner in GitHub**
- Go to your repository **Settings → Actions → Runners**
- Click **New self-hosted runner**
- Select **Linux** as Runner image

#### **Setup Commands for EC2**
```bash
# Create Runner Directory
mkdir actions-runner && cd actions-runner

# Download Latest Runner Package
curl -o actions-runner-linux-x64-2.319.1.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz

# Extract Installer
tar xzf ./actions-runner-linux-x64-2.319.1.tar.gz

# Configure Runner (replace with your repository URL and token)
./config.sh --url https://github.com/YOUR-USERNAME/YOUR-REPOSITORY --token YOUR-REGISTRATION-TOKEN

# Start Runner
./run.sh

# Or run as service (recommended for production)
sudo ./svc.sh install
sudo ./svc.sh start
```

### **8. Deployment Verification**
After deployment, verify the following:
- **FastAPI App**: http://your-ec2-public-ip:8000
- **Phoenix Dashboard**: http://your-ec2-public-ip:6006
- **EC2 Instance**: Running and accessible
- **GitHub Actions**: Runner connected and workflows executing

### **Security Notes**
⚠️ **Important Security Considerations:**
- Never commit API keys to your repository
- Use GitHub secrets for all sensitive information
- Regularly rotate your AWS access keys
- Consider using AWS IAM roles for production
- Monitor your AWS usage and costs
- Enable AWS CloudTrail for audit logging

### **Cost Optimization**
💰 **Cost Management Tips:**
- Use t3.micro for development (eligible for free tier)
- Stop EC2 instances when not in use
- Monitor S3 storage costs
- Set up AWS billing alerts
- Use ECR lifecycle policies

---

## ☁️ Production Deployment

### **Scalability Considerations**
- **Database**: PostgreSQL with connection pooling
- **Model Serving**: Ollama with GPU acceleration (optional)
- **Load Balancing**: Multiple FastAPI instances
- **Caching**: Redis for frequently accessed embeddings
- **Monitoring**: Production Phoenix deployment

### **Performance Optimization**
```python
# Example production configuration
PRODUCTION_CONFIG = {
    "max_workers": 4,
    "batch_size": 32,
    "cache_embeddings": True,
    "enable_gpu": True,
    "connection_pool_size": 20
}
```

---

## 🔍 Architecture Deep Dive

### **Data Flow**
```
1. Document Ingestion (Docling)
   ├── PDF/DOCX parsing
   ├── Intelligent chunking
   └── Metadata extraction

2. Vector Storage (PostgreSQL + PGVector)
   ├── Embedding generation (nomic-embed-text)
   ├── Vector indexing
   └── Metadata storage

3. Query Processing (4-Agent Pipeline)
   ├── Query analysis
   ├── Vector similarity search (Retriever)
   ├── Result reranking (Reranker)
   ├── Response generation (Generator)
   └── Quality evaluation (Evaluator)

4. Response Delivery
   ├── FastAPI web interface
   ├── Phoenix monitoring
   └── RAGAs quality scores
```

### **Agent Interaction Pattern**
```python
class RAGPipeline:
    def process_query(self, query: str) -> Dict:
        # Step 1: Retrieve relevant documents
        documents = self.retriever_agent.search(query)
        
        # Step 2: Rerank for relevance
        ranked_docs = self.reranker_agent.rerank(documents, query)
        
        # Step 3: Generate contextual response
        response = self.generator_agent.generate(query, ranked_docs)
        
        # Step 4: Evaluate quality
        evaluation = self.evaluator_agent.evaluate(query, response, ranked_docs)
        
        return {
            "response": response,
            "sources": ranked_docs,
            "evaluation": evaluation,
            "metadata": self.get_metadata()
        }
```

---

## 📈 Monitoring & Observability

### **Phoenix Integration**
- **Real-time Tracing**: Every query automatically traced
- **Performance Metrics**: Latency, token usage, success rates
- **Agent Visualization**: Multi-agent interaction flows
- **Error Tracking**: Automatic error detection and logging

### **RAGAs Continuous Evaluation**
- **Automatic Scoring**: Every response evaluated in real-time
- **Ground Truth Comparison**: Benchmark against established answers
- **Quality Trends**: Track performance over time
- **Alert System**: Notifications for quality degradation

### **Custom Metrics**
```python
# Example monitoring setup
from contextual_rag.monitoring import MetricsCollector

metrics = MetricsCollector()
metrics.track_query_latency(query_time)
metrics.track_rag_quality(ragas_scores)
metrics.track_agent_performance(agent_metrics)
```

---

## 🛠️ Development Guide

### **Adding New Documents**
1. Place PDF/DOCX files in `data/raw/`
2. Run document processing: `python src/contextual_rag/document_processing/docling_processor.py`
3. Verify processing results in `data/processed/`
4. Test retrieval with new documents

### **Customizing Agents**
```python
# Example: Custom reranking strategy
class CustomRerankerAgent(RerankerAgent):
    def rerank(self, documents, query):
        # Implement custom reranking logic
        return super().rerank(documents, query)
```

### **Adding Evaluation Metrics**
```python
# Extend RAGAs evaluation
from contextual_rag.evaluation.ragas_evaluator import RAGAsEvaluator

class CustomEvaluator(RAGAsEvaluator):
    def evaluate_custom_metric(self, query, response, context):
        # Implement custom evaluation logic
        pass
```

### **Model Configuration**
```bash
# Switch to different Ollama models
ollama pull llama2:13b           # Larger model for better quality
ollama pull mistral:7b           # Alternative LLM
ollama pull all-mpnet-base-v2    # Different embedding model

# Update config/ollama_models.yaml accordingly
```

---

## 🔒 Security & Privacy

### **Data Privacy**
- **Local Processing**: All AI inference runs locally via Ollama
- **No Data Transmission**: Documents never leave your infrastructure
- **Secure Storage**: PostgreSQL with proper access controls
- **Audit Logging**: Complete query and response tracking

### **Authentication & Authorization**
```python
# Example security configuration
SECURITY_CONFIG = {
    "enable_authentication": True,
    "jwt_secret": "your-secret-key",
    "session_timeout": 3600,
    "rate_limiting": {
        "max_requests_per_minute": 60
    }
}
```

---

## 🐛 Troubleshooting

### **Common Issues**

#### **1. Ollama Connection Issues**
```bash
# Check Ollama status
ollama list
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

#### **2. PostgreSQL Connection Issues**
```bash
# Check database connection
python scripts/setup_database.py --test

# Verify PGVector extension
psql -U rag_user -d contextual_rag -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

#### **3. Document Processing Issues**
```bash
# Check Docling setup
python scripts/setup_docling.py --verify

# Process single document for debugging
python src/contextual_rag/document_processing/docling_processor.py --single-file "path/to/document.pdf"
```

#### **4. RAGAs Evaluation Issues**
```bash
# Verify Groq API key
python src/contextual_rag/evaluation/ragas_evaluator.py --test-connection

# Run evaluation in debug mode
python src/contextual_rag/evaluation/ragas_evaluator.py --debug
```

### **Performance Tuning**
```python
# Optimize for speed vs. quality
PERFORMANCE_MODES = {
    "fast": {
        "embedding_model": "all-minilm",
        "max_chunks": 5,
        "skip_reranking": True
    },
    "balanced": {
        "embedding_model": "nomic-embed-text",
        "max_chunks": 10,
        "enable_reranking": True
    },
    "quality": {
        "embedding_model": "nomic-embed-text",
        "max_chunks": 20,
        "enable_reranking": True,
        "enable_evaluation": True
    }
}
```

---

## 📚 Dependencies

### **Core Framework**
```toml
# Essential dependencies from pyproject.toml
python-dotenv = ">=1.0.0"
pydantic = ">=2.5.0"

# Document Processing - Docling Pipeline
docling = ">=1.14.0"
docling-core = ">=1.5.0"
docling-ibm-models = ">=1.0.4"
docling-parse = ">=1.9.0"

# LlamaIndex + PostgreSQL RAG
llama-index = ">=0.10.0"
llama-index-embeddings-ollama = ">=0.1.0"
llama-index-llms-ollama = ">=0.1.0"
llama-index-vector-stores-postgres = ">=0.1.0"

# Database
psycopg2-binary = ">=2.9.9"
pgvector = ">=0.2.4"
sqlalchemy = ">=2.0.0"

# Local AI Models
ollama = ">=0.2.0"

# Agent Framework
crewai = ">=0.28.0"
crewai-tools = ">=0.4.0"
langchain = ">=0.1.0"
langchain-ollama = ">=0.3.6"
langchain-groq = ">=0.1.0"

# Evaluation & Monitoring
ragas = ">=0.1.0"
arize-phoenix = ">=3.0.0"
openinference-instrumentation = ">=0.1.0"

# Web Interface
fastapi = ">=0.104.0"
uvicorn = ">=0.24.0"
```

---

## 📖 API Documentation

### **FastAPI Endpoints**

#### **Chat Interface**
```python
POST /chat
{
    "query": "What are the procurement standards for Abu Dhabi?",
    "conversation_id": "optional-session-id",
    "include_evaluation": true
}

Response:
{
    "response": "Based on the Abu Dhabi Procurement Standards document...",
    "sources": [
        {
            "document": "Abu Dhabi Procurement Standards.PDF",
            "chunk_id": "doc_e9f0c4d3g678_chunk_15",
            "relevance_score": 0.95
        }
    ],
    "evaluation": {
        "overall_score": 0.702,
        "answer_relevancy": 0.846,
        "context_precision": 1.000
    },
    "metadata": {
        "processing_time": 2.3,
        "tokens_used": 1247,
        "model_used": "llama3.2:1b"
    }
}
```

#### **Document Management**
```python
GET /documents
# Lists all processed documents

POST /documents/upload
# Upload new PDF/DOCX for processing

GET /documents/{doc_id}/chunks
# Get chunks for specific document
```

#### **System Status**
```python
GET /health
# System health check

GET /metrics
# Performance metrics

GET /models
# Available Ollama models
```

---

## 🤝 Contributing

### **Development Setup**
```bash
# Fork and clone repository
git clone https://github.com/yourusername/contextual-rag-chatbot.git
cd contextual-rag-chatbot

# Create development environment
python -m venv .venv
.venv\Scripts\activate
pip install uv
uv pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Start development server
python src/contextual_rag/main.py --dev-mode
```

### **Contribution Guidelines**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`python -m pytest`)
5. Run linting (`black . && flake8`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### **Code Standards**
- **Python**: Black formatting, type hints required
- **Documentation**: Docstrings for all public functions
- **Testing**: Minimum 80% code coverage
- **Commit Messages**: Follow conventional commits

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support & Community

### **Getting Help**
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check module docstrings and inline comments
- **Phoenix Dashboard**: Monitor system performance at `localhost:6006`
- **API Docs**: Interactive documentation at `localhost:8000/docs`

### **Community Resources**
- **RAGAs Documentation**: [RAGAs Official Docs](https://docs.ragas.io/)
- **CrewAI Framework**: [CrewAI Documentation](https://docs.crewai.com/)
- **Arize Phoenix**: [Phoenix Observability](https://docs.arize.com/phoenix)
- **Ollama Models**: [Ollama Library](https://ollama.ai/library)

---

## 📊 Project Stats

**📈 Current Status: PRODUCTION READY**

- ✅ **6 Documents Processed** → 2,303 high-quality chunks
- ✅ **4-Agent CrewAI Pipeline** → Fully operational
- ✅ **Pure RAGAs Evaluation** → Industry-standard metrics
- ✅ **Local AI Models** → Privacy-focused, cost-effective
- ✅ **Enterprise Database** → PostgreSQL + PGVector
- ✅ **Real-time Monitoring** → Phoenix observability
- ✅ **Web Interface** → FastAPI with interactive docs
- ✅ **Comprehensive Testing** → 100% core functionality covered

---

**🎯 Enterprise-Ready Contextual RAG System with Advanced Multi-Agent Architecture**

*Powered by CrewAI • Monitored by Phoenix • Evaluated by RAGAs • Served by Ollama*