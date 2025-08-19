"""
Real-time FastAPI Web Application - Sequential Agent Updates
Location: src/contextual_rag/ui/web_app.py
"""

import asyncio
import json
import logging
import time  # ADDED: Import time for processing time calculation
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

# Import your RAG system
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from contextual_rag.agents.master_agent import create_master_agent

# === PHOENIX AUTO-INSTRUMENTATION ===
try:
    import phoenix as px
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    import requests
    
    # Check if Phoenix is running
    requests.get("http://localhost:6006", timeout=1)
    
    # Configure OpenTelemetry to send to Phoenix
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    
    # OTLP exporter to Phoenix
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://localhost:6006/v1/traces",
        headers={}
    )
    
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Auto-instrument everything
    LangChainInstrumentor().instrument()
    LlamaIndexInstrumentor().instrument()
    
    print("✅ Phoenix tracing enabled and configured")
except Exception as e:
    print(f"⚠️ Phoenix tracing failed: {e}")
# === END PHOENIX ===

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Contextual RAG Chatbot", version="1.0.0")

# Static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "static" / "templates"

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# Global instances
master_agent = None

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 WebSocket connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Active connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global master_agent

    logger.info("🚀 Starting Clean RAG Web Application...")

    try:
        master_agent = create_master_agent()
        logger.info("✅ Master agent initialized")
        logger.info("✅ Application startup complete")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    try:
        if master_agent is None:
            return {
                'system_status': 'error',
                'error': 'Master agent not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        system_status = master_agent.get_system_status()
        return {
            'system_status': system_status.get('overall_status', 'unknown'),
            'agent_status': system_status.get('agent_status', {}),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return {
            'system_status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Process chat query"""
    try:
        data = await request.json()
        query = data.get('query', '').strip()

        if not query:
            return {'error': 'Empty query'}

        logger.info(f"💬 Processing query: {query[:50]}...")

        response = master_agent.process_query(
            query=query,
            response_type="adaptive",
            enable_citations=True
        )

        logger.info(f"✅ Query processed successfully")
        return response

    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        return {'error': str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agent updates"""
    try:
        await manager.connect(websocket)

        await manager.send_personal_message({
            'type': 'connection',
            'message': 'Connected to RAG system'
        }, websocket)

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get('type') == 'chat':
                await process_chat_websocket(websocket, message.get('query', ''))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)

async def process_chat_websocket(websocket: WebSocket, query: str):
    """Process chat with REAL-TIME sequential agent updates"""
    try:
        if not query.strip():
            return

        # FIXED: Track total processing time from start
        pipeline_start_time = time.time()
        logger.info(f"🔌 Processing WebSocket query: {query[:50]}...")

        # Stage 1: Document Retrieval - REAL-TIME
        await manager.send_personal_message({
            'type': 'agent_update',
            'agent': 'retriever',
            'status': 'working',
            'message': 'Searching documents...'
        }, websocket)

        # Execute retrieval and immediately show results
        retrieval_result = master_agent._execute_retrieval_stage(query, master_agent.pipeline_config)
        
        await manager.send_personal_message({
            'type': 'agent_result',
            'agent': 'retriever',
            'data': {
                'documents_found': retrieval_result.get('retrieval_count', 0),
                'status': 'completed'
            }
        }, websocket)

        if not retrieval_result.get('documents'):
            await manager.send_personal_message({
                'type': 'final_response',
                'data': master_agent._handle_no_results(query, datetime.now())
            }, websocket)
            return

        # Stage 2: Document Reranking - REAL-TIME
        await manager.send_personal_message({
            'type': 'agent_update',
            'agent': 'reranker',
            'status': 'working',
            'message': 'Reranking documents...'
        }, websocket)

        # Execute reranking and immediately show results
        reranking_result = master_agent._execute_reranking_stage(
            query, retrieval_result['documents'], master_agent.pipeline_config
        )
        
        await manager.send_personal_message({
            'type': 'agent_result',
            'agent': 'reranker',
            'data': {
                'documents_reranked': reranking_result.get('reranking_count', 0),
                'status': 'completed'
            }
        }, websocket)

        # Stage 3: Response Generation - REAL-TIME
        await manager.send_personal_message({
            'type': 'agent_update',
            'agent': 'generator',
            'status': 'working',
            'message': 'Generating response...'
        }, websocket)

        # Execute generation and immediately show results
        generation_result = master_agent._execute_generation_stage(
            query, reranking_result['documents'], "adaptive", True
        )
        
        await manager.send_personal_message({
            'type': 'agent_result',
            'agent': 'generator',
            'data': {
                'response_length': len(generation_result.get('answer', '')),
                'generation_method': 'llama3.2:1b',
                'status': 'completed'
            }
        }, websocket)

        # Stage 4: Response Evaluation - REAL-TIME
        await manager.send_personal_message({
            'type': 'agent_update',
            'agent': 'evaluator',
            'status': 'working',
            'message': 'Running RAGAs evaluation...'
        }, websocket)

        # Execute evaluation and immediately show results
        evaluation_result = master_agent._execute_evaluation_stage(
            query, generation_result['answer'], reranking_result['documents']
        )

        # Extract individual metrics for real-time display
        individual_scores = {}
        if evaluation_result and 'detailed_scores' in evaluation_result:
            detailed_scores = evaluation_result['detailed_scores']
            individual_scores = {
                'answer_relevancy': detailed_scores.get('answer_relevancy', 0),
                'context_precision': detailed_scores.get('context_precision', 0),
                'context_recall': detailed_scores.get('context_recall', 0),
                'answer_correctness': detailed_scores.get('answer_correctness', 0)
            }

        logger.info(f"🔍 Real-time individual scores: {individual_scores}")

        await manager.send_personal_message({
            'type': 'agent_result',
            'agent': 'evaluator',
            'data': {
                'evaluation_score': evaluation_result.get('overall_score', 0) if evaluation_result else 0,
                'individual_metrics': individual_scores,
                'has_ground_truth': evaluation_result.get('has_ground_truth', False) if evaluation_result else False,
                'evaluation_method': 'ragas',
                'status': 'completed'
            }
        }, websocket)

        # FIXED: Calculate total pipeline processing time
        total_pipeline_time = time.time() - pipeline_start_time

        # Compile final response with actual processing time
        final_response = {
            'answer': generation_result.get('answer', ''),
            'query': query,
            'processing_time_seconds': total_pipeline_time,  # FIXED: Actual pipeline time
            'timestamp': datetime.now().isoformat(),
            'pipeline_stages': {
                'retrieval': {
                    'documents_found': retrieval_result.get('retrieval_count', 0),
                    'status': retrieval_result.get('status', 'unknown')
                },
                'reranking': {
                    'documents_reranked': reranking_result.get('reranking_count', 0),
                    'status': reranking_result.get('status', 'unknown')
                },
                'generation': {
                    'response_length': len(generation_result.get('answer', '')),
                    'status': generation_result.get('status', 'unknown')
                },
                'evaluation': {
                    'overall_score': evaluation_result.get('overall_score', 0) if evaluation_result else 0,
                    'individual_metrics': individual_scores,
                    'status': evaluation_result.get('status', 'unknown') if evaluation_result else 'skipped'
                }
            },
            'sources': generation_result.get('sources_used', []),
            'quality_metrics': {
                'evaluation_score': evaluation_result.get('overall_score', 0) if evaluation_result else None,
                'detailed_scores': evaluation_result.get('detailed_scores', {}) if evaluation_result else {},
                'individual_metrics': individual_scores,
                'has_ground_truth': evaluation_result.get('has_ground_truth', False) if evaluation_result else False
            }
        }

        logger.info(f"🎉 Pipeline completed in {total_pipeline_time:.1f}s")

        # Send final response
        await manager.send_personal_message({
            'type': 'final_response',
            'data': final_response
        }, websocket)

        logger.info("✅ WebSocket query completed with real-time sequential updates")

    except Exception as e:
        logger.error(f"❌ WebSocket processing error: {e}")
        await manager.send_personal_message({
            'type': 'error',
            'message': f'Error: {str(e)}'
        }, websocket)

def start_web_app(host: str = "localhost", port: int = 8000):
    """Start the web application"""
    logger.info(f"🌐 Starting web server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_web_app()