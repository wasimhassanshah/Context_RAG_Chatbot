// Final RAG Chatbot Frontend JavaScript - No Answer Similarity & No System Status
class RAGChatbot {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.messageHistory = [];
        this.currentQuery = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
        // REMOVED: this.checkSystemStatus();
        
        this.agents = {
            retriever: { name: 'Document Retriever', status: 'idle', progress: 0 },
            reranker: { name: 'Document Reranker', status: 'idle', progress: 0 },
            generator: { name: 'Response Generator', status: 'idle', progress: 0 },
            evaluator: { name: 'Quality Evaluator', status: 'idle', progress: 0 }
        };
        
        this.renderAgentCards();
    }
    
    initializeElements() {
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.agentContainer = document.getElementById('agentContainer');
        this.ragasMetrics = document.getElementById('ragasMetrics');
        // REMOVED: this.systemStatusElement = document.getElementById('systemStatus');
        
        // Enable send button by default
        if (this.sendButton) {
            this.sendButton.disabled = false;
        }
    }
    
    setupEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Enter key in chat input
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize chat input
        this.chatInput.addEventListener('input', () => {
            this.chatInput.style.height = 'auto';
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log('🔌 Attempting WebSocket connection to:', wsUrl);
        
        try {
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected successfully');
                this.isConnected = true;
                this.updateConnectionStatus(true);
            };
            
            this.websocket.onmessage = (event) => {
                console.log('📨 WebSocket message received:', event.data);
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onclose = (event) => {
                console.log('🔌 WebSocket disconnected. Code:', event.code, 'Reason:', event.reason);
                this.isConnected = false;
                this.updateConnectionStatus(false);
                // Attempt reconnection after 3 seconds
                setTimeout(() => {
                    console.log('🔄 Attempting WebSocket reconnection...');
                    this.connectWebSocket();
                }, 3000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.isConnected = false;
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('❌ WebSocket connection failed:', error);
            this.isConnected = false;
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
            statusIndicator.style.background = connected ? '#28a745' : '#dc3545';
        }
        
        // Always enable send button - we have REST fallback
        if (this.sendButton) {
            this.sendButton.disabled = false;
        }
        
        console.log(`🔌 Connection status updated: ${connected ? 'Connected' : 'Disconnected'}`);
    }
    
    handleWebSocketMessage(data) {
        console.log('📨 Processing WebSocket message:', data);
        
        switch (data.type) {
            case 'connection':
                console.log('🔗 Connection confirmed:', data.message);
                break;
                
            case 'agent_update':
                this.updateAgent(data.agent, data.status, data.message);
                break;
                
            case 'agent_result':
                this.showAgentResult(data.agent, data.data);
                break;
                
            case 'final_response':
                this.handleFinalResponse(data.data);
                break;
                
            case 'error':
                this.showError(data.message);
                break;
                
            default:
                console.log('🤷‍♂️ Unknown message type:', data.type);
        }
    }
    
    // REMOVED: checkSystemStatus() and updateSystemStatus() methods
    
    sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        
        // Clear input
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';
        
        // Disable send button temporarily
        this.sendButton.disabled = true;
        this.currentQuery = message;
        
        // Reset agents
        this.resetAgents();
        
        // Check WebSocket connection
        const isWebSocketReady = this.websocket && 
                                this.websocket.readyState === WebSocket.OPEN && 
                                this.isConnected;
        
        if (isWebSocketReady) {
            console.log('📤 Sending via WebSocket...');
            try {
                this.websocket.send(JSON.stringify({
                    type: 'chat',
                    query: message
                }));
            } catch (error) {
                console.error('❌ WebSocket send failed:', error);
                console.log('📤 Falling back to REST API...');
                this.sendMessageViaRest(message);
            }
        } else {
            console.log('📤 WebSocket not ready, using REST API...');
            this.sendMessageViaRest(message);
        }
    }
    
    async sendMessageViaRest(message) {
        try {
            // Show processing status
            this.updateAgent('retriever', 'working', 'Searching documents...');
            this.updateAgent('reranker', 'working', 'Reranking documents...');
            this.updateAgent('generator', 'working', 'Generating response...');
            this.updateAgent('evaluator', 'working', 'Evaluating quality...');
            
            console.log('🔄 Sending REST request...');
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('✅ REST response received:', data);
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Simulate agent results with delays
            setTimeout(() => {
                this.showAgentResult('retriever', { 
                    documents_found: 10, 
                    status: 'completed' 
                });
            }, 500);
            
            setTimeout(() => {
                this.showAgentResult('reranker', { 
                    documents_reranked: 3, 
                    status: 'completed' 
                });
            }, 1000);
            
            setTimeout(() => {
                this.showAgentResult('generator', { 
                    response_length: data.answer?.length || 0, 
                    generation_method: 'llama3.2:1b',
                    status: 'completed' 
                });
            }, 1500);
            
            // Show evaluation results if available
            const quality_metrics = data.quality_metrics || {};
            if (quality_metrics.evaluation_score !== undefined) {
                // FIXED: Extract individual scores properly from REST response - NO ANSWER SIMILARITY
                const detailed_scores = quality_metrics.detailed_scores || {};
                const individual_metrics = quality_metrics.individual_metrics || {};
                
                console.log('🔍 REST - Detailed scores:', detailed_scores);
                console.log('🔍 REST - Individual metrics:', individual_metrics);
                
                // Use individual_metrics if available, otherwise detailed_scores - EXCLUDE answer_similarity
                const finalMetrics = Object.keys(individual_metrics).length > 0 ? {
                    answer_relevancy: individual_metrics.answer_relevancy || 0,
                    context_precision: individual_metrics.context_precision || 0,
                    context_recall: individual_metrics.context_recall || 0,
                    answer_correctness: individual_metrics.answer_correctness || 0
                } : {
                    answer_relevancy: detailed_scores.answer_relevancy || 0,
                    context_precision: detailed_scores.context_precision || 0,
                    context_recall: detailed_scores.context_recall || 0,
                    answer_correctness: detailed_scores.answer_correctness || 0
                };
                
                setTimeout(() => {
                    this.showAgentResult('evaluator', {
                        evaluation_score: quality_metrics.evaluation_score,
                        individual_metrics: finalMetrics,
                        evaluation_method: 'ragas',
                        has_ground_truth: quality_metrics.has_ground_truth || false,
                        status: 'completed'
                    });
                }, 2000);
            }
            
            // Add assistant response
            this.addMessage('assistant', data.answer, {
                sources: data.sources || [],
                evaluation_score: quality_metrics.evaluation_score,
                processing_time: data.processing_time_seconds || 0  // FIXED: Use actual processing time
            });
            
            // Mark all agents as completed
            Object.keys(this.agents).forEach(agentKey => {
                this.agents[agentKey].status = 'completed';
                this.agents[agentKey].progress = 100;
            });
            this.renderAgentCards();
            
        } catch (error) {
            console.error('❌ REST API error:', error);
            this.showError(`Failed to process query: ${error.message}`);
        } finally {
            this.sendButton.disabled = false;
            this.currentQuery = null;
        }
    }
    
    addMessage(sender, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        let sourcesHtml = '';
        if (metadata.sources && metadata.sources.length > 0) {
            sourcesHtml = `
                <div class="message-sources">
                    <strong>Sources:</strong>
                    ${metadata.sources.map(source => 
                        `<span class="source-tag">${source.filename || source.source || 'Unknown'}</span>`
                    ).join('')}
                </div>
            `;
        }
        
        let qualityHtml = '';
        if (metadata.evaluation_score !== undefined && metadata.evaluation_score !== null) {
            const scoreClass = metadata.evaluation_score >= 0.8 ? 'score-excellent' :
                              metadata.evaluation_score >= 0.6 ? 'score-good' :
                              metadata.evaluation_score >= 0.4 ? 'score-fair' : 'score-poor';
            
            qualityHtml = `
                <div class="message-sources">
                    <strong>Quality Score:</strong>
                    <span class="${scoreClass}">${metadata.evaluation_score.toFixed(3)}</span>
                </div>
            `;
        }
        
        let processingTimeHtml = '';
        if (metadata.processing_time !== undefined) {
            processingTimeHtml = `
                <div class="message-sources">
                    <strong>Processing Time:</strong>
                    <span>${metadata.processing_time.toFixed(1)}s</span>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-content">${content}</div>
            <div class="message-timestamp">${timestamp}</div>
            ${sourcesHtml}
            ${qualityHtml}
            ${processingTimeHtml}
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        // Store in history
        this.messageHistory.push({
            sender,
            content,
            timestamp,
            metadata
        });
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            background: #fee;
            border: 1px solid #fcc;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            color: #c00;
        `;
        errorDiv.textContent = message;
        
        this.chatMessages.appendChild(errorDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        // Re-enable send button
        this.sendButton.disabled = false;
    }
    
    resetAgents() {
        Object.keys(this.agents).forEach(agentKey => {
            this.agents[agentKey].status = 'idle';
            this.agents[agentKey].progress = 0;
            this.agents[agentKey].result = null;
        });
        this.renderAgentCards();
        this.clearRAGAsMetrics();
    }
    
    updateAgent(agentName, status, message) {
        if (this.agents[agentName]) {
            this.agents[agentName].status = status;
            this.agents[agentName].message = message;
            
            if (status === 'working') {
                this.agents[agentName].progress = 50;
            } else if (status === 'completed') {
                this.agents[agentName].progress = 100;
            }
            
            this.renderAgentCards();
        }
    }
    
    showAgentResult(agentName, data) {
        if (this.agents[agentName]) {
            this.agents[agentName].status = 'completed';
            this.agents[agentName].progress = 100;
            this.agents[agentName].result = data;
            
            this.renderAgentCards();
            
            // FIXED: Show individual RAGAs metrics for evaluator - NO ANSWER SIMILARITY
            if (agentName === 'evaluator' && data && data.individual_metrics) {
                console.log('📊 Displaying individual RAGAs metrics:', data.individual_metrics);
                this.showIndividualRAGAsMetrics(data);
            }
        }
    }
    
    showIndividualRAGAsMetrics(evaluationData) {
        if (!this.ragasMetrics) {
            console.log('❌ RAGAs metrics container not found');
            return;
        }
        
        console.log('📊 Showing individual metrics:', evaluationData);
        
        const overallScore = evaluationData.evaluation_score || 0;
        const individualMetrics = evaluationData.individual_metrics || {};
        
        let metricsHtml = `
            <div class="ragas-header">📊 RAGAs Individual Metrics</div>
        `;
        
        // Overall score
        const overallClass = this.getScoreClass(overallScore);
        metricsHtml += `
            <div class="metric-item">
                <span class="metric-name"><strong>Overall Score</strong></span>
                <span class="metric-score ${overallClass}">${overallScore.toFixed(3)}</span>
            </div>
        `;
        
        // Individual metrics - REMOVED answer_similarity
        const metricNames = {
            'answer_relevancy': 'Answer Relevancy',
            'context_precision': 'Context Precision',
            'context_recall': 'Context Recall',
            'answer_correctness': 'Answer Correctness'
        };
        
        Object.entries(metricNames).forEach(([key, displayName]) => {
            const score = individualMetrics[key];
            if (score !== undefined && score !== null && !isNaN(score)) {
                const scoreClass = this.getScoreClass(score);
                metricsHtml += `
                    <div class="metric-item">
                        <span class="metric-name">${displayName}</span>
                        <span class="metric-score ${scoreClass}">${score.toFixed(3)}</span>
                    </div>
                `;
            } else {
                // Show 0.000 if metric is missing or invalid
                metricsHtml += `
                    <div class="metric-item">
                        <span class="metric-name">${displayName}</span>
                        <span class="metric-score score-poor">0.000</span>
                    </div>
                `;
            }
        });
        
        // Additional info
        if (evaluationData.has_ground_truth) {
            metricsHtml += `
                <div class="metric-item">
                    <span class="metric-name">Ground Truth</span>
                    <span class="metric-score score-excellent">✅ Found</span>
                </div>
            `;
        } else {
            metricsHtml += `
                <div class="metric-item">
                    <span class="metric-name">Ground Truth</span>
                    <span class="metric-score score-poor">❌ Not Found</span>
                </div>
            `;
        }
        
        metricsHtml += `
            <div class="metric-item">
                <span class="metric-name">Method</span>
                <span class="metric-score">${(evaluationData.evaluation_method || 'ragas').toUpperCase()}</span>
            </div>
        `;
        
        this.ragasMetrics.innerHTML = metricsHtml;
        console.log('✅ Individual RAGAs metrics displayed successfully');
    }
    
    getScoreClass(score) {
        if (isNaN(score) || score === null || score === undefined) return 'score-poor';
        if (score >= 0.8) return 'score-excellent';
        if (score >= 0.6) return 'score-good'; 
        if (score >= 0.4) return 'score-fair';
        return 'score-poor';
    }
    
    handleFinalResponse(responseData) {
        console.log('🎯 Final response received:', responseData);
        
        // Extract individual metrics from the response - EXCLUDE answer_similarity
        const qualityMetrics = responseData.quality_metrics || {};
        const pipelineStages = responseData.pipeline_stages || {};
        const evaluationStage = pipelineStages.evaluation || {};
        
        // Try multiple sources for individual metrics
        let individualMetrics = qualityMetrics.individual_metrics || 
                               evaluationStage.individual_metrics || 
                               {};
        
        // If no individual metrics, try to extract from detailed_scores - EXCLUDE answer_similarity
        if (Object.keys(individualMetrics).length === 0) {
            const detailed_scores = qualityMetrics.detailed_scores || 
                                  evaluationStage.detailed_scores || 
                                  {};
            
            individualMetrics = {
                answer_relevancy: detailed_scores.answer_relevancy || 0,
                context_precision: detailed_scores.context_precision || 0,
                context_recall: detailed_scores.context_recall || 0,
                answer_correctness: detailed_scores.answer_correctness || 0
            };
        }
        
        // Remove answer_similarity if it exists
        delete individualMetrics.answer_similarity;
        
        console.log('🔍 Final individual metrics (no answer_similarity):', individualMetrics);
        
        // Add assistant message
        this.addMessage('assistant', responseData.answer, {
            sources: responseData.sources || [],
            evaluation_score: qualityMetrics.evaluation_score,
            processing_time: responseData.processing_time_seconds,
            individual_metrics: individualMetrics
        });
        
        // Mark all agents as completed
        Object.keys(this.agents).forEach(agentKey => {
            this.agents[agentKey].status = 'completed';
            this.agents[agentKey].progress = 100;
        });
        this.renderAgentCards();
        
        // Show individual RAGAs metrics if available
        if (Object.keys(individualMetrics).length > 0) {
            this.showIndividualRAGAsMetrics({
                evaluation_score: qualityMetrics.evaluation_score || 0,
                individual_metrics: individualMetrics,
                has_ground_truth: qualityMetrics.has_ground_truth || false,
                evaluation_method: 'ragas'
            });
        }
        
        // Re-enable send button
        this.sendButton.disabled = false;
        this.currentQuery = null;
    }
    
    renderAgentCards() {
        if (!this.agentContainer) return;
        
        this.agentContainer.innerHTML = '';
        
        Object.entries(this.agents).forEach(([key, agent]) => {
            const agentCard = document.createElement('div');
            agentCard.className = `agent-card ${agent.status}`;
            
            let resultHtml = '';
            if (agent.result) {
                const result = agent.result;
                if (key === 'retriever') {
                    resultHtml = `
                        <div class="agent-result">
                            📄 Documents found: ${result.documents_found || 0}
                        </div>
                    `;
                } else if (key === 'reranker') {
                    resultHtml = `
                        <div class="agent-result">
                            📄 Documents reranked: ${result.documents_reranked || 0}
                        </div>
                    `;
                } else if (key === 'generator') {
                    resultHtml = `
                        <div class="agent-result">
                            📝 Response length: ${result.response_length || 0} chars<br>
                            🤖 Method: ${result.generation_method || 'unknown'}
                        </div>
                    `;
                } else if (key === 'evaluator') {
                    resultHtml = `
                        <div class="agent-result">
                            📊 Score: ${(result.evaluation_score || 0).toFixed(3)}<br>
                            🎯 Method: ${result.evaluation_method || 'unknown'}
                        </div>
                    `;
                }
            }
            
            agentCard.innerHTML = `
                <div class="agent-header">
                    <div class="agent-name">${agent.name}</div>
                    <div class="agent-status ${agent.status}">${agent.status}</div>
                </div>
                <div class="agent-progress">
                    <div class="agent-progress-bar" style="width: ${agent.progress}%"></div>
                </div>
                <div class="agent-details">
                    ${agent.message || 'Ready'}
                </div>
                ${resultHtml}
            `;
            
            this.agentContainer.appendChild(agentCard);
        });
    }
    
    clearRAGAsMetrics() {
        if (this.ragasMetrics) {
            this.ragasMetrics.innerHTML = `
                <div class="ragas-header">📊 RAGAs Evaluation</div>
                <div class="loading">
                    <span>Awaiting evaluation...</span>
                </div>
            `;
        }
    }
    
    // Public API methods
    sendTestQuery() {
        const testQueries = [
            "What are the procurement approval requirements?",
            "What are the information security policies?",
            "What are the HR leave policies?",
            "How does the procurement process work?",
            "What are the data protection guidelines?",
            "What is the employee onboarding process?"
        ];
        
        const randomQuery = testQueries[Math.floor(Math.random() * testQueries.length)];
        this.chatInput.value = randomQuery;
        this.sendMessage();
    }
    
    clearChat() {
        this.chatMessages.innerHTML = '';
        this.messageHistory = [];
        this.resetAgents();
        
        // Add welcome message back
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message assistant';
        welcomeDiv.innerHTML = `
            <div class="message-content">
                👋 Welcome to the Contextual RAG Chatbot! 
                <br><br>
                I can help you with questions about:
                <br>• 📄 Procurement policies and procedures
                <br>• 🛡️ Information security guidelines  
                <br>• 👥 HR policies and bylaws
                <br>• 📋 Government standards and regulations
                <br><br>
                Ask me anything about your organizational documents!
            </div>
            <div class="message-timestamp">System</div>
        `;
        this.chatMessages.appendChild(welcomeDiv);
    }
    
    exportChatHistory() {
        const data = {
            timestamp: new Date().toISOString(),
            messages: this.messageHistory,
            system_info: {
                user_agent: navigator.userAgent,
                url: window.location.href,
                session_duration: Date.now() - this.sessionStartTime
            }
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `rag-chat-history-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('📥 Chat history exported');
    }
    
    // Initialize session start time
    sessionStartTime = Date.now();
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing RAG Chatbot...');
    window.ragChatbot = new RAGChatbot();
    console.log('✅ RAG Chatbot initialized');
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+Enter to send test query
        if (e.ctrlKey && e.key === 'Enter') {
            window.ragChatbot.sendTestQuery();
        }
        // Ctrl+L to clear chat
        if (e.ctrlKey && e.key === 'l') {
            e.preventDefault();
            window.ragChatbot.clearChat();
        }
    });
});

// Add some global helper functions for debugging
window.testQuery = () => window.ragChatbot?.sendTestQuery();
window.clearChat = () => window.ragChatbot?.clearChat();
window.exportHistory = () => window.ragChatbot?.exportChatHistory();

// Error handling for uncaught errors
window.addEventListener('error', (e) => {
    console.error('🚨 JavaScript Error:', e.error);
    
    if (window.ragChatbot) {
        window.ragChatbot.showError('A JavaScript error occurred. Please refresh the page if issues persist.');
    }
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && window.ragChatbot) {
        // Reconnect WebSocket if needed when page becomes visible
        if (!window.ragChatbot.isConnected) {
            console.log('🔄 Page visible, attempting WebSocket reconnection...');
            window.ragChatbot.connectWebSocket();
        }
    }
});