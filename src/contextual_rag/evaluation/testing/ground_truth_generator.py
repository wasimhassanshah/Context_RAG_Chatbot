"""
Ground Truth Generator for RAG Evaluation
Location: src/contextual_rag/evaluation/testing/ground_truth_generator.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from contextual_rag.evaluation.testing.test_questions import get_test_questions
from contextual_rag.agents.retriever_agent import create_retriever_agent
from contextual_rag.agents.reranker_agent import create_reranker_agent
from contextual_rag.agents.generator_agent import create_generator_agent

class GroundTruthGenerator:
    """Generates ground truth answers using individual CrewAI agents (NO evaluator)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retriever_agent = None
        self.reranker_agent = None
        self.generator_agent = None
        self.ground_truth_file = Path(__file__).parent / "ground_truth_dataset.json"
    
    def initialize_rag_system(self):
        """Initialize individual RAG agents (excludes evaluator to avoid RAGAs)"""
        
        self.logger.info("🤖 Initializing individual RAG agents for ground truth generation...")
        
        try:
            # Create individual agents directly (NO master agent, NO evaluator)
            self.logger.info("📚 Initializing Retriever Agent...")
            self.retriever_agent = create_retriever_agent()
            
            self.logger.info("🔄 Initializing Reranker Agent...")
            self.reranker_agent = create_reranker_agent()
            
            self.logger.info("🤖 Initializing Generator Agent...")
            self.generator_agent = create_generator_agent()
            
            # Test individual agents
            retriever_status = self.retriever_agent.test_retrieval_system()
            reranker_status = self.reranker_agent.test_reranker_system()
            generator_status = self.generator_agent.test_generator_system()
            
            if (retriever_status.get('status') == 'success' and 
                reranker_status.get('status') == 'success' and 
                generator_status.get('status') == 'success'):
                
                self.logger.info("✅ All RAG agents initialized successfully!")
                self.logger.info(f"   • Retriever: {retriever_status.get('documents_retrieved', 0)} docs test")
                self.logger.info(f"   • Reranker: {reranker_status.get('reranking_working', False)}")
                self.logger.info(f"   • Generator: {generator_status.get('generation_working', False)}")
                return True
            else:
                raise Exception("One or more agents failed initialization")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RAG agents: {e}")
            raise
    
    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate ground truth answer for a single question using individual agents
        (completely bypasses evaluation to avoid RAGAs dependency)
        
        Args:
            question: Test question
        
        Returns:
            Dictionary with question, answer, and metadata
        """
        
        try:
            self.logger.info(f"🔍 Generating answer for: {question[:60]}...")
            start_time = datetime.now()
            
            # Step 1: Document Retrieval
            self.logger.info("📚 Step 1: Retrieving documents...")
            retrieved_docs = self.retriever_agent.retrieve_with_strategy(question)
            
            if not retrieved_docs:
                return {
                    "question": question,
                    "answer": "No relevant documents found for this question.",
                    "contexts": [],
                    "source_files": [],
                    "processing_time": 0,
                    "retrieval_count": 0,
                    "reranked_count": 0,
                    "error": "No documents retrieved",
                    "generation_timestamp": datetime.now().isoformat()
                }
            
            # Step 2: Document Reranking
            self.logger.info(f"🔄 Step 2: Reranking {len(retrieved_docs)} documents...")
            reranked_docs = self.reranker_agent.rerank_documents(
                question, retrieved_docs, top_k=3
            )
            
            if not reranked_docs:
                # Fallback to original docs
                reranked_docs = retrieved_docs[:3]
                self.logger.warning("⚠️ Reranking failed, using top 3 retrieved docs")
            
            # Step 3: Response Generation (NO EVALUATION)
            self.logger.info(f"🤖 Step 3: Generating response with {len(reranked_docs)} documents...")
            response_data = self.generator_agent.generate_response(
                question, reranked_docs, response_type="comprehensive"
            )
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Extract contexts from reranked documents
            contexts = []
            source_files = []
            
            for doc in reranked_docs:
                content = doc.get('content', '')
                if content:
                    contexts.append(content)
                
                filename = doc.get('filename', 'unknown')
                if filename not in source_files:
                    source_files.append(filename)
            
            # Compile ground truth entry (clean, no evaluation data)
            return {
                "question": question,
                "answer": response_data.get('answer', ''),
                "contexts": contexts,
                "source_files": source_files,
                "processing_time": processing_time,
                "retrieval_count": len(retrieved_docs),
                "reranked_count": len(reranked_docs),
                "generation_method": response_data.get('generation_method', 'unknown'),
                "pipeline_stages": {
                    "retrieval": {"documents_found": len(retrieved_docs), "status": "success"},
                    "reranking": {"documents_reranked": len(reranked_docs), "status": "success"},
                    "generation": {"response_generated": bool(response_data.get('answer')), "status": "success"},
                    "evaluation": {"status": "skipped_for_ground_truth_generation"}
                },
                "generation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating answer: {e}")
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "contexts": [],
                "source_files": [],
                "processing_time": 0,
                "retrieval_count": 0,
                "reranked_count": 0,
                "error": str(e),
                "generation_timestamp": datetime.now().isoformat()
            }
    
    def generate_ground_truth_dataset(self, 
                                    questions: Optional[List[str]] = None, 
                                    question_set: str = "basic") -> Dict[str, Any]:
        """
        Generate ground truth dataset for evaluation
        
        Args:
            questions: Custom list of questions (optional)
            question_set: Which question set to use ("basic", "full", "extended")
        
        Returns:
            Ground truth dataset
        """
        
        if questions is None:
            if question_set == "basic":
                questions = get_test_questions()
            elif question_set == "full":
                questions = get_test_questions(full_set=True)
            elif question_set == "extended":
                questions = get_test_questions(extended=True)
            else:
                questions = get_test_questions()
        
        self.logger.info(f"📝 Generating ground truth for {len(questions)} questions...")
        
        # Initialize RAG system if not already done
        if self.retriever_agent is None:
            self.initialize_rag_system()
        
        ground_truth_data = []
        successful_generations = 0
        
        for i, question in enumerate(questions, 1):
            self.logger.info(f"\n📋 Processing question {i}/{len(questions)}")
            
            result = self.generate_answer(question)
            ground_truth_data.append(result)
            
            if not result.get('error'):
                successful_generations += 1
                answer_length = len(result.get('answer', ''))
                contexts_count = len(result.get('contexts', []))
                self.logger.info(f"✅ Answer generated: {answer_length} chars, {contexts_count} contexts")
            else:
                self.logger.warning(f"⚠️ Failed to generate answer: {result.get('error')}")
        
        # Create dataset with metadata
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_questions": len(questions),
                "successful_generations": successful_generations,
                "failed_generations": len(questions) - successful_generations,
                "question_set": question_set,
                "rag_system": "CrewAI_3_Agent_Pipeline_No_Evaluation",
                "models": {
                    "llm": "llama3.2:1b",
                    "embedding": "nomic-embed-text", 
                    "reranking": "all-minilm"
                },
                "description": "Ground truth generated using individual CrewAI agents (retriever, reranker, generator) without evaluation"
            },
            "data": ground_truth_data
        }
        
        self.logger.info(f"\n✅ Ground truth generation complete!")
        self.logger.info(f"📊 Successful: {successful_generations}/{len(questions)}")
        
        return dataset
    
    def save_ground_truth_dataset(self, dataset: Dict[str, Any]) -> bool:
        """
        Save ground truth dataset to JSON file
        
        Args:
            dataset: Ground truth dataset
        
        Returns:
            bool: Success status
        """
        
        try:
            with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Ground truth dataset saved to: {self.ground_truth_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save ground truth: {e}")
            return False
    
    def load_ground_truth_dataset(self) -> Optional[Dict[str, Any]]:
        """
        Load existing ground truth dataset
        
        Returns:
            Ground truth dataset or None if not found
        """
        
        try:
            if not self.ground_truth_file.exists():
                self.logger.info(f"📂 No existing ground truth found at: {self.ground_truth_file}")
                return None
            
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            questions_count = len(dataset.get('data', []))
            generated_at = dataset.get('metadata', {}).get('generated_at', 'unknown')
            
            self.logger.info(f"📂 Loaded ground truth: {questions_count} questions from {generated_at}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"❌ Error loading ground truth: {e}")
            return None
    
    def display_sample_qa(self, dataset: Dict[str, Any], num_samples: int = 2):
        """
        Display sample Q&A pairs from dataset
        
        Args:
            dataset: Ground truth dataset
            num_samples: Number of samples to display
        """
        
        data = dataset.get('data', [])
        metadata = dataset.get('metadata', {})
        
        print(f"\n📊 Ground Truth Dataset Sample")
        print("=" * 80)
        print(f"Generated: {metadata.get('generated_at', 'unknown')}")
        print(f"Total Questions: {metadata.get('total_questions', 0)}")
        print(f"Success Rate: {metadata.get('successful_generations', 0)}/{metadata.get('total_questions', 0)}")
        print(f"Models: {metadata.get('models', {})}")
        
        print(f"\n📋 Sample Q&A Pairs (showing {min(num_samples, len(data))}):")
        print("=" * 80)
        
        for i, item in enumerate(data[:num_samples], 1):
            print(f"\n🔍 Question {i}:")
            print(f"Q: {item.get('question', 'No question')}")
            print(f"\n💡 Answer:")
            answer = item.get('answer', 'No answer')
            print(f"A: {answer[:300]}{'...' if len(answer) > 300 else ''}")
            
            contexts = item.get('contexts', [])
            source_files = item.get('source_files', [])
            processing_time = item.get('processing_time', 0)
            
            print(f"\n📚 Context: {len(contexts)} chunks from {len(source_files)} files")
            if source_files:
                print(f"Files: {', '.join(source_files)}")
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            
            if item.get('error'):
                print(f"❌ Error: {item['error']}")
    
    def validate_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ground truth dataset quality
        
        Args:
            dataset: Ground truth dataset
        
        Returns:
            Validation results
        """
        
        data = dataset.get('data', [])
        metadata = dataset.get('metadata', {})
        
        validation_results = {
            'total_items': len(data),
            'valid_items': 0,
            'items_with_errors': 0,
            'items_without_contexts': 0,
            'average_answer_length': 0,
            'average_context_count': 0,
            'coverage_by_source': {}
        }
        
        total_answer_length = 0
        total_context_count = 0
        source_coverage = {}
        
        for item in data:
            # Check for errors
            if item.get('error'):
                validation_results['items_with_errors'] += 1
                continue
            
            # Check for valid content
            answer = item.get('answer', '')
            contexts = item.get('contexts', [])
            
            if answer and len(answer) > 10:  # Valid answer threshold
                validation_results['valid_items'] += 1
                total_answer_length += len(answer)
            
            if not contexts:
                validation_results['items_without_contexts'] += 1
            else:
                total_context_count += len(contexts)
            
            # Track source file coverage
            for source_file in item.get('source_files', []):
                source_coverage[source_file] = source_coverage.get(source_file, 0) + 1
        
        # Calculate averages
        if validation_results['valid_items'] > 0:
            validation_results['average_answer_length'] = total_answer_length / validation_results['valid_items']
        
        if len(data) > 0:
            validation_results['average_context_count'] = total_context_count / len(data)
        
        validation_results['coverage_by_source'] = source_coverage
        
        return validation_results
    
    def export_for_ragas_evaluation(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export dataset in format suitable for RAGAs evaluation
        
        Args:
            dataset: Ground truth dataset
        
        Returns:
            RAGAs-compatible dataset
        """
        
        data = dataset.get('data', [])
        
        ragas_data = {
            'questions': [],
            'answers': [],
            'contexts': [],
            'ground_truths': []
        }
        
        for item in data:
            if not item.get('error') and item.get('answer'):
                ragas_data['questions'].append(item.get('question', ''))
                ragas_data['answers'].append(item.get('answer', ''))
                ragas_data['contexts'].append(item.get('contexts', []))
                ragas_data['ground_truths'].append(item.get('answer', ''))  # Use generated answer as ground truth
        
        return ragas_data
    
    def create_evaluation_scenarios(self) -> List[Dict[str, Any]]:
        """
        Create different evaluation scenarios for testing
        
        Returns:
            List of evaluation scenarios
        """
        
        scenarios = [
            {
                'name': 'basic_qa_test',
                'description': 'Basic Q&A test with 2 questions',
                'question_set': 'basic',
                'questions': None,
                'expected_duration_minutes': 2
            },
            {
                'name': 'comprehensive_test',
                'description': 'Full test suite with 4 original questions',
                'question_set': 'full',
                'questions': None,
                'expected_duration_minutes': 5
            },
            {
                'name': 'extended_evaluation',
                'description': 'Extended evaluation with all question categories',
                'question_set': 'extended',
                'questions': None,
                'expected_duration_minutes': 15
            },
            {
                'name': 'procurement_focused',
                'description': 'Procurement-specific questions only',
                'question_set': 'custom',
                'questions': [
                    "What are the procurement approval requirements?",
                    "What is the procurement process workflow according to the standards?",
                    "How do procurement evaluation criteria work?",
                    "What are the vendor selection requirements?"
                ],
                'expected_duration_minutes': 4
            },
            {
                'name': 'security_focused',
                'description': 'Information security questions only',
                'question_set': 'custom',
                'questions': [
                    "What are the specific sub-controls listed under \"T5.2.3 USER SECURITY CREDENTIALS MANAGEMENT\"?",
                    "What are the information security guidelines for data protection?",
                    "What security controls are required for information systems?"
                ],
                'expected_duration_minutes': 3
            }
        ]
        
        return scenarios
    
    def run_evaluation_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Run a specific evaluation scenario
        
        Args:
            scenario_name: Name of the scenario to run
        
        Returns:
            Scenario results
        """
        
        scenarios = self.create_evaluation_scenarios()
        scenario = next((s for s in scenarios if s['name'] == scenario_name), None)
        
        if not scenario:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        self.logger.info(f"🎯 Running scenario: {scenario['name']}")
        self.logger.info(f"📝 Description: {scenario['description']}")
        
        # Generate ground truth for scenario
        dataset = self.generate_ground_truth_dataset(
            questions=scenario.get('questions'),
            question_set=scenario.get('question_set', 'basic')
        )
        
        # Validate results
        validation = self.validate_dataset(dataset)
        
        return {
            'scenario': scenario,
            'dataset': dataset,
            'validation': validation,
            'success_rate': validation['valid_items'] / validation['total_items'] if validation['total_items'] > 0 else 0
        }

def create_ground_truth_generator() -> GroundTruthGenerator:
    """Factory function to create ground truth generator"""
    return GroundTruthGenerator()

if __name__ == "__main__":
    # Test the ground truth generator
    generator = create_ground_truth_generator()
    
    print("🧪 Testing Ground Truth Generator")
    print("=" * 60)
    
    try:
        # Initialize system (individual agents only)
        generator.initialize_rag_system()
        
        # Run basic scenario
        print("\n🎯 Running basic evaluation scenario...")
        result = generator.run_evaluation_scenario('basic_qa_test')
        
        print(f"✅ Scenario completed!")
        print(f"Success rate: {result['success_rate']:.1%}")
        
        # Display sample
        generator.display_sample_qa(result['dataset'], num_samples=1)
        
        # Save dataset
        if generator.save_ground_truth_dataset(result['dataset']):
            print(f"\n💾 Dataset saved successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")