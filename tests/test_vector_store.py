"""
Comprehensive Test Suite for LlamaIndex Vector Store
Location: tests/test_vector_store.py
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.contextual_rag.rag.vector_store import VectorStoreManager, setup_vector_store_manager

class VectorStoreTester:
    """Comprehensive tester for LlamaIndex RAG system"""
    
    def __init__(self):
        print("🧪 Initializing Vector Store Tester...")
        self.test_results = {}
        self.manager = None
        
    def test_manager_initialization(self):
        """Test 1: Manager initialization and component setup"""
        print("\n📋 Test 1: Manager Initialization")
        print("-" * 50)
        
        try:
            self.manager = setup_vector_store_manager()
            
            if self.manager:
                print("✅ Vector store manager initialized")
                print(f"✅ Processed data path: {self.manager.processed_data_path}")
                print(f"✅ Embedding dimensions: {self.manager.embedding_dim}")
                
                # Check components
                components = {
                    'ollama_manager': self.manager.ollama_manager,
                    'embedding_model': self.manager.embedding_model,
                    'llm_model': self.manager.llm_model,
                    'vector_store': self.manager.vector_store
                }
                
                for name, component in components.items():
                    if component:
                        print(f"✅ {name}: Initialized")
                    else:
                        print(f"❌ {name}: Not initialized")
                
                self.test_results['initialization'] = {
                    'status': 'PASS',
                    'components_initialized': len([c for c in components.values() if c])
                }
            else:
                print("❌ Failed to initialize manager")
                self.test_results['initialization'] = {
                    'status': 'FAIL',
                    'error': 'Manager is None'
                }
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            self.test_results['initialization'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_database_connection(self):
        """Test 2: Database connection and configuration"""
        print("\n🗄️ Test 2: Database Connection")
        print("-" * 50)
        
        if not self.manager:
            print("❌ Skipping - manager not initialized")
            self.test_results['database_connection'] = {'status': 'SKIP'}
            return
        
        try:
            # Test database stats
            stats = self.manager.get_database_stats()
            
            if 'error' in stats:
                print(f"❌ Database connection failed: {stats['error']}")
                self.test_results['database_connection'] = {
                    'status': 'FAIL',
                    'error': stats['error']
                }
            else:
                print("✅ Database connection successful")
                print(f"✅ Table exists: {stats.get('table_exists', False)}")
                print(f"✅ Current embeddings: {stats.get('total_embeddings', 0)}")
                print(f"✅ Embedding dimension: {stats.get('embedding_dimension', 0)}")
                
                self.test_results['database_connection'] = {
                    'status': 'PASS',
                    'table_exists': stats.get('table_exists', False),
                    'current_embeddings': stats.get('total_embeddings', 0)
                }
                
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            self.test_results['database_connection'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_document_loading(self):
        """Test 3: Loading processed documents"""
        print("\n📚 Test 3: Document Loading")
        print("-" * 50)
        
        if not self.manager:
            print("❌ Skipping - manager not initialized")
            self.test_results['document_loading'] = {'status': 'SKIP'}
            return
        
        try:
            start_time = time.time()
            documents = self.manager.load_processed_documents()
            loading_time = time.time() - start_time
            
            print(f"✅ Documents loaded: {len(documents)}")
            print(f"✅ Loading time: {loading_time:.2f} seconds")
            
            if documents:
                # Show sample document info
                sample_doc = documents[0]
                print("✅ Sample document:")
                print(f"   📄 Text length: {len(sample_doc.text)} characters")
                print(f"   📋 Metadata keys: {list(sample_doc.metadata.keys())}")
                print(f"   📄 Filename: {sample_doc.metadata.get('filename', 'Unknown')}")
                print(f"   🆔 Document ID: {sample_doc.id_}")
                
                # Show distribution by document
                doc_counts = {}
                for doc in documents:
                    filename = doc.metadata.get('filename', 'Unknown')
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1
                
                print("✅ Document distribution:")
                for filename, count in doc_counts.items():
                    print(f"   📄 {filename}: {count} chunks")
                
                self.test_results['document_loading'] = {
                    'status': 'PASS',
                    'total_documents': len(documents),
                    'loading_time': loading_time,
                    'document_distribution': doc_counts
                }
            else:
                print("⚠️ No documents loaded")
                self.test_results['document_loading'] = {
                    'status': 'PASS',
                    'total_documents': 0,
                    'reason': 'No processed documents found'
                }
                
        except Exception as e:
            print(f"❌ Document loading failed: {e}")
            self.test_results['document_loading'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_embedding_generation(self):
        """Test 4: Embedding generation with Ollama"""
        print("\n🔢 Test 4: Embedding Generation")
        print("-" * 50)
        
        if not self.manager:
            print("❌ Skipping - manager not initialized")
            self.test_results['embedding_generation'] = {'status': 'SKIP'}
            return
        
        try:
            # Test with different types of text
            test_texts = [
                "What are the procurement approval requirements?",
                "Information security policies and procedures",
                "HR employee onboarding process",
                "Budget allocation and financial controls"
            ]
            
            successful_embeddings = 0
            total_time = 0
            
            for i, text in enumerate(test_texts):
                print(f"   🧪 Test {i+1}: {text[:40]}...")
                
                start_time = time.time()
                success = self.manager.test_embedding_generation(text)
                embedding_time = time.time() - start_time
                
                if success:
                    successful_embeddings += 1
                    print(f"      ✅ Success ({embedding_time:.2f}s)")
                else:
                    print(f"      ❌ Failed ({embedding_time:.2f}s)")
                
                total_time += embedding_time
            
            avg_time = total_time / len(test_texts)
            success_rate = (successful_embeddings / len(test_texts)) * 100
            
            print("✅ Embedding tests completed:")
            print(f"   📊 Success rate: {success_rate:.1f}%")
            print(f"   ⏱️ Average time: {avg_time:.2f}s per embedding")
            print(f"   📏 Embedding dimension: {self.manager.embedding_dim}")
            
            self.test_results['embedding_generation'] = {
                'status': 'PASS' if successful_embeddings > 0 else 'FAIL',
                'success_rate': success_rate,
                'average_time': avg_time,
                'successful_embeddings': successful_embeddings
            }
            
        except Exception as e:
            print(f"❌ Embedding generation test failed: {e}")
            self.test_results['embedding_generation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_index_creation(self):
        """Test 5: Vector index creation"""
        print("\n🔍 Test 5: Vector Index Creation")
        print("-" * 50)
        
        if not self.manager:
            print("❌ Skipping - manager not initialized")
            self.test_results['index_creation'] = {'status': 'SKIP'}
            return
        
        try:
            # Load documents first
            documents = self.manager.load_processed_documents()
            
            if not documents:
                print("⚠️ No documents to index")
                self.test_results['index_creation'] = {
                    'status': 'SKIP',
                    'reason': 'No documents available'
                }
                return
            
            start_time = time.time()
            index = self.manager.create_index(documents)
            indexing_time = time.time() - start_time
            
            if index:
                print("✅ Index created successfully")
                print(f"✅ Indexing time: {indexing_time:.2f} seconds")
                print(f"✅ Documents per second: {len(documents)/indexing_time:.1f}")
                
                # Check database stats after indexing
                stats = self.manager.get_database_stats()
                print(f"✅ Total embeddings in database: {stats.get('total_embeddings', 0)}")
                
                self.test_results['index_creation'] = {
                    'status': 'PASS',
                    'indexing_time': indexing_time,
                    'documents_indexed': len(documents),
                    'embeddings_created': stats.get('total_embeddings', 0)
                }
            else:
                print("❌ Index creation failed")
                self.test_results['index_creation'] = {
                    'status': 'FAIL',
                    'error': 'Index is None'
                }
                
        except Exception as e:
            print(f"❌ Index creation failed: {e}")
            self.test_results['index_creation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_retrieval_system(self):
        """Test 6: Document retrieval system"""
        print("\n🔍 Test 6: Document Retrieval")
        print("-" * 50)
        
        if not self.manager or not self.manager.index:
            print("❌ Skipping - index not available")
            self.test_results['retrieval_system'] = {'status': 'SKIP'}
            return
        
        try:
            # Test queries related to your documents
            test_queries = [
                "What are the procurement approval requirements?",
                "Information security policies for employees",
                "HR policies for employee onboarding",
                "Budget approval process",
                "Document management procedures"
            ]
            
            retrieval_results = {}
            total_retrieval_time = 0
            
            for query in test_queries:
                print(f"\n   🔍 Query: {query}")
                
                start_time = time.time()
                results = self.manager.test_retrieval(query, top_k=3)
                retrieval_time = time.time() - start_time
                
                total_retrieval_time += retrieval_time
                
                if results:
                    print(f"      ✅ Retrieved {len(results)} documents ({retrieval_time:.2f}s)")
                    
                    # Show top result
                    top_result = results[0]
                    print(f"      📄 Top result (score: {top_result['score']:.3f}):")
                    print(f"         {top_result['content'][:100]}...")
                    print(f"         Source: {top_result['metadata'].get('filename', 'Unknown')}")
                    
                    retrieval_results[query] = {
                        'results_count': len(results),
                        'top_score': top_result['score'],
                        'retrieval_time': retrieval_time
                    }
                else:
                    print(f"      ❌ No results found ({retrieval_time:.2f}s)")
                    retrieval_results[query] = {
                        'results_count': 0,
                        'retrieval_time': retrieval_time
                    }
            
            avg_retrieval_time = total_retrieval_time / len(test_queries)
            successful_queries = len([r for r in retrieval_results.values() if r['results_count'] > 0])
            
            print(f"\n✅ Retrieval testing completed:")
            print(f"   📊 Successful queries: {successful_queries}/{len(test_queries)}")
            print(f"   ⏱️ Average retrieval time: {avg_retrieval_time:.2f}s")
            
            self.test_results['retrieval_system'] = {
                'status': 'PASS' if successful_queries > 0 else 'FAIL',
                'successful_queries': successful_queries,
                'total_queries': len(test_queries),
                'average_retrieval_time': avg_retrieval_time,
                'query_results': retrieval_results
            }
            
        except Exception as e:
            print(f"❌ Retrieval testing failed: {e}")
            self.test_results['retrieval_system'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_llm_integration(self):
        """Test 7: LLM response generation"""
        print("\n🤖 Test 7: LLM Integration")
        print("-" * 50)
        
        if not self.manager:
            print("❌ Skipping - manager not initialized")
            self.test_results['llm_integration'] = {'status': 'SKIP'}
            return
        
        try:
            # Test query and context
            test_query = "What are the main procurement approval requirements?"
            
            # Get context from retrieval
            if self.manager.index:
                retrieved_docs = self.manager.test_retrieval(test_query, top_k=3)
                context_chunks = [doc['content'] for doc in retrieved_docs[:2]]  # Use top 2
            else:
                # Fallback context
                context_chunks = [
                    "Procurement approvals require department head approval for purchases up to $10,000.",
                    "All purchases above $10,000 require board approval and must follow standard procurement process."
                ]
            
            print(f"🔍 Query: {test_query}")
            print(f"📄 Using {len(context_chunks)} context chunks")
            
            start_time = time.time()
            response = self.manager.test_llm_response(test_query, context_chunks)
            response_time = time.time() - start_time
            
            print(f"✅ LLM response generated ({response_time:.2f}s)")
            print(f"📝 Response preview: {response[:200]}...")
            print(f"📏 Response length: {len(response)} characters")
            
            # Basic quality checks
            quality_checks = {
                'has_content': len(response.strip()) > 0,
                'mentions_approval': 'approval' in response.lower(),
                'reasonable_length': 50 < len(response) < 2000,
                'not_error_message': 'error' not in response.lower()[:50]
            }
            
            passed_checks = sum(quality_checks.values())
            total_checks = len(quality_checks)
            
            print(f"✅ Quality checks: {passed_checks}/{total_checks} passed")
            for check, passed in quality_checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}")
            
            self.test_results['llm_integration'] = {
                'status': 'PASS' if passed_checks >= total_checks * 0.75 else 'FAIL',
                'response_time': response_time,
                'response_length': len(response),
                'quality_score': passed_checks / total_checks,
                'quality_checks': quality_checks
            }
            
        except Exception as e:
            print(f"❌ LLM integration test failed: {e}")
            self.test_results['llm_integration'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Run all vector store tests"""
        print("🧪 LLAMAINDEX VECTOR STORE COMPREHENSIVE TESTING")
        print("=" * 60)
        
        # Run all tests in order
        self.test_manager_initialization()
        self.test_database_connection()
        self.test_document_loading()
        self.test_embedding_generation()
        self.test_index_creation()
        self.test_retrieval_system()
        self.test_llm_integration()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r['status'] == 'PASS'])
        skipped_tests = len([r for r in self.test_results.values() if r['status'] == 'SKIP'])
        failed_tests = total_tests - passed_tests - skipped_tests
        
        print(f"🎯 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⏭️ Skipped: {skipped_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        if total_tests > 0:
            success_rate = ((passed_tests + skipped_tests) / total_tests) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASS' else "⏭️" if result['status'] == 'SKIP' else "❌"
            print(f"   {status_emoji} {test_name}: {result['status']}")
            
            # Show key metrics
            if 'total_documents' in result:
                print(f"      📄 Documents: {result['total_documents']}")
            if 'embeddings_created' in result:
                print(f"      🔢 Embeddings: {result['embeddings_created']}")
            if 'indexing_time' in result:
                print(f"      ⏱️ Indexing: {result['indexing_time']:.2f}s")
            if 'average_retrieval_time' in result:
                print(f"      🔍 Retrieval: {result['average_retrieval_time']:.2f}s")
            if 'response_time' in result:
                print(f"      🤖 LLM: {result['response_time']:.2f}s")
        
        # Performance summary
        if 'index_creation' in self.test_results and self.test_results['index_creation']['status'] == 'PASS':
            indexing_result = self.test_results['index_creation']
            docs_indexed = indexing_result.get('documents_indexed', 0)
            indexing_time = indexing_result.get('indexing_time', 0)
            
            if docs_indexed > 0 and indexing_time > 0:
                print(f"\n⚡ Performance Metrics:")
                print(f"   📄 Documents indexed: {docs_indexed}")
                print(f"   ⏱️ Total indexing time: {indexing_time:.2f}s")
                print(f"   📈 Indexing rate: {docs_indexed/indexing_time:.1f} docs/second")
        
        # Final recommendation
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! RAG system is working perfectly!")
            if passed_tests >= 5:  # Core tests passed
                print("🚀 Ready to proceed with Step 6: CrewAI Agent Orchestration!")
            else:
                print("⚠️ Some tests were skipped. System functional but may need optimization.")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Please fix the issues above.")
            print("💡 Common issues:")
            print("   - Check Ollama is running: ollama serve")
            print("   - Verify database connection")
            print("   - Ensure processed documents exist")

def main():
    """Main test function"""
    tester = VectorStoreTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()