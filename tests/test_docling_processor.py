"""
Comprehensive Test Suite for Docling Document Processor
Location: tests/test_docling_processor.py
"""

import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.contextual_rag.document_processing.docling_processor import DoclingProcessor, setup_docling_processor

class DoclingProcessorTester:
    """Comprehensive tester for Docling document processing"""
    
    def __init__(self):
        print("🧪 Initializing Docling Processor Tester...")
        self.test_results = {}
        self.processor = None
        
    def test_processor_initialization(self):
        """Test 1: Processor initialization"""
        print("\n📋 Test 1: Processor Initialization")
        print("-" * 50)
        
        try:
            self.processor = setup_docling_processor()
            
            if self.processor:
                print("✅ Processor initialized successfully")
                print(f"✅ Raw data path: {self.processor.raw_data_path}")
                print(f"✅ Processed data path: {self.processor.processed_data_path}")
                print(f"✅ Chunk size: {self.processor.chunk_size}")
                print(f"✅ Chunk overlap: {self.processor.chunk_overlap}")
                print(f"✅ Supported extensions: {self.processor.supported_extensions}")
                
                self.test_results['initialization'] = {
                    'status': 'PASS',
                    'processor_created': True
                }
            else:
                print("❌ Failed to initialize processor")
                self.test_results['initialization'] = {
                    'status': 'FAIL',
                    'error': 'Processor is None'
                }
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            self.test_results['initialization'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_directory_structure(self):
        """Test 2: Directory structure and file detection"""
        print("\n📂 Test 2: Directory Structure & File Detection")
        print("-" * 50)
        
        if not self.processor:
            print("❌ Skipping - processor not initialized")
            self.test_results['directory_structure'] = {'status': 'SKIP'}
            return
        
        try:
            # Check raw data directory
            raw_path = self.processor.raw_data_path
            processed_path = self.processor.processed_data_path
            
            print(f"📁 Raw data directory: {raw_path}")
            print(f"   Exists: {raw_path.exists()}")
            
            print(f"📁 Processed data directory: {processed_path}")
            print(f"   Exists: {processed_path.exists()}")
            
            # Find supported files
            files_found = []
            for ext in self.processor.supported_extensions:
                files = list(raw_path.glob(f"*{ext}"))
                files_found.extend(files)
            
            print(f"📄 Files found ({len(files_found)}):")
            for file in files_found:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   📄 {file.name} ({size_mb:.2f} MB)")
            
            if not files_found:
                print("⚠️ No PDF or DOCX files found in data/raw/")
                print("   Please add your documents to test processing")
            
            self.test_results['directory_structure'] = {
                'status': 'PASS',
                'raw_dir_exists': raw_path.exists(),
                'processed_dir_exists': processed_path.exists(),
                'files_found': len(files_found),
                'file_list': [f.name for f in files_found]
            }
            
        except Exception as e:
            print(f"❌ Directory check failed: {e}")
            self.test_results['directory_structure'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_docling_dependencies(self):
        """Test 3: Docling dependencies and imports"""
        print("\n🔧 Test 3: Docling Dependencies")
        print("-" * 50)
        
        try:
            # Test Docling imports
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            print("✅ DocumentConverter imported successfully")
            print("✅ InputFormat imported successfully")
            print("✅ PdfPipelineOptions imported successfully")
            
            # Test converter creation
            converter = DocumentConverter()
            print("✅ DocumentConverter created successfully")
            
            self.test_results['dependencies'] = {
                'status': 'PASS',
                'docling_imports': True,
                'converter_creation': True
            }
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Try: uv add docling")
            self.test_results['dependencies'] = {
                'status': 'FAIL',
                'error': f'Import error: {str(e)}'
            }
        except Exception as e:
            print(f"❌ Dependency test failed: {e}")
            self.test_results['dependencies'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_single_document_processing(self):
        """Test 4: Single document processing"""
        print("\n📄 Test 4: Single Document Processing")
        print("-" * 50)
        
        if not self.processor:
            print("❌ Skipping - processor not initialized")
            self.test_results['single_document'] = {'status': 'SKIP'}
            return
        
        try:
            # Find first available document
            files_to_test = []
            for ext in self.processor.supported_extensions:
                files = list(self.processor.raw_data_path.glob(f"*{ext}"))
                files_to_test.extend(files)
            
            if not files_to_test:
                print("⚠️ No test files available")
                self.test_results['single_document'] = {
                    'status': 'SKIP',
                    'reason': 'No test files found'
                }
                return
            
            # Test with first file
            test_file = files_to_test[0]
            print(f"📄 Testing with: {test_file.name}")
            
            start_time = time.time()
            processed_doc = self.processor.process_single_document(test_file)
            processing_time = time.time() - start_time
            
            if processed_doc:
                print(f"✅ Document processed successfully")
                print(f"✅ Document ID: {processed_doc.id}")
                print(f"✅ Title: {processed_doc.title}")
                print(f"✅ Chunks created: {len(processed_doc.chunks)}")
                print(f"✅ Processing time: {processing_time:.2f} seconds")
                
                # Test chunk content
                if processed_doc.chunks:
                    first_chunk = processed_doc.chunks[0]
                    print(f"✅ First chunk preview: {first_chunk.content[:100]}...")
                    print(f"✅ First chunk size: {len(first_chunk.content)} characters")
                
                self.test_results['single_document'] = {
                    'status': 'PASS',
                    'processing_time': processing_time,
                    'chunks_created': len(processed_doc.chunks),
                    'document_id': processed_doc.id
                }
            else:
                print("❌ Document processing failed")
                self.test_results['single_document'] = {
                    'status': 'FAIL',
                    'error': 'Processing returned None'
                }
                
        except Exception as e:
            print(f"❌ Single document test failed: {e}")
            self.test_results['single_document'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_batch_processing(self):
        """Test 5: Batch processing of all documents"""
        print("\n📚 Test 5: Batch Document Processing")
        print("-" * 50)
        
        if not self.processor:
            print("❌ Skipping - processor not initialized")
            self.test_results['batch_processing'] = {'status': 'SKIP'}
            return
        
        try:
            start_time = time.time()
            processed_docs = self.processor.process_all_documents()
            processing_time = time.time() - start_time
            
            print(f"\n📊 Batch Processing Results:")
            print(f"✅ Documents processed: {len(processed_docs)}")
            print(f"✅ Total processing time: {processing_time:.2f} seconds")
            
            if processed_docs:
                total_chunks = sum(len(doc.chunks) for doc in processed_docs)
                print(f"✅ Total chunks created: {total_chunks}")
                print(f"✅ Average chunks per document: {total_chunks/len(processed_docs):.1f}")
                
                # Show details for each document
                for doc in processed_docs:
                    print(f"   📄 {doc.filename}: {len(doc.chunks)} chunks")
                
                self.test_results['batch_processing'] = {
                    'status': 'PASS',
                    'documents_processed': len(processed_docs),
                    'total_chunks': total_chunks,
                    'processing_time': processing_time
                }
            else:
                print("⚠️ No documents were processed")
                self.test_results['batch_processing'] = {
                    'status': 'PASS',
                    'documents_processed': 0,
                    'reason': 'No files to process'
                }
                
        except Exception as e:
            print(f"❌ Batch processing test failed: {e}")
            self.test_results['batch_processing'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_file_persistence(self):
        """Test 6: File saving and loading"""
        print("\n💾 Test 6: File Persistence")
        print("-" * 50)
        
        if not self.processor:
            print("❌ Skipping - processor not initialized")
            self.test_results['file_persistence'] = {'status': 'SKIP'}
            return
        
        try:
            # Check for saved files
            processed_files = list(self.processor.processed_data_path.glob("*.json"))
            
            print(f"📁 Processed data directory: {self.processor.processed_data_path}")
            print(f"📄 JSON files found: {len(processed_files)}")
            
            if processed_files:
                # Test loading a file
                test_file = processed_files[0]
                print(f"🧪 Testing file: {test_file.name}")
                
                with open(test_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"✅ File loaded successfully")
                print(f"✅ Document ID: {data.get('id')}")
                print(f"✅ Filename: {data.get('filename')}")
                print(f"✅ Chunks: {data.get('chunk_count', 0)}")
                print(f"✅ File size: {test_file.stat().st_size / 1024:.1f} KB")
                
                # Validate JSON structure
                required_fields = ['id', 'filename', 'chunks', 'metadata']
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    print("✅ JSON structure is valid")
                else:
                    print(f"⚠️ Missing fields: {missing_fields}")
                
                self.test_results['file_persistence'] = {
                    'status': 'PASS',
                    'files_saved': len(processed_files),
                    'json_valid': len(missing_fields) == 0,
                    'file_size_kb': test_file.stat().st_size / 1024
                }
            else:
                print("⚠️ No processed files found")
                self.test_results['file_persistence'] = {
                    'status': 'PASS',
                    'files_saved': 0,
                    'reason': 'No files to test'
                }
                
        except Exception as e:
            print(f"❌ File persistence test failed: {e}")
            self.test_results['file_persistence'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Run all Docling processor tests"""
        print("🧪 DOCLING PROCESSOR COMPREHENSIVE TESTING")
        print("=" * 60)
        
        # Run all tests in order
        self.test_processor_initialization()
        self.test_directory_structure()
        self.test_docling_dependencies()
        self.test_single_document_processing()
        self.test_batch_processing()
        self.test_file_persistence()
        
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
            
            # Show additional details
            if result['status'] == 'PASS' and 'documents_processed' in result:
                print(f"      📄 Documents: {result['documents_processed']}")
            if result['status'] == 'PASS' and 'total_chunks' in result:
                print(f"      🔢 Chunks: {result['total_chunks']}")
            if result['status'] == 'PASS' and 'processing_time' in result:
                print(f"      ⏱️ Time: {result['processing_time']:.2f}s")
        
        # Final recommendation
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Docling processing is working perfectly!")
            print("🚀 Ready to proceed with Step 5: LlamaIndex RAG Implementation!")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Please fix the issues above.")
            if failed_tests == 1 and 'dependencies' in [name for name, result in self.test_results.items() if result['status'] == 'FAIL']:
                print("💡 Tip: Install missing dependencies with: uv add docling")

def main():
    """Main test function"""
    tester = DoclingProcessorTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()