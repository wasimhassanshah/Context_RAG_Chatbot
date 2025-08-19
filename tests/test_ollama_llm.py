"""
Test Module for Ollama LLM (llama3.1:8b)
Comprehensive testing of your local language model
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.contextual_rag.agents.ollama_manager import OllamaManager

class OllamaLLMTester:
    """Comprehensive tester for your Ollama LLM"""
    
    def __init__(self):
        print("🚀 Initializing Ollama LLM Tester...")
        self.manager = OllamaManager()
        self.test_results = {}
    
    def test_basic_response(self):
        """Test basic text generation"""
        print("\n📝 Test 1: Basic Response Generation")
        print("-" * 50)
        
        prompt = "What is artificial intelligence? Give a brief answer."
        
        try:
            start_time = time.time()
            response = self.manager.generate_response(prompt)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            print(f"✅ Prompt: {prompt}")
            print(f"✅ Response: {response[:200]}...")
            print(f"✅ Response Time: {response_time:.2f} seconds")
            print(f"✅ Response Length: {len(response)} characters")
            
            self.test_results['basic_response'] = {
                'status': 'PASS',
                'response_time': response_time,
                'response_length': len(response)
            }
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.test_results['basic_response'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_complex_reasoning(self):
        """Test reasoning capabilities"""
        print("\n🧠 Test 2: Complex Reasoning")
        print("-" * 50)
        
        prompt = """
        You are helping with a document analysis system. 
        Given these document chunks about procurement policies:
        
        Chunk 1: "All purchases above $10,000 require board approval and must follow the standard procurement process."
        Chunk 2: "Emergency purchases can bypass normal approval if authorized by the CEO or CFO."
        Chunk 3: "Vendor selection must consider price, quality, and delivery timeline equally."
        
        Question: What approval is needed for a $15,000 emergency purchase?
        """
        
        try:
            start_time = time.time()
            response = self.manager.generate_response(prompt)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            print(f"✅ Complex Prompt: Multi-document reasoning test")
            print(f"✅ Response: {response}")
            print(f"✅ Response Time: {response_time:.2f} seconds")
            
            # Check if response mentions key concepts
            key_concepts = ['emergency', 'CEO', 'CFO', '$15,000', 'approval']
            concepts_found = [concept for concept in key_concepts if concept.lower() in response.lower()]
            
            print(f"✅ Key Concepts Found: {concepts_found}")
            
            self.test_results['complex_reasoning'] = {
                'status': 'PASS',
                'response_time': response_time,
                'concepts_found': len(concepts_found),
                'total_concepts': len(key_concepts)
            }
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.test_results['complex_reasoning'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_document_summarization(self):
        """Test document summarization"""
        print("\n📋 Test 3: Document Summarization")
        print("-" * 50)
        
        document_text = """
        Abu Dhabi Procurement Standards Manual
        
        Section 1: Introduction
        The Abu Dhabi government has established comprehensive procurement standards to ensure transparency, 
        efficiency, and value for money in all government purchases. These standards apply to all government 
        entities and must be followed for all procurement activities exceeding AED 50,000.
        
        Section 2: Procurement Process
        The procurement process consists of several key stages:
        1. Planning and budgeting
        2. Market research and vendor identification
        3. Request for proposal (RFP) preparation
        4. Vendor evaluation and selection
        5. Contract negotiation and award
        6. Contract management and monitoring
        
        Section 3: Approval Requirements
        All procurement must receive appropriate approvals based on the value:
        - Up to AED 100,000: Department head approval
        - AED 100,001 to AED 500,000: Director approval
        - Above AED 500,000: Board approval required
        
        Emergency procurements may bypass normal approval processes under specific circumstances.
        """
        
        prompt = f"Please summarize this procurement document in 3-4 key points:\n\n{document_text}"
        
        try:
            start_time = time.time()
            response = self.manager.generate_response(prompt)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            print(f"✅ Document Length: {len(document_text)} characters")
            print(f"✅ Summary: {response}")
            print(f"✅ Summary Time: {response_time:.2f} seconds")
            print(f"✅ Compression Ratio: {len(response)/len(document_text):.2f}")
            
            self.test_results['summarization'] = {
                'status': 'PASS',
                'response_time': response_time,
                'compression_ratio': len(response)/len(document_text)
            }
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.test_results['summarization'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_response_consistency(self):
        """Test response consistency"""
        print("\n🔄 Test 4: Response Consistency")
        print("-" * 50)
        
        prompt = "What is the capital of the United Arab Emirates?"
        
        responses = []
        response_times = []
        
        try:
            for i in range(3):
                print(f"   📝 Response {i+1}/3...")
                start_time = time.time()
                response = self.manager.generate_response(prompt)
                end_time = time.time()
                
                responses.append(response)
                response_times.append(end_time - start_time)
            
            print(f"✅ Response 1: {responses[0]}")
            print(f"✅ Response 2: {responses[1]}")
            print(f"✅ Response 3: {responses[2]}")
            print(f"✅ Average Response Time: {sum(response_times)/len(response_times):.2f} seconds")
            
            # Check if all responses mention Abu Dhabi
            consistent = all('abu dhabi' in resp.lower() for resp in responses)
            print(f"✅ Consistency Check: {'PASS' if consistent else 'FAIL'}")
            
            self.test_results['consistency'] = {
                'status': 'PASS' if consistent else 'FAIL',
                'avg_response_time': sum(response_times)/len(response_times),
                'consistent_answers': consistent
            }
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.test_results['consistency'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_temperature_settings(self):
        """Test different temperature settings"""
        print("\n🌡️ Test 5: Temperature Settings")
        print("-" * 50)
        
        prompt = "Generate a creative name for a procurement software system."
        
        try:
            # Test with low temperature (more focused)
            print("   🔹 Testing Low Temperature (0.1)...")
            self.manager.llm_model.temperature = 0.1
            response_low = self.manager.generate_response(prompt)
            
            # Test with high temperature (more creative)
            print("   🔹 Testing High Temperature (0.8)...")
            self.manager.llm_model.temperature = 0.8
            response_high = self.manager.generate_response(prompt)
            
            # Reset to default
            self.manager.llm_model.temperature = 0.1
            
            print(f"✅ Low Temp Response: {response_low}")
            print(f"✅ High Temp Response: {response_high}")
            print(f"✅ Responses Different: {response_low != response_high}")
            
            self.test_results['temperature'] = {
                'status': 'PASS',
                'responses_different': response_low != response_high
            }
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.test_results['temperature'] = {'status': 'FAIL', 'error': str(e)}
    
    def run_all_tests(self):
        """Run all LLM tests"""
        print("🧪 OLLAMA LLM COMPREHENSIVE TESTING")
        print("=" * 60)
        
        # Run all tests
        self.test_basic_response()
        self.test_complex_reasoning()
        self.test_document_summarization()
        self.test_response_consistency()
        self.test_temperature_settings()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r['status'] == 'PASS'])
        
        print(f"🎯 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASS' else "❌"
            print(f"   {status_emoji} {test_name}: {result['status']}")
            
            if 'response_time' in result:
                print(f"      ⏱️ Response Time: {result['response_time']:.2f}s")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Your LLM is working perfectly!")
            print("🚀 Ready to proceed with PostgreSQL setup!")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} tests failed. Check the errors above.")

def main():
    """Main test function"""
    tester = OllamaLLMTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()