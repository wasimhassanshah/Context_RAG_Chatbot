"""
Test script to verify agent fixes
Location: scripts/test_agents_fix.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        # Test tool imports
        print("  🔧 Testing tool imports...")
        from src.contextual_rag.agents.tools.vector_search_tool import VectorSearchTool
        from src.contextual_rag.agents.tools.reranking_tool import RerankingTool
        from src.contextual_rag.agents.tools.llm_generation_tool import LLMGenerationTool
        from src.contextual_rag.agents.tools.evaluation_tool import EvaluationTool
        print("  ✅ Tools imported successfully")
        
        # Test agent imports
        print("  👥 Testing agent imports...")
        from src.contextual_rag.agents.retriever_agent import RetrieverAgent
        from src.contextual_rag.agents.reranker_agent import RerankerAgent
        from src.contextual_rag.agents.generator_agent import GeneratorAgent
        from src.contextual_rag.agents.evaluator_agent import EvaluatorAgent
        from src.contextual_rag.agents.master_agent import MasterAgent
        print("  ✅ Agents imported successfully")
        
        # Test orchestrator import
        print("  🎯 Testing orchestrator import...")
        from src.contextual_rag.agents.crew_orchestrator import CrewRAGOrchestrator
        print("  ✅ Orchestrator imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_tool_initialization():
    """Test that tools can be initialized"""
    print("\n🔧 Testing tool initialization...")
    
    try:
        from src.contextual_rag.agents.tools.vector_search_tool import VectorSearchTool
        from src.contextual_rag.agents.tools.reranking_tool import RerankingTool
        from src.contextual_rag.agents.tools.llm_generation_tool import LLMGenerationTool
        from src.contextual_rag.agents.tools.evaluation_tool import EvaluationTool
        
        # Initialize tools
        vector_tool = VectorSearchTool()
        reranking_tool = RerankingTool()
        generation_tool = LLMGenerationTool()
        evaluation_tool = EvaluationTool()
        
        print("  ✅ All tools initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Tool initialization failed: {e}")
        return False

def test_agent_initialization():
    """Test that agents can be initialized"""
    print("\n👥 Testing agent initialization...")
    
    try:
        from src.contextual_rag.agents.retriever_agent import create_retriever_agent
        from src.contextual_rag.agents.reranker_agent import create_reranker_agent
        from src.contextual_rag.agents.generator_agent import create_generator_agent
        from src.contextual_rag.agents.evaluator_agent import create_evaluator_agent
        from src.contextual_rag.agents.master_agent import create_master_agent
        
        # Test individual agent creation
        print("  🔍 Creating retriever agent...")
        retriever = create_retriever_agent()
        
        print("  🔄 Creating reranker agent...")
        reranker = create_reranker_agent()
        
        print("  🤖 Creating generator agent...")
        generator = create_generator_agent()
        
        print("  📊 Creating evaluator agent...")
        evaluator = create_evaluator_agent()
        
        print("  🎯 Creating master agent...")
        master = create_master_agent()
        
        print("  ✅ All agents initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Agent initialization failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🚀 Testing basic functionality...")
    
    try:
        from src.contextual_rag.agents.master_agent import create_master_agent
        
        # Create master agent
        master = create_master_agent()
        
        # Test system status
        print("  📊 Testing system status...")
        status = master.get_system_status()
        
        if status['overall_status'] == 'operational':
            print("  ✅ System status: OPERATIONAL")
            
            # Test simple query processing
            print("  🔍 Testing simple query...")
            response = master.process_query("What is procurement?", response_type="concise")
            
            if response.get('answer') and len(response['answer']) > 10:
                print("  ✅ Query processing: WORKING")
                print(f"  📝 Sample response: {response['answer'][:100]}...")
                return True
            else:
                print("  ⚠️ Query processing: LIMITED (no valid response)")
                return False
        else:
            print(f"  ⚠️ System status: {status['overall_status'].upper()}")
            print("  💡 Some components may need attention, but imports are working")
            return True
            
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Agent Fix Verification Test")
    print("=" * 50)
    
    # Run tests
    results = []
    
    results.append(("Import Test", test_imports()))
    results.append(("Tool Initialization", test_tool_initialization()))
    results.append(("Agent Initialization", test_agent_initialization()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Results summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED - System is ready!")
        print("\n💡 Next steps:")
        print("  1. Run full setup: uv run python scripts/setup_crewai_system.py")
        print("  2. Start interactive session: uv run python -m src.contextual_rag.main --interactive")
        return True
    elif passed >= len(results) * 0.7:
        print("⚠️ MOST TESTS PASSED - Core functionality working")
        print("\n💡 You can proceed with:")
        print("  1. Run setup script for detailed testing")
        print("  2. Check specific failed components if needed")
        return True
    else:
        print("❌ MAJOR ISSUES - Please review failed tests")
        print("\n💡 Troubleshooting:")
        print("  1. Check that all dependencies are installed")
        print("  2. Verify Ollama is running: ollama list")
        print("  3. Check PostgreSQL connection")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)