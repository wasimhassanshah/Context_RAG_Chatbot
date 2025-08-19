"""
CrewAI System Setup and Test Script
Location: scripts/setup_crewai_system.py
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.contextual_rag.agents.crew_orchestrator import create_crew_orchestrator
from src.contextual_rag.agents.master_agent import create_master_agent
from src.contextual_rag.agents.retriever_agent import create_retriever_agent
from src.contextual_rag.agents.reranker_agent import create_reranker_agent
from src.contextual_rag.agents.generator_agent import create_generator_agent
from src.contextual_rag.agents.evaluator_agent import create_evaluator_agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_individual_agents():
    """Test each agent individually"""
    
    print("🧪 Testing Individual Agents")
    print("=" * 50)
    
    agents_tests = [
        ("Retriever Agent", create_retriever_agent, lambda a: a.test_retrieval_system()),
        ("Reranker Agent", create_reranker_agent, lambda a: a.test_reranker_system()),
        ("Generator Agent", create_generator_agent, lambda a: a.test_generator_system()),
        ("Evaluator Agent", create_evaluator_agent, lambda a: a.test_evaluator_system())
    ]
    
    test_results = {}
    
    for agent_name, create_func, test_func in agents_tests:
        print(f"\n🔍 Testing {agent_name}...")
        
        try:
            agent = create_func()
            result = test_func(agent)
            
            if result.get('status') == 'success':
                print(f"✅ {agent_name}: PASSED")
                test_results[agent_name] = 'PASSED'
            else:
                print(f"❌ {agent_name}: FAILED - {result.get('error', 'Unknown error')}")
                test_results[agent_name] = 'FAILED'
                
        except Exception as e:
            print(f"❌ {agent_name}: ERROR - {str(e)}")
            test_results[agent_name] = 'ERROR'
    
    return test_results

def test_master_agent():
    """Test master agent orchestration"""
    
    print("\n🎯 Testing Master Agent")
    print("=" * 30)
    
    try:
        master = create_master_agent()
        
        # Test system status
        status = master.get_system_status()
        print(f"📊 System Status: {status['overall_status']}")
        
        if status['overall_status'] == 'operational':
            # Test full pipeline
            print("🚀 Testing full RAG pipeline...")
            
            test_query = "What are the procurement approval requirements?"
            response = master.process_query(
                query=test_query,
                response_type="comprehensive"
            )
            
            print(f"✅ Pipeline test completed!")
            print(f"📝 Response length: {len(response.get('answer', ''))} characters")
            print(f"⏱️  Processing time: {response.get('processing_time_seconds', 0):.2f}s")
            print(f"📚 Sources used: {len(response.get('sources', []))}")
            
            eval_score = response.get('quality_metrics', {}).get('evaluation_score')
            if eval_score is not None:
                print(f"📊 Evaluation score: {eval_score:.3f}")
            
            return 'PASSED'
        else:
            print(f"❌ System not operational: {status}")
            return 'FAILED'
            
    except Exception as e:
        print(f"❌ Master agent test failed: {str(e)}")
        return 'ERROR'

def test_crewai_orchestrator():
    """Test CrewAI orchestrator"""
    
    print("\n🤖 Testing CrewAI Orchestrator")
    print("=" * 35)
    
    try:
        # Test with Phoenix disabled for setup
        orchestrator = create_crew_orchestrator(enable_phoenix=False)
        
        # Test single query
        test_query = "What are the information security policies?"
        response = orchestrator.process_query(
            query=test_query,
            response_type="concise"
        )
        
        if response.get('answer') and len(response['answer']) > 10:
            print("✅ CrewAI Orchestrator: PASSED")
            print(f"📝 Response: {response['answer'][:100]}...")
            return 'PASSED'
        else:
            print(f"❌ CrewAI Orchestrator: FAILED - No valid response")
            return 'FAILED'
            
    except Exception as e:
        print(f"❌ CrewAI Orchestrator test failed: {str(e)}")
        return 'ERROR'

def run_performance_benchmark():
    """Run performance benchmark"""
    
    print("\n⚡ Performance Benchmark")
    print("=" * 25)
    
    test_queries = [
        "What are the procurement approval requirements?",
        "What are the HR leave policies?", 
        "What are the information security guidelines?",
        "How do I submit a procurement request?",
        "What are the employee onboarding procedures?"
    ]
    
    try:
        orchestrator = create_crew_orchestrator(enable_phoenix=False)
        
        # Run batch evaluation
        results = orchestrator.batch_evaluate_performance(test_queries)
        
        batch_stats = results['batch_statistics']
        
        print(f"📊 Benchmark Results:")
        print(f"   • Total queries: {batch_stats['total_queries']}")
        print(f"   • Successful responses: {batch_stats['successful_responses']}")
        print(f"   • Average processing time: {batch_stats['average_processing_time']:.2f}s")
        print(f"   • Average evaluation score: {batch_stats['average_evaluation_score']:.3f}")
        
        score_dist = batch_stats['score_distribution']
        print(f"   • Score distribution:")
        print(f"     - Excellent (≥0.8): {score_dist['excellent']}")
        print(f"     - Good (0.6-0.8): {score_dist['good']}")
        print(f"     - Fair (0.4-0.6): {score_dist['fair']}")
        print(f"     - Poor (<0.4): {score_dist['poor']}")
        
        return 'PASSED' if batch_stats['successful_responses'] == batch_stats['total_queries'] else 'PARTIAL'
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {str(e)}")
        return 'ERROR'

def main():
    """Main setup and test function"""
    
    print("🚀 CrewAI RAG System Setup and Testing")
    print("=" * 60)
    print("This script will test the complete 4-agent CrewAI system:")
    print("   🔍 Retriever Agent - Document search")
    print("   🔄 Reranker Agent - BGE relevance optimization")
    print("   🤖 Generator Agent - Llama 3.2 response generation")
    print("   📊 Evaluator Agent - RAGAs quality assessment")
    print("   🎯 Master Agent - Pipeline orchestration")
    print("   🤖 CrewAI Orchestrator - Full system integration")
    
    print("\n" + "=" * 60)
    
    # Test results tracking
    all_results = {}
    
    # 1. Test individual agents
    print("Phase 1: Individual Agent Testing")
    agent_results = test_individual_agents()
    all_results.update(agent_results)
    
    # 2. Test master agent
    print("\nPhase 2: Master Agent Integration Testing")
    master_result = test_master_agent()
    all_results['Master Agent'] = master_result
    
    # 3. Test CrewAI orchestrator
    print("\nPhase 3: CrewAI Orchestrator Testing")
    crew_result = test_crewai_orchestrator()
    all_results['CrewAI Orchestrator'] = crew_result
    
    # 4. Performance benchmark
    print("\nPhase 4: Performance Benchmark")
    benchmark_result = run_performance_benchmark()
    all_results['Performance Benchmark'] = benchmark_result
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 SETUP AND TESTING COMPLETE")
    print("=" * 60)
    
    passed_tests = sum(1 for result in all_results.values() if result == 'PASSED')
    total_tests = len(all_results)
    
    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    print("\n📋 Detailed Results:")
    
    for test_name, result in all_results.items():
        if result == 'PASSED':
            icon = "✅"
        elif result == 'PARTIAL':
            icon = "⚠️"
        else:
            icon = "❌"
        
        print(f"   {icon} {test_name}: {result}")
    
    # System readiness assessment
    critical_components = ['Retriever Agent', 'Generator Agent', 'Master Agent']
    critical_passed = all(all_results.get(comp) == 'PASSED' for comp in critical_components)
    
    if critical_passed:
        print(f"\n🎯 SYSTEM STATUS: READY FOR PRODUCTION")
        print("✅ All critical components are operational")
        print("🚀 You can now run the interactive system:")
        print("   python -m src.contextual_rag.agents.crew_orchestrator")
    elif passed_tests >= total_tests * 0.7:
        print(f"\n⚠️  SYSTEM STATUS: READY WITH LIMITATIONS")
        print("✅ Core functionality available")
        print("⚠️  Some advanced features may be limited")
    else:
        print(f"\n❌ SYSTEM STATUS: NEEDS ATTENTION")
        print("❌ Critical issues need to be resolved")
        print("🔧 Please check the failed components before proceeding")
    
    # Next steps
    print(f"\n📋 Next Steps:")
    if critical_passed:
        print("1. Run interactive session: python scripts/setup_crewai_system.py --interactive")
        print("2. Enable Phoenix monitoring for production use")
        print("3. Configure custom evaluation metrics if needed")
        print("4. Set up automated monitoring and alerting")
    else:
        print("1. Review and fix failed agent tests")
        print("2. Check Ollama models are properly downloaded")
        print("3. Verify PostgreSQL database connectivity")
        print("4. Re-run setup script after fixes")
    
    return all_results

def interactive_demo():
    """Run interactive demo session"""
    
    print("🎮 Starting Interactive Demo Session")
    print("=" * 40)
    
    try:
        orchestrator = create_crew_orchestrator(enable_phoenix=True)
        orchestrator.interactive_session()
    except KeyboardInterrupt:
        print("\n👋 Demo session ended by user")
    except Exception as e:
        print(f"❌ Demo session failed: {e}")
    finally:
        print("🔄 Cleaning up...")

def run_specific_test(test_name: str):
    """Run a specific test by name"""
    
    test_functions = {
        'agents': test_individual_agents,
        'master': test_master_agent,
        'crew': test_crewai_orchestrator,
        'benchmark': run_performance_benchmark
    }
    
    if test_name in test_functions:
        print(f"🧪 Running {test_name} test...")
        result = test_functions[test_name]()
        print(f"📊 Test result: {result}")
        return result
    else:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {list(test_functions.keys())}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CrewAI RAG System Setup and Testing")
    parser.add_argument('--interactive', action='store_true', help='Run interactive demo session')
    parser.add_argument('--test', type=str, help='Run specific test (agents, master, crew, benchmark)')
    parser.add_argument('--phoenix', action='store_true', help='Enable Phoenix monitoring')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    elif args.test:
        run_specific_test(args.test)
    else:
        # Run full setup and testing
        results = main()
        
        # Exit with appropriate code
        passed_count = sum(1 for r in results.values() if r == 'PASSED')
        total_count = len(results)
        
        if passed_count == total_count:
            sys.exit(0)  # All tests passed
        elif passed_count >= total_count * 0.7:
            sys.exit(1)  # Most tests passed, but some issues
        else:
            sys.exit(2)  # Major issues