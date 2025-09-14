"""
Test script for document re-ranking functionality
Tests the re-ranking module independently and integrated with RAG system
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_reranker import DocumentReranker, create_reranker_from_env
from src.nvidia_embeddings import NVIDIAEmbeddings
from src.vector_database import VectorDatabase
from src.rag_agent import RAGAgent
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_documents():
    """Create sample documents for testing"""
    test_docs = [
        Document(
            page_content="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            metadata={"source": "test_doc_1.pdf", "page": 1}
        ),
        Document(
            page_content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            metadata={"source": "test_doc_2.pdf", "page": 1}
        ),
        Document(
            page_content="Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            metadata={"source": "test_doc_3.pdf", "page": 1}
        ),
        Document(
            page_content="Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.",
            metadata={"source": "test_doc_4.pdf", "page": 1}
        ),
        Document(
            page_content="Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data, and apply knowledge and actionable insights from data across a broad range of application domains.",
            metadata={"source": "test_doc_5.pdf", "page": 1}
        ),
        Document(
            page_content="Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
            metadata={"source": "test_doc_6.pdf", "page": 1}
        ),
        Document(
            page_content="Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. In applying statistics to a scientific, industrial, or social problem, it is conventional to begin with a statistical population or a statistical model to be studied.",
            metadata={"source": "test_doc_7.pdf", "page": 1}
        ),
        Document(
            page_content="Algorithms are step-by-step procedures for calculations, data processing, and automated reasoning tasks. An algorithm is a finite sequence of well-defined computer-implementable instructions, typically to solve a class of problems or to perform a computation.",
            metadata={"source": "test_doc_8.pdf", "page": 1}
        )
    ]
    return test_docs


def test_reranker_initialization():
    """Test re-ranker initialization"""
    print("🧪 Testing Re-ranker Initialization...")
    
    try:
        # Test with default settings
        reranker = DocumentReranker()
        print(f"✅ Default re-ranker initialized: Available={reranker.is_available()}")
        
        # Test with environment variables
        reranker_env = create_reranker_from_env()
        print(f"✅ Environment re-ranker initialized: Available={reranker_env.is_available()}")
        
        # Print model info
        print(f"Model info: {reranker.get_model_info()}")
        
        return reranker
        
    except Exception as e:
        print(f"❌ Re-ranker initialization failed: {str(e)}")
        return None


def test_standalone_reranking():
    """Test re-ranking functionality independently"""
    print("\n🧪 Testing Standalone Re-ranking...")
    
    reranker = test_reranker_initialization()
    if not reranker or not reranker.is_available():
        print("❌ Re-ranker not available, skipping standalone test")
        return False
    
    try:
        # Create test documents
        test_docs = create_test_documents()
        
        # Test queries with different characteristics
        test_queries = [
            "What is machine learning?",  # Should rank ML doc highly
            "deep learning neural networks",  # Should rank deep learning doc highly
            "programming language Python",  # Should rank Python doc highly
            "statistical analysis data"  # Should rank statistics doc highly
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            
            start_time = time.time()
            reranked_docs, scores = reranker.rerank_documents(
                query=query,
                documents=test_docs,
                top_k=3,
                initial_k=len(test_docs)
            )
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing time: {processing_time:.3f} seconds")
            print(f"📊 Top 3 results:")
            
            for i, (doc, score) in enumerate(zip(reranked_docs, scores), 1):
                preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                print(f"  {i}. Score: {score:.4f} | {preview}")
        
        # Test benchmark functionality
        print(f"\n📈 Benchmarking re-ranking performance:")
        benchmark_results = reranker.benchmark_reranking(
            query="machine learning algorithms",
            documents=test_docs,
            top_k_values=[3, 5, 8]
        )
        
        for key, results in benchmark_results.items():
            if isinstance(results, dict):
                print(f"  {key}: {results['processing_time']:.3f}s | Avg score: {results['avg_score']:.4f}")
        
        print("✅ Standalone re-ranking test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Standalone re-ranking test failed: {str(e)}")
        return False


def test_integrated_reranking():
    """Test re-ranking integrated with RAG system"""
    print("\n🧪 Testing Integrated Re-ranking with RAG System...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize RAG agent with re-ranking enabled
        rag_agent = RAGAgent(
            docs_folder="Data/Docs",
            api_key=api_key,
            enable_reranking=True,
            reranking_top_k=3,
            initial_retrieval_k=10
        )
        
        print(f"✅ RAG Agent initialized")
        print(f"   - Hybrid search enabled: {rag_agent.is_hybrid_search_enabled()}")
        print(f"   - Re-ranking enabled: {rag_agent.is_reranking_enabled()}")
        
        # Check if knowledge base exists
        stats = rag_agent.get_knowledge_base_stats()
        if not stats.get("index_exists", False):
            print("📝 Knowledge base not found, setting up...")
            if not rag_agent.setup_knowledge_base():
                print("❌ Failed to setup knowledge base")
                return False
        else:
            print(f"✅ Knowledge base loaded: {stats.get('document_count', 0)} documents")
        
        # Test questions to evaluate re-ranking impact
        test_questions = [
            "What is machine learning?",
            "How do neural networks work?",
            "What are the main concepts in data science?",
            "Explain artificial intelligence algorithms"
        ]
        
        for question in test_questions:
            print(f"\n📝 Question: '{question}'")
            
            # Test with re-ranking enabled
            start_time = time.time()
            response_with_rerank = rag_agent.ask_question(question, k=3)
            time_with_rerank = time.time() - start_time
            
            # Test without re-ranking (by getting documents directly)
            start_time = time.time()
            docs_without_rerank = rag_agent.get_relevant_documents(
                query=question, 
                k=3, 
                use_reranking=False
            )
            time_without_rerank = time.time() - start_time
            
            print(f"⏱️  Time with re-ranking: {time_with_rerank:.3f}s")
            print(f"⏱️  Time without re-ranking: {time_without_rerank:.3f}s")
            print(f"📊 Retrieved {len(response_with_rerank.source_documents)} documents")
            
            # Show source document previews
            print("📄 Top sources:")
            for i, doc in enumerate(response_with_rerank.source_documents[:2], 1):
                source = doc.metadata.get('source', 'Unknown')
                preview = doc.page_content[:100].replace('\n', ' ') + "..."
                print(f"  {i}. {source}: {preview}")
        
        # Test configuration changes
        print(f"\n🔧 Testing re-ranking configuration:")
        
        # Test disabling re-ranking
        rag_agent.configure_reranking(enable=False)
        print(f"   Re-ranking disabled: {not rag_agent.is_reranking_enabled()}")
        
        # Test re-enabling with different settings
        rag_agent.configure_reranking(enable=True, top_k=5, initial_k=15)
        print(f"   Re-ranking re-enabled: {rag_agent.is_reranking_enabled()}")
        print(f"   New settings - top_k: {rag_agent.reranking_top_k}, initial_k: {rag_agent.initial_retrieval_k}")
        
        # Test system health with re-ranking
        health = rag_agent.get_system_health()
        reranker_health = health.get("components", {}).get("reranker", {})
        print(f"\n🏥 System health - Re-ranker: {reranker_health.get('status', 'unknown')}")
        
        print("✅ Integrated re-ranking test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integrated re-ranking test failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_reranking_edge_cases():
    """Test re-ranking with edge cases"""
    print("\n🧪 Testing Re-ranking Edge Cases...")
    
    reranker = DocumentReranker()
    if not reranker.is_available():
        print("❌ Re-ranker not available, skipping edge case tests")
        return False
    
    try:
        # Test with empty documents
        empty_result = reranker.rerank_documents("test query", [], top_k=3)
        print(f"✅ Empty documents test: {len(empty_result[0])} docs returned")
        
        # Test with single document
        single_doc = [Document(page_content="Single test document", metadata={"source": "test"})]
        single_result = reranker.rerank_documents("test", single_doc, top_k=3)
        print(f"✅ Single document test: {len(single_result[0])} docs returned")
        
        # Test with more top_k than available documents
        test_docs = create_test_documents()[:3]  # Only 3 docs
        large_k_result = reranker.rerank_documents("test", test_docs, top_k=10)
        print(f"✅ Large top_k test: {len(large_k_result[0])} docs returned (max 3)")
        
        # Test with very long query
        long_query = "machine learning " * 100  # Very long query
        long_query_result = reranker.rerank_documents(long_query, test_docs, top_k=2)
        print(f"✅ Long query test: {len(long_query_result[0])} docs returned")
        
        print("✅ Edge case tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Edge case tests failed: {str(e)}")
        return False


def main():
    """Run all re-ranking tests"""
    print("🚀 Starting Re-ranking System Tests...")
    print("=" * 60)
    
    # Track test results
    test_results = {}
    
    # Test 1: Initialization
    test_results["initialization"] = test_reranker_initialization() is not None
    
    # Test 2: Standalone functionality
    test_results["standalone"] = test_standalone_reranking()
    
    # Test 3: Integration with RAG
    test_results["integration"] = test_integrated_reranking()
    
    # Test 4: Edge cases
    test_results["edge_cases"] = test_reranking_edge_cases()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All re-ranking tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)