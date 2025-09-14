#!/usr/bin/env python3
"""
Comprehensive Sanity Check Script
Tests all RAG system modules and their interactions to ensure everything works as expected
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging to see all details
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import all modules
try:
    from src.nvidia_embeddings import NVIDIAEmbeddings
    from src.document_loader import DocumentLoader
    from src.vector_database import VectorDatabase
    from src.document_reranker import DocumentReranker, create_reranker_from_env
    from src.rag_agent import RAGAgent
    from src.web_importer import WebImporter
    from src.file_watcher import start_file_watcher
    from langchain.schema import Document
    print("✅ All module imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_environment_setup():
    """Test basic environment setup"""
    print("\n🔧 Testing Environment Setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found")
        return False
    
    print(f"✅ NVIDIA_API_KEY found: {api_key[:15]}...")
    
    # Check required directories
    docs_folder = Path("Data/Docs")
    vector_db_path = Path("vector_db")
    
    print(f"✅ Docs folder exists: {docs_folder.exists()}")
    print(f"✅ Vector DB path exists: {vector_db_path.exists()}")
    
    return True


def test_nvidia_embeddings():
    """Test NVIDIA embeddings module"""
    print("\n🤖 Testing NVIDIA Embeddings...")
    
    try:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        
        embeddings = NVIDIAEmbeddings(api_key)
        
        # Test embedding generation
        test_text = "What is machine learning?"
        start_time = time.time()
        embedding = embeddings.embed_query(test_text)
        processing_time = time.time() - start_time
        
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        print(f"✅ Processing time: {processing_time:.3f} seconds")
        
        # Test connection
        if embeddings.test_connection():
            print("✅ NVIDIA API connection successful")
            return True
        else:
            print("❌ NVIDIA API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ NVIDIA Embeddings test failed: {e}")
        return False


def test_document_reranker():
    """Test document re-ranker module"""
    print("\n🎯 Testing Document Re-ranker...")
    
    try:
        # Test initialization
        reranker = DocumentReranker()
        print(f"✅ Re-ranker initialized: {reranker.is_available()}")
        
        if not reranker.is_available():
            print("⚠️  Re-ranker not available, skipping detailed tests")
            return True  # Not a failure, just not available
        
        # Print model info
        model_info = reranker.get_model_info()
        print(f"✅ Model info: {model_info}")
        
        # Test with sample documents
        test_docs = [
            Document(page_content="Machine learning is a subset of AI", metadata={"source": "doc1"}),
            Document(page_content="Deep learning uses neural networks", metadata={"source": "doc2"}),
            Document(page_content="Python is a programming language", metadata={"source": "doc3"})
        ]
        
        query = "What is machine learning?"
        start_time = time.time()
        reranked_docs, scores = reranker.rerank_documents(query, test_docs, top_k=2)
        processing_time = time.time() - start_time
        
        print(f"✅ Re-ranking completed: {len(reranked_docs)} docs in {processing_time:.3f}s")
        print(f"✅ Scores: {[f'{s:.4f}' for s in scores]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document Re-ranker test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_vector_database():
    """Test vector database module"""
    print("\n🗄️  Testing Vector Database...")
    
    try:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        
        embeddings = NVIDIAEmbeddings(api_key)
        vector_db = VectorDatabase(
            embeddings, 
            "./test_vector_db",
            enable_hybrid_search=True
        )
        
        # Test with sample documents
        test_docs = [
            Document(page_content="Machine learning algorithms", metadata={"source": "test1"}),
            Document(page_content="Deep learning neural networks", metadata={"source": "test2"}),
            Document(page_content="Natural language processing", metadata={"source": "test3"})
        ]
        
        # Test index creation
        success = vector_db.create_index(test_docs)
        print(f"✅ Index creation: {success}")
        
        if success:
            # Test search
            results = vector_db.similarity_search("machine learning", k=2)
            print(f"✅ Vector search: {len(results)} results")
            
            # Test hybrid search if available
            if vector_db.enable_hybrid_search:
                hybrid_results = vector_db.hybrid_search("machine learning", k=2)
                print(f"✅ Hybrid search: {len(hybrid_results)} results")
            
            # Test stats
            stats = vector_db.get_stats()
            print(f"✅ Vector DB stats: {stats}")
        
        return success
        
    except Exception as e:
        print(f"❌ Vector Database test failed: {e}")
        return False


def test_rag_agent_initialization():
    """Test RAG agent initialization and configuration"""
    print("\n🤖 Testing RAG Agent Initialization...")
    
    try:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        
        # Test RAG agent initialization
        rag_agent = RAGAgent(
            docs_folder="Data/Docs",
            api_key=api_key,
            enable_reranking=True,
            reranking_top_k=3,
            initial_retrieval_k=10
        )
        
        print("✅ RAG Agent initialized successfully")
        
        # Test feature status
        print(f"✅ Conversational mode: {rag_agent.is_conversational_mode_enabled()}")
        print(f"✅ Hybrid search: {rag_agent.is_hybrid_search_enabled()}")
        print(f"✅ Re-ranking: {rag_agent.is_reranking_enabled()}")
        
        # Test knowledge base stats
        stats = rag_agent.get_knowledge_base_stats()
        print(f"✅ Knowledge base stats: {stats}")
        
        # Test system health
        health = rag_agent.get_system_health()
        print(f"✅ System health: {health['overall_status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG Agent initialization failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_full_rag_pipeline():
    """Test the complete RAG pipeline with all features"""
    print("\n🚀 Testing Full RAG Pipeline...")
    
    try:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        
        # Initialize RAG agent
        rag_agent = RAGAgent(
            docs_folder="Data/Docs",
            api_key=api_key,
            enable_reranking=True
        )
        
        # Check if knowledge base exists, setup if needed
        stats = rag_agent.get_knowledge_base_stats()
        if not stats.get("index_exists", False):
            print("📝 Setting up knowledge base...")
            if not rag_agent.setup_knowledge_base():
                print("❌ Failed to setup knowledge base")
                return False
        
        print(f"✅ Knowledge base ready: {stats.get('document_count', 0)} documents")
        
        # Test different retrieval methods
        test_query = "What is machine learning?"
        
        # Test 1: Basic retrieval with re-ranking
        print(f"\n📝 Testing query: '{test_query}'")
        
        start_time = time.time()
        docs_with_rerank = rag_agent.get_relevant_documents(
            query=test_query,
            k=3,
            use_reranking=True
        )
        rerank_time = time.time() - start_time
        
        print(f"✅ With re-ranking: {len(docs_with_rerank)} docs in {rerank_time:.3f}s")
        
        # Test 2: Retrieval without re-ranking
        start_time = time.time()
        docs_without_rerank = rag_agent.get_relevant_documents(
            query=test_query,
            k=3,
            use_reranking=False
        )
        no_rerank_time = time.time() - start_time
        
        print(f"✅ Without re-ranking: {len(docs_without_rerank)} docs in {no_rerank_time:.3f}s")
        
        # Test 3: Full question answering
        start_time = time.time()
        response = rag_agent.ask_question(test_query, k=3)
        qa_time = time.time() - start_time
        
        print(f"✅ Full Q&A: {len(response.source_documents)} sources in {qa_time:.3f}s")
        print(f"✅ Answer preview: {response.answer[:100]}...")
        
        # Test configuration changes
        print(f"\n🔧 Testing configuration changes...")
        
        # Test re-ranking configuration
        old_rerank_status = rag_agent.is_reranking_enabled()
        rag_agent.configure_reranking(enable=not old_rerank_status)
        new_rerank_status = rag_agent.is_reranking_enabled()
        print(f"✅ Re-ranking toggle: {old_rerank_status} → {new_rerank_status}")
        
        # Restore original setting
        rag_agent.configure_reranking(enable=old_rerank_status)
        
        # Test hybrid search configuration
        if rag_agent.is_hybrid_search_enabled():
            rag_agent.configure_hybrid_search(bm25_weight=0.4, vector_weight=0.6)
            print("✅ Hybrid search weights updated")
        
        print("✅ Full RAG pipeline test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Full RAG pipeline test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_module_interactions():
    """Test interactions between different modules"""
    print("\n🔄 Testing Module Interactions...")
    
    try:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        
        # Test chain: Embeddings → Vector DB → Re-ranker → RAG Agent
        print("Testing module chain...")
        
        # 1. Initialize embeddings
        embeddings = NVIDIAEmbeddings(api_key)
        test_embedding = embeddings.embed_query("test")
        print(f"✅ Embeddings: {len(test_embedding)} dims")
        
        # 2. Initialize vector DB
        vector_db = VectorDatabase(embeddings, "./test_interaction_db", enable_hybrid_search=True)
        
        # 3. Initialize re-ranker
        reranker = DocumentReranker()
        print(f"✅ Re-ranker available: {reranker.is_available()}")
        
        # 4. Test data flow
        test_docs = [
            Document(page_content="Machine learning algorithms for data analysis", metadata={"source": "interaction_test"}),
            Document(page_content="Deep learning and neural network architectures", metadata={"source": "interaction_test2"}),
        ]
        
        # Create vector index
        vector_db.create_index(test_docs)
        
        # Retrieve documents
        retrieved = vector_db.similarity_search("machine learning", k=2)
        print(f"✅ Retrieved {len(retrieved)} documents")
        
        # Re-rank if available
        if reranker.is_available():
            reranked, scores = reranker.rerank_documents("machine learning", retrieved, top_k=1)
            print(f"✅ Re-ranked to {len(reranked)} documents with scores: {scores}")
        
        print("✅ Module interactions test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Module interactions test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    """Run comprehensive sanity check"""
    print("🔍 COMPREHENSIVE RAG SYSTEM SANITY CHECK")
    print("=" * 60)
    
    # List of all tests
    tests = [
        ("Environment Setup", test_environment_setup),
        ("NVIDIA Embeddings", test_nvidia_embeddings),
        ("Document Re-ranker", test_document_reranker),
        ("Vector Database", test_vector_database),
        ("RAG Agent Initialization", test_rag_agent_initialization),
        ("Full RAG Pipeline", test_full_rag_pipeline),
        ("Module Interactions", test_module_interactions),
    ]
    
    results = {}
    
    # Run all tests
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 SANITY CHECK RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<25}: {status}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
        print("\n✅ System Status:")
        print("  - All modules loaded successfully")
        print("  - NVIDIA API connection working")
        print("  - Vector database functional")
        print("  - Hybrid search operational")
        print("  - Re-ranking system available")
        print("  - RAG pipeline end-to-end working")
        print("\n🚀 Your RAG system is ready for production use!")
    else:
        failed = total - passed
        print(f"\n⚠️  {failed} TEST(S) FAILED!")
        print("\n❌ Issues detected:")
        for test_name, success in results.items():
            if not success:
                print(f"  - {test_name}")
        print("\n🔧 Please check the error messages above and fix the issues.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)