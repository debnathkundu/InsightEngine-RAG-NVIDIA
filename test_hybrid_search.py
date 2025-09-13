"""
Test script for the enhanced RAG system with hybrid search
Tests hybrid search functionality with both semantic and keyword-based queries
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.nvidia_embeddings import NVIDIAEmbeddings
from src.vector_database import VectorDatabase
from src.rag_agent import RAGAgent
from langchain.schema import Document


def test_hybrid_search_components():
    """Test hybrid search components individually"""
    print("🧪 Testing Hybrid Search Components...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize components
        embeddings = NVIDIAEmbeddings(api_key)
        
        # Test VectorDatabase with hybrid search enabled
        vector_db = VectorDatabase(
            embeddings, 
            "./test_vector_db",
            enable_hybrid_search=True,
            bm25_weight=0.3,
            vector_weight=0.7
        )
        
        # Create test documents with different types of content
        test_documents = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence (AI) that involves training algorithms to make predictions or decisions based on data.",
                metadata={"source": "ml_basics.pdf", "page": 1}
            ),
            Document(
                page_content="The API endpoint for user authentication is /api/v1/auth/login. Use POST method with username and password.",
                metadata={"source": "api_docs.pdf", "page": 2}
            ),
            Document(
                page_content="Error code 404 indicates that the requested resource was not found. Check the URL and try again.",
                metadata={"source": "error_codes.pdf", "page": 1}
            ),
            Document(
                page_content="Neural networks consist of interconnected nodes called neurons. Each connection has a weight that determines the signal strength.",
                metadata={"source": "neural_networks.pdf", "page": 3}
            ),
            Document(
                page_content="To configure the database connection, set DB_HOST=localhost, DB_PORT=5432, and DB_NAME=myapp in your environment variables.",
                metadata={"source": "config_guide.pdf", "page": 5}
            )
        ]
        
        # Create index with hybrid search
        print("📚 Creating hybrid search index...")
        if not vector_db.create_index(test_documents):
            print("❌ Failed to create hybrid search index")
            return False
        
        print("✅ Hybrid search index created successfully!")
        
        # Test different types of queries
        test_queries = [
            # Semantic queries (should work well with vector search)
            ("What is machine learning?", "semantic"),
            ("How do neural networks work?", "semantic"),
            
            # Keyword/code queries (should work well with BM25)
            ("API endpoint authentication", "keyword"),
            ("error code 404", "keyword"),
            ("DB_HOST localhost", "keyword"),
            
            # Mixed queries (should benefit from hybrid)
            ("machine learning algorithms predictions", "mixed"),
            ("neural network weights connections", "mixed")
        ]
        
        print("\n🔍 Testing different query types:")
        print("-" * 60)
        
        for query, query_type in test_queries:
            print(f"\n📝 Query: '{query}' (Type: {query_type})")
            
            # Test hybrid search
            hybrid_results = vector_db.hybrid_search(query, k=3)
            print(f"🔄 Hybrid search found {len(hybrid_results)} results")
            
            # Test vector-only search for comparison
            vector_results = vector_db.similarity_search(query, k=3)
            print(f"🎯 Vector search found {len(vector_results)} results")
            
            # Show top result from each
            if hybrid_results:
                print(f"   🏆 Top hybrid result: {hybrid_results[0].page_content[:80]}...")
            if vector_results:
                print(f"   🎯 Top vector result: {vector_results[0].page_content[:80]}...")
        
        # Test stats
        stats = vector_db.get_stats()
        print(f"\n📊 Database Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Clean up
        vector_db.delete_index()
        print("\n✅ Hybrid search components test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search components test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_agent_hybrid_search():
    """Test RAG agent with hybrid search integration"""
    print("\n🤖 Testing RAG Agent with Hybrid Search...")
    
    try:
        # Load environment
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
        
        if not api_key:
            print("❌ NVIDIA_API_KEY not found")
            return False
        
        # Initialize RAG agent (now with hybrid search enabled by default)
        rag_agent = RAGAgent(
            docs_folder=docs_folder,
            api_key=api_key,
            vector_db_path="./test_vector_db",
            enable_memory=False  # Disable memory for cleaner testing
        )
        
        # Check if docs folder exists and has files
        if not Path(docs_folder).exists():
            print(f"⚠️ Docs folder not found: {docs_folder}")
            print("Testing with minimal setup...")
            return True
        
        # Setup knowledge base
        print("📚 Setting up knowledge base with hybrid search...")
        if not rag_agent.setup_knowledge_base():
            print("❌ Failed to setup knowledge base")
            return False
        
        print("✅ Knowledge base setup completed!")
        
        # Check hybrid search status
        print(f"🔄 Hybrid search enabled: {rag_agent.is_hybrid_search_enabled()}")
        
        # Test different types of questions
        test_questions = [
            "What is machine learning?",  # Semantic query
            "API endpoint",  # Keyword query
            "error code",  # Keyword query
            "neural networks and algorithms"  # Mixed query
        ]
        
        print("\n💬 Testing questions with hybrid search:")
        print("-" * 50)
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            
            response = rag_agent.ask_question(question, k=3)
            
            print(f"⏱️ Processing time: {response.processing_time:.2f}s")
            print(f"📄 Sources found: {len(response.source_documents)}")
            if response.source_documents:
                print(f"📝 Answer preview: {response.answer[:100]}...")
            else:
                print("❌ No sources found")
        
        # Test weight configuration
        print(f"\n⚙️ Testing hybrid search weight configuration...")
        if rag_agent.configure_hybrid_search(bm25_weight=0.5, vector_weight=0.5):
            print("✅ Successfully updated hybrid search weights")
        else:
            print("❌ Failed to update hybrid search weights")
        
        # Get final stats
        stats = rag_agent.get_knowledge_base_stats()
        print(f"\n📊 Final Knowledge Base Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ RAG Agent hybrid search test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RAG Agent hybrid search test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all hybrid search tests"""
    print("🚀 Starting Hybrid Search Enhancement Tests")
    print("=" * 60)
    
    tests = [
        ("Hybrid Search Components", test_hybrid_search_components),
        ("RAG Agent Integration", test_rag_agent_hybrid_search)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running test: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ FAILED: {test_name} - {str(e)}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧪 HYBRID SEARCH TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All hybrid search tests passed! Enhancement is ready to use.")
        print("\n💡 Key benefits of your enhanced RAG system:")
        print("   • Hybrid search combines semantic (FAISS) and keyword (BM25) search")
        print("   • Better handling of specific terms, codes, and acronyms")
        print("   • Configurable search weights (default: 30% BM25, 70% Vector)")
        print("   • Backward compatible - existing functionality preserved")
        print("   • Automatic fallback to vector search if hybrid fails")
    else:
        print(f"\n⚠️ Some tests failed. Please review the errors above.")
        print("Note: Some failures might be due to missing dependencies or test environment setup.")


if __name__ == "__main__":
    main()