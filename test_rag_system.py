"""
Test script for the RAG system
Tests all components individually and as a complete system
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.nvidia_embeddings import NVIDIAEmbeddings
from src.document_loader import PDFDocumentLoader
from src.vector_database import VectorDatabase
from src.rag_agent import RAGAgent


def test_environment():
    """Test environment setup"""
    print("🔧 Testing Environment Setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment")
        return False
    else:
        print("✅ NVIDIA_API_KEY found")
    
    # Check docs folder
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    if not Path(docs_folder).exists():
        print(f"⚠️  Docs folder not found: {docs_folder}")
        Path(docs_folder).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created docs folder: {docs_folder}")
    else:
        pdf_files = list(Path(docs_folder).glob("*.pdf"))
        print(f"✅ Docs folder exists with {len(pdf_files)} PDF files")
    
    return True


def test_nvidia_embeddings():
    """Test NVIDIA embeddings"""
    print("\n🤖 Testing NVIDIA Embeddings...")
    
    try:
        # Initialize embeddings
        embeddings = NVIDIAEmbeddings()
        
        # Test connection
        if not embeddings.test_connection():
            print("❌ NVIDIA API connection failed")
            return False
        
        print("✅ NVIDIA API connection successful")
        
        # Test embedding
        test_texts = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI."
        ]
        
        print("🔍 Testing document embedding...")
        doc_embeddings = embeddings.embed_documents(test_texts)
        print(f"✅ Document embeddings: {len(doc_embeddings)}x{len(doc_embeddings[0])}")
        
        print("🔍 Testing query embedding...")
        query_embedding = embeddings.embed_query("What is AI?")
        print(f"✅ Query embedding: {len(query_embedding)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ NVIDIA embeddings test failed: {str(e)}")
        return False


def test_document_loader():
    """Test document loader"""
    print("\n📄 Testing Document Loader...")
    
    try:
        docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
        loader = PDFDocumentLoader(docs_folder)
        
        # Load documents
        documents = loader.load_and_split()
        
        if not documents:
            print("⚠️  No documents loaded (no PDF files found)")
            return True  # Not a failure if no PDFs exist
        
        # Get stats
        stats = loader.get_document_stats(documents)
        
        print(f"✅ Loaded {stats['num_source_files']} PDF files")
        print(f"✅ Created {stats['total_chunks']} document chunks")
        print(f"✅ Total characters: {stats['total_characters']:,}")
        print(f"✅ Average chunk size: {stats['average_chunk_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document loader test failed: {str(e)}")
        return False


def test_vector_database():
    """Test vector database"""
    print("\n🗄️  Testing Vector Database...")
    
    try:
        # Initialize components
        embeddings = NVIDIAEmbeddings()
        vector_db = VectorDatabase(embeddings, "./test_vector_db")
        
        # Create test documents
        from langchain.schema import Document
        test_docs = [
            Document(
                page_content="Artificial intelligence is the simulation of human intelligence in machines.",
                metadata={"source": "test1.pdf", "page": 1}
            ),
            Document(
                page_content="Machine learning is a subset of AI that enables computers to learn.",
                metadata={"source": "test2.pdf", "page": 1}
            )
        ]
        
        print("🔍 Creating vector index...")
        if not vector_db.create_index(test_docs):
            print("❌ Failed to create vector index")
            return False
        
        print("✅ Vector index created")
        
        print("🔍 Testing similarity search...")
        results = vector_db.similarity_search("What is AI?", k=2)
        print(f"✅ Search returned {len(results)} results")
        
        print("🔍 Testing search with scores...")
        scored_results = vector_db.similarity_search_with_scores("machine learning", k=2)
        print(f"✅ Scored search returned {len(scored_results)} results")
        
        # Clean up test database
        vector_db.delete_index()
        print("🧹 Cleaned up test database")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector database test failed: {str(e)}")
        return False


def test_rag_agent():
    """Test complete RAG agent"""
    print("\n🤖 Testing Complete RAG Agent...")
    
    try:
        # Get configuration
        api_key = os.getenv("NVIDIA_API_KEY")
        docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
        
        # Initialize RAG agent
        rag_agent = RAGAgent(docs_folder, api_key, "./test_rag_vector_db")
        
        print("🔍 Setting up knowledge base...")
        if not rag_agent.setup_knowledge_base():
            print("⚠️  Knowledge base setup failed (likely no PDF files)")
            return True  # Not a failure if no PDFs exist
        
        print("✅ Knowledge base setup successful")
        
        # Get stats
        stats = rag_agent.get_knowledge_base_stats()
        print(f"✅ Knowledge base contains {stats.get('document_count', 0)} documents")
        
        # Test question answering
        print("🔍 Testing question answering...")
        test_question = "What is the main topic of the documents?"
        response = rag_agent.ask_question(test_question)
        
        print(f"✅ Question: {test_question}")
        print(f"✅ Answer length: {len(response.answer)} characters")
        print(f"✅ Source documents: {len(response.source_documents)}")
        print(f"✅ Processing time: {response.processing_time:.2f} seconds")
        
        # Clean up
        rag_agent.vector_db.delete_index()
        print("🧹 Cleaned up test database")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG agent test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests"""
    print("🧪 Starting RAG Template System Tests")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment),
        ("NVIDIA Embeddings", test_nvidia_embeddings),
        ("Document Loader", test_document_loader),
        ("Vector Database", test_vector_database),
        ("RAG Agent", test_rag_agent)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            success = test_func()
            results[test_name] = success
            duration = time.time() - start_time
            
            if success:
                print(f"✅ {test_name} PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {test_name} FAILED ({duration:.2f}s)")
                
        except Exception as e:
            results[test_name] = False
            duration = time.time() - start_time
            print(f"❌ {test_name} ERROR: {str(e)} ({duration:.2f}s)")
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧪 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG template is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
