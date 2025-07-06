"""
Main Application - Interactive RAG Agent Interface
Provides CLI interface for interacting with the RAG agent
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_agent import RAGAgent
from src.nvidia_embeddings import NVIDIAEmbeddings


def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🤖 RAG Agent - NVIDIA NemoRetriever Template")
    print("=" * 60)
    print("Ask questions about your PDF documents!")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("Type 'help' for available commands")
    print("Type 'stats' to see knowledge base statistics")
    print("Type 'rebuild' to rebuild the knowledge base")
    print("-" * 60)


def print_help():
    """Print help information"""
    print("\n📖 Available Commands:")
    print("  help     - Show this help message")
    print("  stats    - Show knowledge base statistics")
    print("  rebuild  - Rebuild the knowledge base from PDFs")
    print("  clear    - Clear the screen")
    print("  quit/exit/q - Exit the application")
    print("\n💡 Tips:")
    print("  - Ask specific questions about your documents")
    print("  - The system will show relevant source documents")
    print("  - Processing time is displayed for each query")
    print()


def print_stats(rag_agent: RAGAgent):
    """Print knowledge base statistics"""
    print("\n📊 Knowledge Base Statistics:")
    stats = rag_agent.get_knowledge_base_stats()
    
    for key, value in stats.items():
        if key == "status":
            status_emoji = "✅" if value == "Index loaded" else "❌"
            print(f"  {status_emoji} Status: {value}")
        elif key == "document_count":
            print(f"  📄 Documents: {value}")
        elif key == "pdf_files_available":
            print(f"  📁 PDF Files: {value}")
        elif key == "docs_folder":
            print(f"  📂 Docs Folder: {value}")
        elif key == "index_path":
            print(f"  💾 Index Path: {value}")
        else:
            print(f"  {key}: {value}")
    print()


def format_response(response):
    """Format and display the RAG response"""
    print(f"\n🤖 Answer:")
    print("-" * 40)
    print(response.answer)
    print("-" * 40)
    
    if response.source_documents:
        print(f"\n📚 Sources ({len(response.source_documents)} documents):")
        for i, doc in enumerate(response.source_documents, 1):
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "Unknown")
            
            print(f"\n  [{i}] {source_file} (Page: {page}, Chunk: {chunk_id})")
            # Show first 150 characters of the source text
            preview = doc.page_content[:150]
            if len(doc.page_content) > 150:
                preview += "..."
            print(f"      \"{preview}\"")
    
    print(f"\n⏱️  Processing time: {response.processing_time:.2f} seconds")
    print()


def setup_rag_agent():
    """Initialize and setup the RAG agent"""
    print("🔧 Initializing RAG Agent...")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    api_key = os.getenv("NVIDIA_API_KEY")
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in environment variables")
        print("Please check your .env file and ensure the API key is set")
        return None
    
    # Check if docs folder exists
    if not Path(docs_folder).exists():
        print(f"📁 Creating documents folder: {docs_folder}")
        Path(docs_folder).mkdir(parents=True, exist_ok=True)
        print(f"📝 Please add your PDF files to: {Path(docs_folder).absolute()}")
    
    # Test NVIDIA API connection
    print("🔌 Testing NVIDIA API connection...")
    try:
        embeddings = NVIDIAEmbeddings(api_key)
        if not embeddings.test_connection():
            print("❌ Failed to connect to NVIDIA API")
            return None
        print("✅ NVIDIA API connection successful!")
    except Exception as e:
        print(f"❌ NVIDIA API connection failed: {str(e)}")
        return None
    
    # Initialize RAG agent
    try:
        rag_agent = RAGAgent(docs_folder, api_key, vector_db_path)
        print("✅ RAG Agent initialized successfully!")
        return rag_agent
    except Exception as e:
        print(f"❌ Failed to initialize RAG Agent: {str(e)}")
        return None


def main():
    """Main application loop"""
    print_banner()
    
    # Setup RAG agent
    rag_agent = setup_rag_agent()
    if not rag_agent:
        print("❌ Failed to initialize RAG Agent. Exiting...")
        return

    # Setup knowledge base
    print("\n🔨 Setting up knowledge base...")
    if not rag_agent.setup_knowledge_base():
        print("❌ Failed to setup knowledge base.")
        print("Please ensure you have PDF files in the documents folder and try again.")

        # Ask if user wants to continue anyway
        response = input("\nWould you like to continue anyway? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            return
    else:
        print("✅ Knowledge base setup completed!")
        print_stats(rag_agent)

    print("\n🚀 RAG Agent is ready! Ask me anything about your documents.")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            print("\n" + "="*60)
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using RAG Agent! Goodbye!")
                break
            
            elif question.lower() == 'help':
                print_help()
                continue
            
            elif question.lower() == 'stats':
                print_stats(rag_agent)
                continue
            
            elif question.lower() == 'rebuild':
                print("\n🔨 Rebuilding knowledge base...")
                if rag_agent.setup_knowledge_base(force_rebuild=True):
                    print("✅ Knowledge base rebuilt successfully!")
                    print_stats(rag_agent)
                else:
                    print("❌ Failed to rebuild knowledge base")
                continue
            
            elif question.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            
            # Process question
            print(f"\n🔍 Searching knowledge base...")
            response = rag_agent.ask_question(question)
            format_response(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again or type 'help' for available commands.")


if __name__ == "__main__":
    main()
