"""
RAG Agent - Main pipeline for Retrieval-Augmented Generation
Combines document retrieval with question answering using NVIDIA models
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM

from .document_loader import PDFDocumentLoader
from .nvidia_embeddings import NVIDIAEmbeddings
from .vector_database import VectorDatabase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG agent"""
    answer: str
    source_documents: List[Document]
    confidence_scores: Optional[List[float]] = None
    query: str = ""
    processing_time: float = 0.0


class SimpleNVIDIALLM:
    """Simple NVIDIA LLM wrapper for basic text generation"""

    def __init__(self, api_key: str, model_name: str = "meta/llama-3.1-8b-instruct"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://integrate.api.nvidia.com/v1"

    def generate_response(self, prompt: str) -> str:
        """Generate a response using NVIDIA API"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return "I apologize, but I'm unable to generate a response at the moment."

        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."


class RAGAgent:
    """Main RAG Agent that combines retrieval and generation"""
    
    def __init__(
        self,
        docs_folder: str,
        api_key: str,
        vector_db_path: str = "./vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize RAG Agent
        
        Args:
            docs_folder: Path to PDF documents folder
            api_key: NVIDIA API key
            vector_db_path: Path for vector database storage
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.docs_folder = docs_folder
        self.api_key = api_key
        self.vector_db_path = vector_db_path
        
        # Initialize components
        self.document_loader = PDFDocumentLoader(docs_folder, chunk_size, chunk_overlap)
        self.embeddings = NVIDIAEmbeddings(api_key)
        self.vector_db = VectorDatabase(self.embeddings, vector_db_path)
        self.llm = SimpleNVIDIALLM(api_key)

        # Custom prompt template
        self.prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
        
        logger.info("RAG Agent initialized successfully")
    
    def setup_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """
        Set up the knowledge base from PDF documents
        
        Args:
            force_rebuild: Whether to force rebuild the index even if it exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to load existing index first
            if not force_rebuild and self.vector_db.load_index():
                logger.info("✅ Loaded existing knowledge base")
                return True
            
            logger.info("Building knowledge base from PDF documents...")
            
            # Load and process documents
            documents = self.document_loader.load_and_split()
            
            if not documents:
                logger.error("No documents found to build knowledge base")
                return False
            
            # Get document stats
            stats = self.document_loader.get_document_stats(documents)
            logger.info(f"Loaded {stats['num_source_files']} PDF files with {stats['total_chunks']} chunks")
            
            # Create vector index
            if not self.vector_db.create_index(documents):
                logger.error("Failed to create vector index")
                return False
            
            # Save index
            if not self.vector_db.save_index():
                logger.error("Failed to save vector index")
                return False
            
            logger.info("✅ Knowledge base setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup knowledge base: {str(e)}")
            return False
    

    
    def ask_question(self, question: str, k: int = 4) -> RAGResponse:
        """
        Ask a question and get an answer from the knowledge base

        Args:
            question: The question to ask
            k: Number of relevant documents to retrieve

        Returns:
            RAGResponse with answer and source documents
        """
        import time
        start_time = time.time()

        try:
            if not self.vector_db.vectorstore:
                logger.error("Vector database not initialized. Please setup knowledge base first.")
                return RAGResponse(
                    answer="Knowledge base not initialized. Please setup the knowledge base first.",
                    source_documents=[],
                    query=question,
                    processing_time=time.time() - start_time
                )

            logger.info(f"Processing question: {question}")

            # Get relevant documents
            scored_docs = self.vector_db.similarity_search_with_scores(question, k=k)

            if not scored_docs:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    source_documents=[],
                    query=question,
                    processing_time=time.time() - start_time
                )

            # Extract documents and scores
            source_docs = [doc for doc, score in scored_docs]
            scores = [score for doc, score in scored_docs]

            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in source_docs])

            # Create prompt
            prompt = self.prompt_template.format(context=context, question=question)

            # Generate answer using LLM
            answer = self.llm.generate_response(prompt)

            processing_time = time.time() - start_time

            logger.info(f"Question answered in {processing_time:.2f} seconds")

            return RAGResponse(
                answer=answer,
                source_documents=source_docs,
                confidence_scores=scores,
                query=question,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                source_documents=[],
                query=question,
                processing_time=time.time() - start_time
            )
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Get relevant documents for a query with similarity scores
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_db.similarity_search_with_scores(query, k=k)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with knowledge base statistics
        """
        vector_stats = self.vector_db.get_stats()
        
        # Add document loader stats if available
        try:
            if os.path.exists(self.docs_folder):
                pdf_files = list(Path(self.docs_folder).glob("*.pdf"))
                vector_stats["pdf_files_available"] = len(pdf_files)
                vector_stats["docs_folder"] = self.docs_folder
        except Exception:
            pass
        
        return vector_stats
    
    def add_documents_to_knowledge_base(self, new_docs_folder: Optional[str] = None) -> bool:
        """
        Add new documents to the existing knowledge base
        
        Args:
            new_docs_folder: Optional path to new documents folder
            
        Returns:
            True if successful, False otherwise
        """
        try:
            folder = new_docs_folder or self.docs_folder
            
            # Load new documents
            loader = PDFDocumentLoader(folder)
            new_documents = loader.load_and_split()
            
            if not new_documents:
                logger.warning("No new documents found to add")
                return False
            
            # Add to vector database
            if self.vector_db.add_documents(new_documents):
                # Save updated index
                self.vector_db.save_index()
                logger.info(f"✅ Added {len(new_documents)} new document chunks")
                return True
            else:
                logger.error("Failed to add new documents")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False


def main():
    """Test the RAG agent"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get configuration
    api_key = os.getenv("NVIDIA_API_KEY")
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment variables")
        return
    
    # Initialize RAG agent
    rag_agent = RAGAgent(docs_folder, api_key)
    
    # Setup knowledge base
    if rag_agent.setup_knowledge_base():
        print("✅ Knowledge base setup successful!")
        
        # Get stats
        stats = rag_agent.get_knowledge_base_stats()
        print(f"Knowledge base stats: {stats}")
        
        # Test question
        test_question = "What is the main topic of the documents?"
        response = rag_agent.ask_question(test_question)
        
        print(f"\nQuestion: {test_question}")
        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.source_documents)} documents")
        print(f"Processing time: {response.processing_time:.2f} seconds")
        
    else:
        print("❌ Failed to setup knowledge base")


if __name__ == "__main__":
    main()
