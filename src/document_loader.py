"""
PDF Document Loader for RAG Agent
Handles loading and processing PDF documents from local folder
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """Handles loading and processing PDF documents"""
    
    def __init__(self, docs_folder: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF document loader
        
        Args:
            docs_folder: Path to folder containing PDF documents
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.docs_folder = Path(docs_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Ensure docs folder exists
        self.docs_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized PDF loader for folder: {self.docs_folder}")
    
    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the specified folder
        
        Returns:
            List of Document objects
        """
        if not self.docs_folder.exists():
            logger.error(f"Documents folder does not exist: {self.docs_folder}")
            return []
        
        # Get all PDF files
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.docs_folder}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading: {pdf_file.name}")
                
                # Load individual PDF
                loader = PyPDFLoader(str(pdf_file))
                pdf_documents = loader.load()
                
                # Add metadata
                for doc in pdf_documents:
                    doc.metadata.update({
                        "source_file": pdf_file.name,
                        "file_path": str(pdf_file)
                    })
                
                documents.extend(pdf_documents)
                logger.info(f"Loaded {len(pdf_documents)} pages from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split document chunks
        """
        if not documents:
            logger.warning("No documents to split")
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                "chunk_id": i,
                "chunk_size": len(doc.page_content)
            })
        
        logger.info(f"Created {len(split_docs)} document chunks")
        return split_docs
    
    def load_and_split(self) -> List[Document]:
        """
        Load all PDFs and split them into chunks
        
        Returns:
            List of split document chunks
        """
        documents = self.load_documents()
        return self.split_documents(documents)
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about loaded documents
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_documents": 0, "total_chunks": 0, "total_characters": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        source_files = set(doc.metadata.get("source_file", "unknown") for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(documents) if documents else 0,
            "source_files": list(source_files),
            "num_source_files": len(source_files)
        }


def main():
    """Test the document loader"""
    from dotenv import load_dotenv
    load_dotenv()
    
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    
    loader = PDFDocumentLoader(docs_folder)
    documents = loader.load_and_split()
    
    stats = loader.get_document_stats(documents)
    print("Document Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
