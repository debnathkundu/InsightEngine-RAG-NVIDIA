"""
Universal Document Loader for RAG Agent
Handles loading and processing of various document types from a local folder.
Supports: PDF, DOCX, PPTX, TXT, and more via UnstructuredFileLoader.
"""

import os
import logging
from pathlib import Path
from typing import List, Any, Dict

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles loading and processing of various document types."""
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredLoader,
        ".pptx": UnstructuredLoader,
        ".txt": UnstructuredLoader,
        # For image support (optional, uncomment to enable)
        # ".png": UnstructuredImageLoader,
        # ".jpg": UnstructuredImageLoader,
    }

    def __init__(self, docs_folder: str):
        """
        Initialize the document loader.
        
        Args:
            docs_folder: Path to the folder containing documents.
        """
        self.docs_folder = Path(docs_folder)
        
        # Supported file extensions and their loaders
        self.SUPPORTED_EXTENSIONS: Dict[str, Any] = {
            ".pdf": PyPDFLoader,
            ".docx": UnstructuredLoader,
            ".pptx": UnstructuredLoader,
            ".txt": UnstructuredLoader,
            # For image support (optional, uncomment to enable)
            # ".png": UnstructuredImageLoader,
            # ".jpg": UnstructuredImageLoader,
        }
        
        self.docs_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized document loader for folder: {self.docs_folder}")

    def _get_loader(self, file_path: Path) -> Any:
        """Factory method to get the appropriate loader for a file extension."""
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {ext}. Skipping.")
            return None
        
        logger.debug(f"Using {self.SUPPORTED_EXTENSIONS[ext].__name__} for {file_path.name}")
        return self.SUPPORTED_EXTENSIONS[ext](str(file_path))

    def load_file(self, file_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Load a single file and split it into chunks.
        
        Args:
            file_path: Path to the file to load.
            chunk_size: The size of each text chunk.
            chunk_overlap: The overlap between consecutive chunks.
            
        Returns:
            List of Document objects from the file.
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return []
        
        loader = self._get_loader(path)
        if not loader:
            return []
        
        try:
            logger.info(f"Loading single file: {path.name}")
            
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata.update({
                    "source_file": path.name,
                    "file_path": str(path),
                    "source": str(path),
                })

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
            split_docs = text_splitter.split_documents(loaded_docs)
            
            logger.info(f"Loaded and split {path.name} into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return []

    def load_documents(self) -> List[Document]:
        """
        Load all supported documents from the folder.
        
        Returns:
            List of loaded documents.
        """
        all_docs = []
        logger.info(f"Loading all documents from {self.docs_folder}")
        
        for file_path in self.docs_folder.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                loader = self._get_loader(file_path)
                if loader:
                    try:
                        loaded_docs = loader.load()
                        for doc in loaded_docs:
                            doc.metadata.update({
                                "source_file": file_path.name,
                                "file_path": str(file_path),
                                "source": str(file_path),
                            })
                        all_docs.extend(loaded_docs)
                        logger.info(f"Successfully loaded {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {e}")
        
        logger.info(f"Loaded {len(all_docs)} documents from {len(list(self.docs_folder.rglob('*')))} files.")
        return all_docs

    def load_and_split(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Load all documents and split them into chunks.
        
        Args:
            chunk_size: The size of each text chunk.
            chunk_overlap: The overlap between consecutive chunks.
            
        Returns:
            List of split document chunks.
        """
        documents = self.load_documents()
        
        if not documents:
            logger.warning("No documents to split.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
        return split_docs
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents.
        
        Args:
            documents: List of documents.
            
        Returns:
            Dictionary with document statistics.
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
    """Test the document loader."""
    from dotenv import load_dotenv
    load_dotenv()
    
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    
    loader = DocumentLoader(docs_folder)
    documents = loader.load_and_split()
    
    if documents:
        stats = loader.get_document_stats(documents)
        print("Document Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("No documents were loaded.")


if __name__ == "__main__":
    main()
