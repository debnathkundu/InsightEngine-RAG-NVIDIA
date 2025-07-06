"""
Vector Database for RAG Agent
Handles local vector storage using FAISS for document embeddings
"""

import os
import pickle
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .nvidia_embeddings import NVIDIAEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """Local vector database using FAISS for efficient similarity search"""
    
    def __init__(
        self,
        embeddings: NVIDIAEmbeddings,
        db_path: str = "./vector_db",
        index_name: str = "faiss_index"
    ):
        """
        Initialize vector database
        
        Args:
            embeddings: NVIDIA embeddings instance
            db_path: Path to store the vector database
            index_name: Name of the FAISS index
        """
        self.embeddings = embeddings
        self.db_path = Path(db_path)
        self.index_name = index_name
        self.vectorstore: Optional[FAISS] = None
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Paths for saving/loading
        self.index_path = self.db_path / f"{index_name}.faiss"
        self.metadata_path = self.db_path / f"{index_name}_metadata.pkl"
        
        logger.info(f"Initialized vector database at: {self.db_path}")
    
    def create_index(self, documents: List[Document]) -> bool:
        """
        Create vector index from documents
        
        Args:
            documents: List of documents to index
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return False
        
        try:
            logger.info(f"Creating vector index from {len(documents)} documents...")
            
            # Extract texts
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create FAISS vectorstore
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            logger.info("✅ Vector index created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}")
            return False
    
    def save_index(self) -> bool:
        """
        Save the vector index to disk
        
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("No vector index to save")
            return False
        
        try:
            logger.info("Saving vector index to disk...")
            
            # Save FAISS index
            self.vectorstore.save_local(str(self.db_path), self.index_name)
            
            logger.info(f"✅ Vector index saved to: {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vector index: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """
        Load vector index from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index_path.exists():
                logger.warning(f"No saved index found at: {self.index_path}")
                return False
            
            logger.info("Loading vector index from disk...")
            
            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                str(self.db_path),
                self.embeddings,
                self.index_name
            )
            
            logger.info("✅ Vector index loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector index: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to existing index
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided to add")
            return False
        
        try:
            if not self.vectorstore:
                logger.info("No existing index, creating new one...")
                return self.create_index(documents)
            
            logger.info(f"Adding {len(documents)} documents to existing index...")
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to vectorstore
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            
            logger.info("✅ Documents added successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            logger.error("No vector index available for search")
            return []
        
        try:
            logger.debug(f"Searching for: '{query}' (k={k})")
            
            if score_threshold is not None:
                # Search with score threshold
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                filtered_results = [
                    doc for doc, score in results 
                    if score >= score_threshold
                ]
                logger.debug(f"Found {len(filtered_results)} results above threshold {score_threshold}")
                return filtered_results
            else:
                # Regular similarity search
                results = self.vectorstore.similarity_search(query, k=k)
                logger.debug(f"Found {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            logger.error("No vector index available for search")
            return []
        
        try:
            logger.debug(f"Searching with scores for: '{query}' (k={k})")
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.debug(f"Found {len(results)} results with scores")
            return results
            
        except Exception as e:
            logger.error(f"Search with scores failed: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database
        
        Returns:
            Dictionary with database statistics
        """
        if not self.vectorstore:
            return {"status": "No index loaded", "document_count": 0}
        
        try:
            # Get document count from FAISS index
            doc_count = self.vectorstore.index.ntotal
            
            return {
                "status": "Index loaded",
                "document_count": doc_count,
                "index_path": str(self.index_path),
                "index_exists": self.index_path.exists()
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"status": "Error getting stats", "error": str(e)}
    
    def delete_index(self) -> bool:
        """
        Delete the vector index from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.index_path.exists():
                self.index_path.unlink()
                logger.info("Vector index deleted from disk")
            
            if self.metadata_path.exists():
                self.metadata_path.unlink()
                logger.info("Metadata file deleted from disk")
            
            self.vectorstore = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index: {str(e)}")
            return False


def main():
    """Test the vector database"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize components
    embeddings = NVIDIAEmbeddings()
    vector_db = VectorDatabase(embeddings)
    
    # Test documents
    test_docs = [
        Document(
            page_content="Artificial intelligence is the simulation of human intelligence in machines.",
            metadata={"source": "test1.pdf", "page": 1}
        ),
        Document(
            page_content="Machine learning is a subset of AI that enables computers to learn without explicit programming.",
            metadata={"source": "test2.pdf", "page": 1}
        )
    ]
    
    # Create and save index
    if vector_db.create_index(test_docs):
        vector_db.save_index()
        
        # Test search
        results = vector_db.similarity_search("What is AI?", k=2)
        print(f"Search results: {len(results)}")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content[:100]}...")
        
        # Get stats
        stats = vector_db.get_stats()
        print(f"Database stats: {stats}")


if __name__ == "__main__":
    main()
