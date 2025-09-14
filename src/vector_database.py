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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi

from .nvidia_embeddings import NVIDIAEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """Local vector database using FAISS for efficient similarity search with hybrid BM25 support"""
    
    def __init__(
        self,
        embeddings: NVIDIAEmbeddings,
        db_path: str = "./vector_db",
        index_name: str = "faiss_index",
        enable_hybrid_search: bool = True,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7
    ):
        """
        Initialize vector database with hybrid search capabilities
        
        Args:
            embeddings: NVIDIA embeddings instance
            db_path: Path to store the vector database
            index_name: Name of the FAISS index
            enable_hybrid_search: Whether to enable hybrid search (BM25 + Vector)
            bm25_weight: Weight for BM25 retriever in ensemble (0.0-1.0)
            vector_weight: Weight for vector retriever in ensemble (0.0-1.0)
        """
        self.embeddings = embeddings
        self.db_path = Path(db_path)
        self.index_name = index_name
        self.vectorstore: Optional[FAISS] = None
        
        # Hybrid search components
        self.enable_hybrid_search = enable_hybrid_search
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self.documents_corpus: List[Document] = []  # Store documents for BM25
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Paths for saving/loading
        self.index_path = self.db_path / f"{index_name}.faiss"
        self.metadata_path = self.db_path / f"{index_name}_metadata.pkl"
        self.bm25_path = self.db_path / f"{index_name}_bm25.pkl"
        
        logger.info(f"Initialized vector database at: {self.db_path}")
        if enable_hybrid_search:
            logger.info(f"Hybrid search enabled - BM25 weight: {bm25_weight}, Vector weight: {vector_weight}")
    
    def create_index(self, documents: List[Document]) -> bool:
        """
        Create vector index from documents and optionally BM25 index for hybrid search
        
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
            
            # Store documents for BM25
            self.documents_corpus = documents.copy()
            
            # Extract texts
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create FAISS vectorstore
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Create BM25 index if hybrid search is enabled
            if self.enable_hybrid_search:
                self._create_bm25_index(documents)
                self._create_ensemble_retriever()
            
            logger.info("✅ Vector index created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}")
            return False
    
    def _create_bm25_index(self, documents: List[Document]) -> bool:
        """
        Create BM25 index from documents
        
        Args:
            documents: List of documents to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating BM25 index for hybrid search...")
            
            # Create BM25 retriever from documents
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            
            # Set default search parameters
            self.bm25_retriever.k = 4  # Default k value
            
            logger.info("✅ BM25 index created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create BM25 index: {str(e)}")
            return False
    
    def _create_ensemble_retriever(self) -> bool:
        """
        Create ensemble retriever combining FAISS and BM25
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vectorstore or not self.bm25_retriever:
                logger.warning("Cannot create ensemble retriever - missing components")
                return False
            
            logger.info("Creating hybrid ensemble retriever...")
            
            # Create vector retriever from FAISS vectorstore
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # Create ensemble retriever with weighted combination
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[self.bm25_weight, self.vector_weight]
            )
            
            logger.info(f"✅ Hybrid ensemble retriever created (BM25: {self.bm25_weight}, Vector: {self.vector_weight})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create ensemble retriever: {str(e)}")
            return False

    def save_index(self) -> bool:
        """
        Save the vector index and BM25 data to disk
        
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
            
            # Save BM25 data if hybrid search is enabled
            if self.enable_hybrid_search and self.bm25_retriever:
                self._save_bm25_index()
            
            logger.info(f"✅ Vector index saved to: {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vector index: {str(e)}")
            return False
    
    def _save_bm25_index(self) -> bool:
        """
        Save BM25 retriever and documents corpus to disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Saving BM25 index...")
            
            bm25_data = {
                'documents_corpus': self.documents_corpus,
                'bm25_retriever': self.bm25_retriever,
                'weights': {
                    'bm25_weight': self.bm25_weight,
                    'vector_weight': self.vector_weight
                }
            }
            
            with open(self.bm25_path, 'wb') as f:
                pickle.dump(bm25_data, f)
            
            logger.info("✅ BM25 index saved successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """
        Load vector index and BM25 data from disk
        
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
                self.index_name,
                allow_dangerous_deserialization=True
            )
            
            # Load BM25 data if hybrid search is enabled and file exists
            if self.enable_hybrid_search and self.bm25_path.exists():
                self._load_bm25_index()
                self._create_ensemble_retriever()
            
            logger.info("✅ Vector index loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector index: {str(e)}")
            return False
    
    def _load_bm25_index(self) -> bool:
        """
        Load BM25 retriever and documents corpus from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Loading BM25 index...")
            
            with open(self.bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
            
            self.documents_corpus = bm25_data.get('documents_corpus', [])
            self.bm25_retriever = bm25_data.get('bm25_retriever')
            
            # Load weights if available
            weights = bm25_data.get('weights', {})
            if weights:
                self.bm25_weight = weights.get('bm25_weight', self.bm25_weight)
                self.vector_weight = weights.get('vector_weight', self.vector_weight)
            
            logger.info("✅ BM25 index loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to existing index and update BM25 if enabled
        
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
            
            # Add to documents corpus for BM25
            self.documents_corpus.extend(documents)
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to vectorstore
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            
            # Update BM25 index if hybrid search is enabled
            if self.enable_hybrid_search:
                self._update_bm25_index()
                self._create_ensemble_retriever()
            
            logger.info("✅ Documents added successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    def _update_bm25_index(self) -> bool:
        """
        Update BM25 index with current documents corpus
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.documents_corpus:
                logger.warning("No documents in corpus for BM25 update")
                return False
            
            logger.info("Updating BM25 index...")
            
            # Recreate BM25 retriever with updated corpus
            self.bm25_retriever = BM25Retriever.from_documents(self.documents_corpus)
            self.bm25_retriever.k = 4  # Default k value
            
            logger.info("✅ BM25 index updated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {str(e)}")
            return False

    def delete_documents_by_source(self, source_file: str) -> bool:
        """
        Delete all document chunks associated with a specific source file from both FAISS and BM25.
        
        Args:
            source_file: The source file path to delete.
            
        Returns:
            True if documents were deleted, False otherwise.
        """
        if not self.vectorstore:
            logger.warning("Cannot delete: No vector index loaded.")
            return False

        try:
            # Get all current documents except those from the source file
            all_docs = []
            docs_to_delete_count = 0
            
            # Iterate through all documents in the vectorstore
            for doc_id in self.vectorstore.docstore._dict.keys():
                doc = self.vectorstore.docstore._dict[doc_id]
                if doc.metadata.get("source") == source_file:
                    docs_to_delete_count += 1
                else:
                    all_docs.append(doc)

            if docs_to_delete_count == 0:
                logger.info(f"No documents found with source: {source_file}")
                return False

            # Update documents corpus for BM25
            if self.enable_hybrid_search:
                self.documents_corpus = [
                    doc for doc in self.documents_corpus 
                    if doc.metadata.get("source") != source_file
                ]

            # Recreate vectorstore without deleted documents
            if all_docs:
                texts = [doc.page_content for doc in all_docs]
                metadatas = [doc.metadata for doc in all_docs]
                
                self.vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
                
                # Update BM25 index if hybrid search is enabled
                if self.enable_hybrid_search and self.documents_corpus:
                    self._update_bm25_index()
                    self._create_ensemble_retriever()
            else:
                # No documents left, create empty vectorstore
                self.vectorstore = None
                self.documents_corpus = []
                self.bm25_retriever = None
                self.ensemble_retriever = None
                
            logger.info(f"Deleted {docs_to_delete_count} chunks for source: {source_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete documents for source {source_file}: {str(e)}")
            return False

    def update_document(self, source_file: str, new_documents: List[Document]) -> bool:
        """
        Update a document in the index by deleting old chunks and adding new ones.
        
        Args:
            source_file: The source file being updated.
            new_documents: The list of new document chunks from the file.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # First, delete all existing chunks for this source
            self.delete_documents_by_source(source_file)
            
            # Then, add the new chunks
            if new_documents:
                self.add_documents(new_documents)
            
            logger.info(f"Successfully updated document: {source_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {source_file}: {str(e)}")
            return False
    
    def hybrid_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Perform hybrid search using both BM25 and vector similarity (if enabled)
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of documents from hybrid search
        """
        if not self.vectorstore:
            logger.error("No vector index available for search")
            return []
        
        try:
            # Use hybrid search if available, otherwise fall back to vector search
            if self.enable_hybrid_search and self.ensemble_retriever:
                logger.debug(f"Hybrid searching for: '{query}' (k={k})")
                
                # Set k for both retrievers
                if self.bm25_retriever:
                    self.bm25_retriever.k = k
                
                # Update vector retriever k
                vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
                self.ensemble_retriever.retrievers[1] = vector_retriever
                
                results = self.ensemble_retriever.get_relevant_documents(query)
                
                # Limit results to k (ensemble might return more)
                results = results[:k]
                
                logger.debug(f"Hybrid search found {len(results)} results")
                return results
            else:
                # Fall back to regular vector search
                logger.debug(f"Falling back to vector search for: '{query}' (k={k})")
                return self.similarity_search(query, k=k)
                
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            # Fall back to regular vector search
            return self.similarity_search(query, k=k)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search for similar documents using vector similarity only
        
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
            logger.debug(f"Vector searching for: '{query}' (k={k})")
            
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
            logger.error(f"Vector search failed: {str(e)}")
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
    
    def get_retriever(self, use_hybrid: bool = None, k: int = 4):
        """
        Get the appropriate retriever based on configuration
        
        Args:
            use_hybrid: Override hybrid search setting (None=use default)
            k: Number of documents to retrieve
            
        Returns:
            Retriever instance (ensemble, vector, or None)
        """
        if not self.vectorstore:
            logger.error("No vector index available")
            return None
        
        # Determine whether to use hybrid search
        should_use_hybrid = (
            use_hybrid if use_hybrid is not None 
            else (self.enable_hybrid_search and self.ensemble_retriever is not None)
        )
        
        if should_use_hybrid:
            # Update k values for all retrievers
            if self.bm25_retriever:
                self.bm25_retriever.k = k
            
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            self.ensemble_retriever.retrievers[1] = vector_retriever
            
            logger.debug("Using hybrid ensemble retriever")
            return self.ensemble_retriever
        else:
            # Return vector-only retriever
            logger.debug("Using vector-only retriever")
            return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database including hybrid search status
        
        Returns:
            Dictionary with database statistics
        """
        if not self.vectorstore:
            return {
                "status": "No index loaded", 
                "document_count": 0,
                "hybrid_search_enabled": self.enable_hybrid_search,
                "hybrid_search_available": False
            }
        
        try:
            # Get document count from FAISS index
            doc_count = self.vectorstore.index.ntotal
            
            stats = {
                "status": "Index loaded",
                "document_count": doc_count,
                "index_path": str(self.index_path),
                "index_exists": self.index_path.exists(),
                "hybrid_search_enabled": self.enable_hybrid_search,
                "hybrid_search_available": self.ensemble_retriever is not None,
                "documents_in_corpus": len(self.documents_corpus)
            }
            
            if self.enable_hybrid_search:
                stats.update({
                    "bm25_weight": self.bm25_weight,
                    "vector_weight": self.vector_weight,
                    "bm25_index_exists": self.bm25_path.exists()
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"status": "Error getting stats", "error": str(e)}
    
    def optimize_index(self) -> bool:
        """
        Optimize the vector index periodically for better performance
        
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.warning("No vector index available for optimization")
            return False
        
        try:
            doc_count = self.vectorstore.index.ntotal
            logger.info(f"Optimizing index with {doc_count} documents...")
            
            # For large collections, consider index optimization
            if doc_count > 10000:
                logger.info("Large collection detected - consider using IVF index for better performance")
                # Future enhancement: implement IVF index for better performance
            
            # For now, this is a placeholder for future optimizations
            # In production, you might want to:
            # 1. Rebuild index with optimal parameters
            # 2. Compress embeddings
            # 3. Use more efficient index types
            
            logger.info("Index optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Index optimization failed: {str(e)}")
            return False

    def should_rebuild_index(self) -> bool:
        """
        Determine if full rebuild is needed based on system health metrics
        
        Returns:
            True if rebuild is recommended, False otherwise
        """
        if not self.vectorstore:
            return True
        
        try:
            stats = self.get_stats()
            
            # Simple heuristics for when to rebuild:
            # 1. Very small index after many deletions
            # 2. Performance degradation indicators
            doc_count = stats.get('document_count', 0)
            
            # If we have very few documents, consider rebuild
            if doc_count < 10:
                logger.info("Small index detected - rebuild recommended")
                return True
            
            # Additional checks can be added here based on:
            # - Query performance metrics
            # - Index fragmentation
            # - Error rates
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check rebuild status: {str(e)}")
            return False
    
    def delete_index(self) -> bool:
        """
        Delete the vector index and BM25 data from disk
        
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
                
            if self.bm25_path.exists():
                self.bm25_path.unlink()
                logger.info("BM25 index deleted from disk")
            
            # Clear in-memory components
            self.vectorstore = None
            self.bm25_retriever = None
            self.ensemble_retriever = None
            self.documents_corpus = []
            
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
