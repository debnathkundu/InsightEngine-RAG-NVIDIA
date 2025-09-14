"""
Document Re-ranker Module
Provides advanced re-ranking capabilities using cross-encoder models
to improve the relevance of retrieved documents before sending to LLM.
"""

import os
import logging
from typing import List, Tuple, Optional
from datetime import datetime
from langchain.schema import Document

try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    LANGCHAIN_CROSS_ENCODER_AVAILABLE = True
except ImportError:
    LANGCHAIN_CROSS_ENCODER_AVAILABLE = False
    HuggingFaceCrossEncoder = None

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None

CROSS_ENCODER_AVAILABLE = LANGCHAIN_CROSS_ENCODER_AVAILABLE or SENTENCE_TRANSFORMERS_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentReranker:
    """
    Advanced document re-ranker using cross-encoder models.
    
    This class provides sophisticated re-ranking of retrieved documents
    using cross-encoder models that can better understand query-document
    relevance compared to simple embedding similarity.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: str = None
    ):
        """
        Initialize the document re-ranker.
        
        Args:
            model_name: Name of the cross-encoder model from HuggingFace
            max_length: Maximum sequence length for the model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.model = None
        self.model_type = None
        self.is_initialized = False
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning(
                "Cross-encoder not available. Install sentence-transformers: "
                "pip install sentence-transformers"
            )
            return
        
        # Try sentence-transformers CrossEncoder first (more stable)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"🚀 Initializing sentence-transformers CrossEncoder: {self.model_name}")
                if self.device:
                    self.model = CrossEncoder(self.model_name, device=self.device)
                else:
                    self.model = CrossEncoder(self.model_name)  # Let it choose the best device
                self.model_type = "sentence_transformers"
                self.is_initialized = True
                actual_device = getattr(self.model, 'device', 'unknown')
                logger.info(f"✅ CrossEncoder (sentence-transformers) initialized successfully on device: {actual_device}")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize sentence-transformers CrossEncoder: {e}")
        
        # Fallback to LangChain HuggingFaceCrossEncoder
        if LANGCHAIN_CROSS_ENCODER_AVAILABLE:
            try:
                logger.info(f"🚀 Initializing LangChain HuggingFaceCrossEncoder: {self.model_name}")
                self.model = HuggingFaceCrossEncoder(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device}
                )
                self.model_type = "langchain"
                self.is_initialized = True
                logger.info("✅ HuggingFaceCrossEncoder (LangChain) initialized successfully")
                return
            except Exception as e:
                logger.error(f"❌ Failed to initialize LangChain HuggingFaceCrossEncoder: {e}")
        
        # Both failed
        logger.error("❌ Failed to initialize any cross-encoder model")
        self.is_initialized = False
    
    def is_available(self) -> bool:
        """Check if re-ranking is available."""
        return CROSS_ENCODER_AVAILABLE and self.is_initialized
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        initial_k: int = 20
    ) -> Tuple[List[Document], List[float]]:
        """
        Re-rank documents based on query relevance using cross-encoder.
        
        Args:
            query: The user query
            documents: List of retrieved documents
            top_k: Number of top documents to return after re-ranking
            initial_k: Number of documents to consider for re-ranking
        
        Returns:
            Tuple of (reranked_documents, relevance_scores)
        """
        logger.info(f"🎯 rerank_documents() called with:")
        logger.info(f"   - Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        logger.info(f"   - Documents: {len(documents)}")
        logger.info(f"   - Top K: {top_k}")
        logger.info(f"   - Initial K: {initial_k}")
        logger.info(f"   - Is available: {self.is_available()}")
        
        if not self.is_available():
            logger.warning("❌ Re-ranking not available, returning original documents")
            logger.warning(f"   - Cross-encoder available: {CROSS_ENCODER_AVAILABLE}")
            logger.warning(f"   - Model initialized: {self.is_initialized}")
            return documents[:top_k], [1.0] * min(len(documents), top_k)
        
        if not documents:
            logger.warning("⚠️  No documents provided for re-ranking")
            return [], []
        
        try:
            # Limit to initial_k documents for efficiency
            docs_to_rerank = documents[:initial_k]
            
            logger.info(f"🔄 Re-ranking {len(docs_to_rerank)} documents for query")
            
            # Filter out documents that are too short for effective cross-encoder scoring
            valid_docs = []
            skipped_count = 0
            min_content_length = 30  # Minimum chars for meaningful cross-encoder scoring
            min_words = 3  # Minimum words for meaningful content
            
            for i, doc in enumerate(docs_to_rerank):
                content = doc.page_content.strip()
                content_length = len(content)
                word_count = len(content.split())
                
                # Skip very short content that causes cross-encoder to return NaN
                # Cross-encoders need sufficient semantic content (minimum ~30 chars) to:
                # - Generate meaningful token sequences after tokenization
                # - Create stable attention patterns in transformer layers
                # - Avoid mathematical instabilities in softmax/normalization operations
                # Short fragments like "API", "Chapter 1", or "\uf0b7\tLeav" lack semantic depth
                # and cause undefined behavior in the model's attention mechanism
                if content_length < min_content_length:
                    logger.warning(f"   📋 Skipping doc {i+1}: too short ({content_length} chars < {min_content_length}): {repr(content[:30])}")
                    skipped_count += 1
                    continue
                    
                # Skip content with too few words (likely headers/fragments)
                # Cross-encoders are trained on meaningful text pairs with semantic relationships
                # Content with <3 words typically represents:
                # - Document headers ("Machine Learning", "API Documentation")
                # - Navigation elements ("See below", "Next page")
                # - Corrupted text fragments or metadata
                # These lack the semantic richness needed for the model to compute reliable
                # query-document relevance scores, leading to NaN or unstable outputs
                if word_count < min_words:
                    logger.warning(f"   📋 Skipping doc {i+1}: too few words ({word_count} < {min_words}): {repr(content[:30])}")
                    skipped_count += 1
                    continue
                    
                valid_docs.append(doc)
            
            logger.info(f"📊 Document filtering: {len(docs_to_rerank)} → {len(valid_docs)} valid (skipped {skipped_count})")
            
            # If no valid documents for re-ranking, return original documents
            if not valid_docs:
                logger.warning("⚠️  No valid documents for re-ranking after filtering, returning original documents")
                return documents[:top_k], [1.0] * min(len(documents), top_k)
            
            # Prepare query-document pairs for valid documents only
            query_doc_pairs = []
            for i, doc in enumerate(valid_docs):
                # Truncate document content if too long
                content = doc.page_content
                original_length = len(content)
                if len(content) > self.max_length - len(query) - 10:
                    content = content[:self.max_length - len(query) - 10]
                    logger.debug(f"   Doc {i+1}: Truncated from {original_length} to {len(content)} chars")
                query_doc_pairs.append([query, content])
            
            logger.info(f"📊 Prepared {len(query_doc_pairs)} query-document pairs")
            
            # Get relevance scores
            start_time = datetime.now()
            logger.info("🧠 Computing relevance scores with cross-encoder...")
            
            try:
                if self.model_type == "sentence_transformers":
                    # Use sentence-transformers CrossEncoder (more stable)
                    logger.info(f"   Using sentence-transformers CrossEncoder for {len(query_doc_pairs)} pairs...")
                    raw_scores = self.model.predict(query_doc_pairs)
                    logger.info(f"   ✅ Sentence-transformers scoring completed")
                    
                    # Convert and validate scores
                    scores = []
                    for i, score in enumerate(raw_scores):
                        try:
                            # Convert numpy/tensor to float and validate
                            float_score = float(score)
                            if str(float_score).lower() in ['nan', 'inf', '-inf']:
                                logger.warning(f"   Invalid score at index {i}: {float_score}, using 0.0")
                                float_score = 0.0
                            scores.append(float_score)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"   Cannot convert score at index {i}: {score} ({e}), using 0.0")
                            scores.append(0.0)
                    
                elif self.model_type == "langchain":
                    # Use LangChain HuggingFaceCrossEncoder
                    logger.info(f"   Using LangChain HuggingFaceCrossEncoder for {len(query_doc_pairs)} pairs...")
                    scores = []
                    for i, pair in enumerate(query_doc_pairs):
                        try:
                            raw_score = self.model.score(pair)
                            float_score = float(raw_score)
                            if str(float_score).lower() in ['nan', 'inf', '-inf']:
                                logger.warning(f"   Invalid score at index {i}: {float_score}, using 0.0")
                                float_score = 0.0
                            scores.append(float_score)
                            if (i + 1) % 2 == 0:  # Log every 2 pairs
                                logger.debug(f"   Processed {i+1}/{len(query_doc_pairs)} pairs")
                        except Exception as pair_e:
                            logger.warning(f"   Error scoring pair {i+1}: {pair_e}")
                            scores.append(0.0)  # Default score for failed pairs
                    
                    logger.info(f"   ✅ LangChain scoring completed - {len(scores)} pairs")
                    
                else:
                    # Fallback method detection
                    if hasattr(self.model, 'predict'):
                        raw_scores = self.model.predict(query_doc_pairs)
                        scores = [float(s) if str(float(s)).lower() not in ['nan', 'inf', '-inf'] else 0.0 for s in raw_scores]
                        logger.info(f"   Using 'predict' method for {len(scores)} pairs")
                    elif hasattr(self.model, 'score'):
                        raw_scores = [self.model.score(pair) for pair in query_doc_pairs]
                        scores = [float(s) if str(float(s)).lower() not in ['nan', 'inf', '-inf'] else 0.0 for s in raw_scores]
                        logger.info(f"   Using 'score' method for {len(scores)} pairs")
                    else:
                        logger.error("   Model has neither 'score' nor 'predict' method")
                        raise AttributeError("Cross-encoder model missing required methods")
                    
            except Exception as scoring_e:
                logger.error(f"   Error during scoring: {scoring_e}")
                logger.error(f"   Error type: {type(scoring_e).__name__}")
                # Fallback to uniform scores
                scores = [0.5] * len(query_doc_pairs)
                logger.warning(f"   Using fallback uniform scores for {len(scores)} documents")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Re-ranking completed in {processing_time:.2f} seconds")
            logger.info(f"📈 Raw scores: {[f'{score:.4f}' if isinstance(score, (int, float)) else str(score) for score in scores[:3]]}{'...' if len(scores) > 3 else ''}")
            
            # Create scored documents using valid documents that were actually scored
            scored_docs = list(zip(valid_docs, scores))
            
            # Sort by relevance score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # If we don't have enough scored documents, add back skipped ones with low score
            if len(scored_docs) < top_k and skipped_count > 0:
                logger.info(f"🔄 Adding {min(skipped_count, top_k - len(scored_docs))} skipped documents with low scores")
                skipped_docs = [doc for doc in docs_to_rerank if doc not in valid_docs]
                for doc in skipped_docs[:top_k - len(scored_docs)]:
                    scored_docs.append((doc, 0.01))  # Very low score for skipped docs
            
            # Return top_k documents with their scores
            top_docs = [doc for doc, score in scored_docs[:top_k]]
            top_scores = [float(score) for doc, score in scored_docs[:top_k]]
            
            logger.info(f"🏆 Re-ranking results:")
            logger.info(f"   - Top score: {max(top_scores):.4f}")
            logger.info(f"   - Avg score: {sum(top_scores)/len(top_scores):.4f}")
            logger.info(f"   - Returned {len(top_docs)} documents")
            
            # Log top document sources for debugging
            for i, (doc, score) in enumerate(zip(top_docs[:3], top_scores[:3])):
                source = doc.metadata.get('source', 'Unknown')
                preview = doc.page_content[:80].replace('\n', ' ')
                logger.info(f"   - #{i+1} (score: {score:.4f}): {source} | {preview}...")
            
            return top_docs, top_scores
            
        except Exception as e:
            logger.error(f"❌ Error during re-ranking: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            # Fallback to original documents
            logger.warning("🔄 Falling back to original documents without re-ranking")
            return documents[:top_k], [1.0] * min(len(documents), top_k)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "is_available": self.is_available()
        }
    
    def benchmark_reranking(
        self,
        query: str,
        documents: List[Document],
        top_k_values: List[int] = [3, 5, 10]
    ) -> dict:
        """
        Benchmark re-ranking performance with different top_k values.
        
        Args:
            query: Test query
            documents: Test documents
            top_k_values: Different top_k values to test
        
        Returns:
            Dictionary with benchmark results
        """
        if not self.is_available():
            return {"error": "Re-ranking not available"}
        
        results = {}
        
        for top_k in top_k_values:
            start_time = datetime.now()
            reranked_docs, scores = self.rerank_documents(query, documents, top_k)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            results[f"top_{top_k}"] = {
                "processing_time": processing_time,
                "num_documents": len(reranked_docs),
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "min_score": min(scores) if scores else 0
            }
        
        return results


def create_reranker_from_env() -> DocumentReranker:
    """
    Create a DocumentReranker instance from environment variables.
    
    Returns:
        Configured DocumentReranker instance
    """
    logger.info("🔧 Creating DocumentReranker from environment variables...")
    
    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_length = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
    device = os.getenv("RERANKER_DEVICE", "cpu")
    
    logger.info(f"📋 Environment configuration:")
    logger.info(f"   - RERANKER_MODEL: {model_name}")
    logger.info(f"   - RERANKER_MAX_LENGTH: {max_length}")
    logger.info(f"   - RERANKER_DEVICE: {device}")
    
    return DocumentReranker(
        model_name=model_name,
        max_length=max_length,
        device=device
    )


# Export main classes and functions
__all__ = [
    "DocumentReranker",
    "create_reranker_from_env",
    "CROSS_ENCODER_AVAILABLE"
]