"""
NVIDIA Embedding Integration for RAG Agent
Handles connection to NVIDIA LLaMA 3.2 NemoRetriever embedding model
"""

import os
import logging
import requests
import time
from typing import List, Optional, Dict, Any
from langchain.embeddings.base import Embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NVIDIAEmbeddings(Embeddings):
    """NVIDIA LLaMA 3.2 NemoRetriever embedding model integration"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "nvidia/nv-embed-v1",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize NVIDIA embeddings
        
        Args:
            api_key: NVIDIA API key
            model_name: Name of the embedding model
            base_url: Base URL for NVIDIA API
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key is required. Set NVIDIA_API_KEY environment variable.")
        
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        logger.info(f"Initialized NVIDIA embeddings with model: {self.model_name}")
        logger.info("Note: Using nvidia/nv-embed-v1 as fallback since llama-3_2-nemoretriever-1b-vlm-embed-v1 is not yet available")
    
    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """
        Make request to NVIDIA API for embeddings
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "input": texts,
            "model": self.model_name,
            "encoding_format": "float"
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making embedding request (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    logger.debug(f"Successfully got embeddings for {len(texts)} texts")
                    return embeddings
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
        
        raise Exception(f"Failed to get embeddings after {self.max_retries} attempts")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} documents")
        
        # Process in batches to avoid API limits
        batch_size = 10  # Adjust based on API limits
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = self._make_request(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay between batches to be respectful to the API
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        logger.info(f"Successfully embedded {len(all_embeddings)} documents")
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        logger.debug("Embedding query text")
        embeddings = self._make_request([text])
        return embeddings[0]
    
    def test_connection(self) -> bool:
        """
        Test the connection to NVIDIA API
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing NVIDIA API connection...")
            test_embedding = self.embed_query("test")
            logger.info(f"Connection successful! Embedding dimension: {len(test_embedding)}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model
        
        Returns:
            Embedding dimension
        """
        try:
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {str(e)}")
            return 1024  # Default fallback


def main():
    """Test the NVIDIA embeddings"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize embeddings
    embeddings = NVIDIAEmbeddings()
    
    # Test connection
    if embeddings.test_connection():
        print("✅ NVIDIA API connection successful!")
        
        # Test embedding
        test_texts = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI that focuses on algorithms."
        ]
        
        print(f"\nTesting embedding of {len(test_texts)} documents...")
        doc_embeddings = embeddings.embed_documents(test_texts)
        print(f"✅ Document embeddings successful! Shape: {len(doc_embeddings)}x{len(doc_embeddings[0])}")
        
        print("\nTesting query embedding...")
        query_embedding = embeddings.embed_query("What is artificial intelligence?")
        print(f"✅ Query embedding successful! Dimension: {len(query_embedding)}")
        
    else:
        print("❌ NVIDIA API connection failed!")


if __name__ == "__main__":
    main()
