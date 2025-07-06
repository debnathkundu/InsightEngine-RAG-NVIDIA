"""
NVIDIA RAG Agent Package
"""

__version__ = "1.0.0"
__author__ = "RAG Agent Developer"
__description__ = "NVIDIA LLaMA 3.2 NemoRetriever RAG Agent"

from .nvidia_embeddings import NVIDIAEmbeddings
from .document_loader import PDFDocumentLoader
from .vector_database import VectorDatabase
from .rag_agent import RAGAgent, RAGResponse

__all__ = [
    "NVIDIAEmbeddings",
    "PDFDocumentLoader", 
    "VectorDatabase",
    "RAGAgent",
    "RAGResponse"
]
