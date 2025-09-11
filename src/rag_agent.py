"""
RAG Agent - Main pipeline for Retrieval-Augmented Generation
Combines document retrieval with question answering using NVIDIA models
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain.schema import Document
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult

from .document_loader import DocumentLoader
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
    chat_history: Optional[List[Tuple[str, str]]] = None


class NVIDIALangChainLLM(LLM):
    """LangChain-compatible NVIDIA LLM wrapper for conversational memory support"""

    api_key: str
    model_name: str = "meta/llama-3.1-8b-instruct"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    max_tokens: int = 1024
    temperature: float = 0.1

    def __init__(self, api_key: str, model_name: str = "meta/llama-3.1-8b-instruct", **kwargs):
        # Pass all parameters to parent constructor, including api_key
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "nvidia_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the NVIDIA API and return the output."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
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

    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "base_url": self.base_url,
        }

    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate multiple outputs for multiple prompts."""
        generations = []
        for prompt in prompts:
            output = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=output)])
        
        return LLMResult(generations=generations)
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


class SimpleNVIDIALLM:
    """Simple NVIDIA LLM wrapper for basic text generation (backward compatibility)"""

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
    """Main RAG Agent that combines retrieval and generation with conversational memory"""
    
    def __init__(
        self,
        docs_folder: str,
        api_key: str,
        vector_db_path: str = "./vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        status_callback=None,
        enable_memory: bool = True,
        memory_window_size: int = 10
    ):
        """
        Initialize RAG Agent
        
        Args:
            docs_folder: Path to PDF documents folder
            api_key: NVIDIA API key
            vector_db_path: Path for vector database storage
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            status_callback: Optional callback for status updates
            enable_memory: Whether to enable conversational memory
            memory_window_size: Number of conversation turns to remember
        """
        self.docs_folder = docs_folder
        self.api_key = api_key
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_memory = enable_memory
        self.memory_window_size = memory_window_size
        
        # Initialize components
        self.document_loader = DocumentLoader(docs_folder)
        self.embeddings = NVIDIAEmbeddings(api_key)
        self.vector_db = VectorDatabase(self.embeddings, vector_db_path)
        
        # Initialize both LLM wrappers for flexibility
        self.llm = SimpleNVIDIALLM(api_key)  # For backward compatibility
        
        # Try to initialize LangChain LLM wrapper with error handling
        try:
            self.langchain_llm = NVIDIALangChainLLM(api_key=api_key)  # For conversational chains
            logger.info("✅ LangChain LLM wrapper initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain LLM wrapper: {str(e)}")
            logger.warning("Falling back to basic LLM only - conversational memory will be disabled")
            self.langchain_llm = None
            self.enable_memory = False
        
        # Initialize conversational memory if enabled and LangChain LLM is available
        self.memory = None
        self.conversational_chain = None
        self.basic_chain = None
        
        if self.enable_memory and self.langchain_llm:
            self._initialize_memory()
        elif self.enable_memory and not self.langchain_llm:
            logger.warning("Conversational memory disabled due to LangChain LLM initialization failure")
        
        # File watcher (will be started after knowledge base setup)
        self.file_watcher_observer = None

        # Custom prompt templates
        self.prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """

        # Conversational prompt template for follow-up questions
        self.conversational_prompt_template = """Use the following pieces of context to answer the question at the end, considering the conversation history.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Be aware of previous questions and answers to provide coherent follow-up responses.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer: """
        
        logger.info("RAG Agent initialized successfully")
    
    def _initialize_memory(self):
        """Initialize conversational memory and chains"""
        try:
            if not self.enable_memory:
                return
                
            # Initialize memory with window buffer to remember last N conversations
            self.memory = ConversationBufferWindowMemory(
                k=self.memory_window_size,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Initialize chains when vector store is ready
            if hasattr(self.vector_db, 'vectorstore') and self.vector_db.vectorstore:
                self._setup_chains()
                
            logger.info(f"✅ Conversational memory initialized (window size: {self.memory_window_size})")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory: {str(e)}")
            self.enable_memory = False
    
    def _setup_chains(self):
        """Setup the LangChain retrieval chains"""
        try:
            if not self.vector_db.vectorstore:
                logger.warning("Vector store not available for chain setup")
                return
            
            if not self.langchain_llm:
                logger.warning("LangChain LLM not available - skipping chain setup")
                return
                
            retriever = self.vector_db.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
            
            # Setup basic retrieval chain for non-conversational queries
            self.basic_chain = RetrievalQA.from_chain_type(
                llm=self.langchain_llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=self.prompt_template,
                        input_variables=["context", "question"]
                    )
                }
            )
            
            # Setup conversational retrieval chain for follow-up questions
            if self.enable_memory and self.memory:
                self.conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.langchain_llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=False,
                    combine_docs_chain_kwargs={
                        "prompt": PromptTemplate(
                            template=self.conversational_prompt_template,
                            input_variables=["context", "chat_history", "question"]
                        )
                    }
                )
                
            logger.info("✅ LangChain retrieval chains initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup chains: {str(e)}")
            logger.warning("Chain setup failed - falling back to basic mode only")
            self.basic_chain = None
            self.conversational_chain = None
            self.enable_memory = False
    
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
            documents = self.document_loader.load_and_split(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
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
            
            # Setup LangChain retrieval chains now that vector store is ready
            self._setup_chains()
            
            # Start file watcher for incremental updates
            self.start_file_watcher()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup knowledge base: {str(e)}")
            return False
    

    
    def ask_question(self, question: str, k: int = 4, chat_history: Optional[List[Tuple[str, str]]] = None) -> RAGResponse:
        """
        Ask a question and get an answer from the knowledge base with optional conversational memory

        Args:
            question: The question to ask
            k: Number of relevant documents to retrieve
            chat_history: Optional chat history for conversational context

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

            # Determine if this is a conversational follow-up or a new question
            use_conversational_chain = (
                self.enable_memory and 
                self.conversational_chain and 
                (chat_history or (self.memory and len(self.memory.chat_memory.messages) > 0))
            )

            if use_conversational_chain:
                # Use conversational chain for follow-up questions
                try:
                    # If external chat history is provided, update memory
                    if chat_history and self.memory:
                        # Clear existing memory and load from chat history
                        self.memory.clear()
                        for human_msg, ai_msg in chat_history[-self.memory_window_size:]:
                            self.memory.chat_memory.add_user_message(human_msg)
                            self.memory.chat_memory.add_ai_message(ai_msg)
                    
                    # Use conversational retrieval chain
                    result = self.conversational_chain.invoke({"question": question})
                    
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])
                    
                    # Extract chat history for response
                    response_chat_history = []
                    if self.memory:
                        messages = self.memory.chat_memory.messages
                        for i in range(0, len(messages), 2):
                            if i + 1 < len(messages):
                                human = messages[i].content
                                ai = messages[i + 1].content
                                response_chat_history.append((human, ai))
                    
                    processing_time = time.time() - start_time
                    logger.info(f"Question answered using conversational chain in {processing_time:.2f} seconds")

                    return RAGResponse(
                        answer=answer,
                        source_documents=source_docs,
                        query=question,
                        processing_time=processing_time,
                        chat_history=response_chat_history
                    )
                    
                except Exception as e:
                    logger.error(f"Conversational chain detailed error: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    logger.warning(f"Conversational chain failed, falling back to basic chain: {str(e)}")
                    # Fall back to basic chain
                    use_conversational_chain = False

            if not use_conversational_chain:
                # Use basic chain for standalone questions or fallback
                if self.basic_chain:
                    try:
                        # Use basic retrieval chain
                        result = self.basic_chain.invoke({"query": question})
                        answer = result["result"]
                        source_docs = result.get("source_documents", [])
                        
                        processing_time = time.time() - start_time
                        logger.info(f"Question answered using basic chain in {processing_time:.2f} seconds")

                        return RAGResponse(
                            answer=answer,
                            source_documents=source_docs,
                            query=question,
                            processing_time=processing_time,
                            chat_history=chat_history
                        )
                        
                    except Exception as e:
                        logger.error(f"Basic chain detailed error: {type(e).__name__}: {str(e)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        logger.warning(f"LangChain approach failed, using legacy method: {str(e)}")

                # Legacy approach as final fallback
                # Get relevant documents
                scored_docs = self.vector_db.similarity_search_with_scores(question, k=k)

                if not scored_docs:
                    return RAGResponse(
                        answer="I couldn't find any relevant information to answer your question.",
                        source_documents=[],
                        query=question,
                        processing_time=time.time() - start_time,
                        chat_history=chat_history
                    )

                # Extract documents and scores
                source_docs = [doc for doc, score in scored_docs]
                scores = [score for doc, score in scored_docs]

                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in source_docs])

                # Create prompt with chat history if available
                if chat_history:
                    # Format chat history for context
                    formatted_history = "\n".join([
                        f"Human: {h}\nAssistant: {a}" 
                        for h, a in chat_history[-3:]  # Use last 3 exchanges for context
                    ])
                    prompt = self.conversational_prompt_template.format(
                        context=context, 
                        chat_history=formatted_history,
                        question=question
                    )
                else:
                    prompt = self.prompt_template.format(context=context, question=question)

                # Generate answer using LLM
                answer = self.llm.generate_response(prompt)

                processing_time = time.time() - start_time
                logger.info(f"Question answered using legacy approach in {processing_time:.2f} seconds")

                return RAGResponse(
                    answer=answer,
                    source_documents=source_docs,
                    confidence_scores=scores,
                    query=question,
                    processing_time=processing_time,
                    chat_history=chat_history
                )

        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                source_documents=[],
                query=question,
                processing_time=time.time() - start_time,
                chat_history=chat_history
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
                all_files = list(Path(self.docs_folder).rglob("*"))
                supported_files = [
                    f for f in all_files if f.is_file() and f.suffix.lower() in self.document_loader.SUPPORTED_EXTENSIONS
                ]
                vector_stats["files_available"] = len(supported_files)
                vector_stats["docs_folder"] = self.docs_folder
        except Exception:
            pass
        
        return vector_stats
    
    def clear_conversation_memory(self):
        """Clear the conversation memory"""
        try:
            if self.memory:
                self.memory.clear()
                logger.info("✅ Conversation memory cleared")
        except Exception as e:
            logger.error(f"Failed to clear conversation memory: {str(e)}")
    
    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """
        Get the current conversation history
        
        Returns:
            List of (human_message, ai_message) tuples
        """
        history = []
        try:
            if self.memory and self.memory.chat_memory:
                messages = self.memory.chat_memory.messages
                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        human = messages[i].content
                        ai = messages[i + 1].content
                        history.append((human, ai))
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
        
        return history
    
    def set_conversation_history(self, chat_history: List[Tuple[str, str]]):
        """
        Set the conversation history manually
        
        Args:
            chat_history: List of (human_message, ai_message) tuples
        """
        try:
            if self.memory:
                self.memory.clear()
                for human_msg, ai_msg in chat_history[-self.memory_window_size:]:
                    self.memory.chat_memory.add_user_message(human_msg)
                    self.memory.chat_memory.add_ai_message(ai_msg)
                logger.info(f"✅ Conversation history set with {len(chat_history)} exchanges")
        except Exception as e:
            logger.error(f"Failed to set conversation history: {str(e)}")
    
    def is_conversational_mode_enabled(self) -> bool:
        """Check if conversational mode is enabled and working"""
        return (
            self.enable_memory and 
            self.memory is not None and 
            self.conversational_chain is not None and
            self.langchain_llm is not None
        )
    
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
            loader = DocumentLoader(folder)
            new_documents = loader.load_and_split(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
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

    def add_document(self, file_path: str) -> bool:
        """
        Add a single document to the knowledge base
        
        Args:
            file_path: Path to the PDF file to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from pathlib import Path
            
            # Check if file exists
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Load the single document
            loader = DocumentLoader(str(path.parent))
            documents = loader.load_file(
                path, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            if not documents:
                logger.warning(f"No content extracted from: {file_path}")
                return False
            
            # Add to vector database
            if self.vector_db.add_documents(documents):
                logger.info(f"✅ Added document: {path.name} ({len(documents)} chunks)")
                return True
            else:
                logger.error(f"Failed to add document: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {str(e)}")
            return False

    def update_document(self, file_path: str) -> bool:
        """
        Update a document in the knowledge base
        
        Args:
            file_path: Path to the file to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from pathlib import Path
            
            # Check if file exists
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Load the updated document
            loader = DocumentLoader(str(path.parent))
            documents = loader.load_file(
                path, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            if not documents:
                logger.warning(f"No content extracted from: {file_path}")
                return False
            
            # Update in vector database (this will delete old and add new)
            if self.vector_db.update_document(str(path), documents):
                logger.info(f"✅ Updated document: {path.name} ({len(documents)} chunks)")
                return True
            else:
                logger.error(f"Failed to update document: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update document {file_path}: {str(e)}")
            return False

    def remove_document(self, file_path: str) -> bool:
        """
        Remove a document from the knowledge base
        
        Args:
            file_path: Path to the PDF file to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from pathlib import Path
            
            path = Path(file_path)
            
            # Remove from vector database by source
            if self.vector_db.delete_documents_by_source(str(path)):
                logger.info(f"✅ Removed document: {path.name}")
                return True
            else:
                logger.warning(f"Document not found in knowledge base: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove document {file_path}: {str(e)}")
            return False

    def start_file_watcher(self) -> bool:
        """
        Start the file watcher for incremental updates
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from .file_watcher import start_file_watcher
            
            if self.file_watcher_observer is not None:
                logger.warning("File watcher already running")
                return True
                
            self.file_watcher_observer = start_file_watcher(self, self.docs_folder)
            
            if self.file_watcher_observer:
                logger.info("🔄 Incremental updates enabled - file watcher started")
                return True
            else:
                logger.warning("Failed to start file watcher")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start file watcher: {str(e)}")
            return False

    def stop_file_watcher(self) -> None:
        """Stop the file watcher"""
        try:
            if hasattr(self, 'file_watcher_observer') and self.file_watcher_observer is not None:
                self.file_watcher_observer.stop()
                self.file_watcher_observer.join()
                self.file_watcher_observer = None
                logger.info("📴 File watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping file watcher: {str(e)}")

    def optimize_knowledge_base(self) -> bool:
        """
        Optimize the knowledge base for better performance
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🔧 Optimizing knowledge base...")
            
            # Check if optimization is needed
            if self.vector_db.should_rebuild_index():
                logger.info("Full index rebuild recommended")
                return self.setup_knowledge_base(force_rebuild=True)
            else:
                # Perform lightweight optimization
                return self.vector_db.optimize_index()
                
        except Exception as e:
            logger.error(f"Knowledge base optimization failed: {str(e)}")
            return False

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information
        
        Returns:
            Dictionary with system health metrics
        """
        health = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Check NVIDIA API
            try:
                test_embedding = self.embeddings.embed_query("test")
                health["components"]["nvidia_api"] = {
                    "status": "online",
                    "embedding_dimension": len(test_embedding)
                }
            except Exception as e:
                health["components"]["nvidia_api"] = {
                    "status": "offline",
                    "error": str(e)
                }
                health["overall_status"] = "degraded"
            
            # Get unified stats
            vector_stats = self.get_knowledge_base_stats()

            # Check vector database
            health["components"]["vector_database"] = {
                "status": "online" if vector_stats.get("status") == "Index loaded" else "offline",
                "document_count": vector_stats.get("document_count", 0),
                "index_exists": vector_stats.get("index_exists", False)
            }
            
            if not vector_stats.get("index_exists", False):
                health["overall_status"] = "degraded"
            
            # Check file watcher
            health["components"]["file_watcher"] = {
                "status": "active" if self.file_watcher_observer and self.file_watcher_observer.is_alive() else "inactive"
            }
            
            # Check documents folder
            try:
                docs_path = Path(self.docs_folder)
                health["components"]["documents_folder"] = {
                    "status": "accessible" if docs_path.exists() else "not_found",
                    "files_available": vector_stats.get("files_available", 0),
                    "path": str(docs_path)
                }
            except Exception as e:
                health["components"]["documents_folder"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "timestamp": time.time(),
                "overall_status": "error",
                "error": str(e)
            }

    def __del__(self):
        """Cleanup when RAGAgent is destroyed"""
        try:
            self.stop_file_watcher()
        except Exception:
            pass


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
