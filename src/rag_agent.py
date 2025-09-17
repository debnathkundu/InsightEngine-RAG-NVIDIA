"""
RAG Agent - Main pipeline for Retrieval-Augmented Generation
Combines document retrieval with question answering using NVIDIA models
"""

import os
import time
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

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
from .document_reranker import DocumentReranker, create_reranker_from_env

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
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feedback: Optional[str] = None  # "like", "dislike", None
    feedback_timestamp: Optional[datetime] = None


class FeedbackAnalytics:
    """Class to handle feedback analytics and reporting"""
    
    @staticmethod
    def get_feedback_summary(messages: List[Dict]) -> Dict[str, Any]:
        """
        Get comprehensive feedback summary from chat messages
        
        Args:
            messages: List of chat messages with potential feedback
            
        Returns:
            Dictionary with feedback analytics
        """
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        total_responses = len(assistant_messages)
        
        if total_responses == 0:
            return {
                "total_responses": 0,
                "feedback_received": 0,
                "feedback_rate": "0%",
                "liked_responses": 0,
                "disliked_responses": 0,
                "satisfaction_score": "N/A",
                "detailed_feedback": []
            }
        
        liked_responses = sum(1 for msg in assistant_messages if msg.get("feedback") == "like")
        disliked_responses = sum(1 for msg in assistant_messages if msg.get("feedback") == "dislike")
        feedback_received = liked_responses + disliked_responses
        
        feedback_rate = round((feedback_received / total_responses) * 100, 1) if total_responses > 0 else 0
        satisfaction_score = round((liked_responses / feedback_received) * 100, 1) if feedback_received > 0 else "N/A"
        
        # Detailed feedback for export
        detailed_feedback = []
        for msg in assistant_messages:
            if msg.get("feedback"):
                detailed_feedback.append({
                    "message_id": msg.get("message_id", "unknown"),
                    "question": msg.get("question", ""),
                    "answer_preview": msg.get("content", "")[:100] + "...",
                    "feedback": msg.get("feedback"),
                    "feedback_timestamp": msg.get("feedback_timestamp"),
                    "processing_time": msg.get("processing_time", 0)
                })
        
        return {
            "total_responses": total_responses,
            "feedback_received": feedback_received,
            "feedback_rate": f"{feedback_rate}%",
            "liked_responses": liked_responses,
            "disliked_responses": disliked_responses,
            "satisfaction_score": f"{satisfaction_score}%" if satisfaction_score != "N/A" else "N/A",
            "detailed_feedback": detailed_feedback,
            "trends": {
                "most_recent_feedback": detailed_feedback[-5:] if detailed_feedback else [],
                "feedback_distribution": {
                    "positive": liked_responses,
                    "negative": disliked_responses,
                    "no_feedback": total_responses - feedback_received
                }
            }
        }


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
        memory_window_size: int = 10,
        enable_reranking: bool = True,
        reranking_top_k: int = 5,
        initial_retrieval_k: int = 20
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
            enable_reranking: Whether to enable document re-ranking with cross-encoder
            reranking_top_k: Number of documents to return after re-ranking
            initial_retrieval_k: Number of documents to retrieve before re-ranking
        """
        self.docs_folder = docs_folder
        self.api_key = api_key
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_memory = enable_memory
        self.memory_window_size = memory_window_size
        self.enable_reranking = enable_reranking
        self.reranking_top_k = reranking_top_k
        self.initial_retrieval_k = initial_retrieval_k
        
        # Initialize components
        self.document_loader = DocumentLoader(docs_folder)
        self.embeddings = NVIDIAEmbeddings(api_key)
        # Enable hybrid search by default with balanced weights
        self.vector_db = VectorDatabase(
            self.embeddings, 
            vector_db_path,
            enable_hybrid_search=True,
            bm25_weight=0.3,
            vector_weight=0.7
        )
        
        # Initialize document re-ranker
        logger.info("🎯 Initializing DocumentReranker...")
        self.reranker = create_reranker_from_env()
        logger.info(f"✅ Re-ranker initialized - Available: {self.reranker.is_available()}")
        logger.info(f"   - Enable re-ranking: {self.enable_reranking}")
        logger.info(f"   - Re-ranking top_k: {self.reranking_top_k}")
        logger.info(f"   - Initial retrieval_k: {self.initial_retrieval_k}")
        if self.reranker:
            model_info = self.reranker.get_model_info()
            logger.info(f"   - Model info: {model_info}")
        
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
        
        # File watcher - start immediately to monitor for documents
        self.file_watcher_observer = None
        self.start_file_watcher()

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
        """Setup the LangChain retrieval chains with hybrid search support"""
        try:
            if not self.vector_db.vectorstore:
                logger.warning("Vector store not available for chain setup")
                return
            
            if not self.langchain_llm:
                logger.warning("LangChain LLM not available - skipping chain setup")
                return
                
            # Get the appropriate retriever (hybrid or vector-only)
            retriever = self.vector_db.get_retriever(k=4)
            
            if not retriever:
                logger.error("Failed to get retriever for chain setup")
                return
            
            # Log which type of retriever is being used
            retriever_type = "hybrid" if self.vector_db.enable_hybrid_search and self.vector_db.ensemble_retriever else "vector-only"
            logger.info(f"Setting up chains with {retriever_type} retriever")
            
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
                try:
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
                    logger.info("✅ Conversational retrieval chain initialized successfully")
                except Exception as conv_e:
                    logger.error(f"Failed to setup conversational chain: {str(conv_e)}")
                    logger.warning("Conversational chain setup failed - will use basic mode with manual memory")
                    self.conversational_chain = None
                    # Don't disable memory entirely, we can still use manual memory management
                
            logger.info("✅ LangChain retrieval chains initialized with hybrid search support")
            
        except Exception as e:
            logger.error(f"Failed to setup chains: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            logger.warning("Chain setup failed - falling back to basic mode only")
            self.basic_chain = None
            self.conversational_chain = None
            # Don't disable memory completely - we can still provide conversational experience manually
    
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
                # Check if the loaded index actually has documents
                if len(self.vector_db.documents_corpus) > 0:
                    logger.info(f"✅ Loaded existing knowledge base with {len(self.vector_db.documents_corpus)} documents")
                    return True
                else:
                    logger.info("Loaded index is empty, checking for new documents to process...")
                    force_rebuild = True  # Force rebuild since index is empty but documents might exist

            logger.info("Building knowledge base from documents...")
            
            # Load and process documents
            documents = self.document_loader.load_and_split(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            if not documents:
                logger.warning("No documents found - initializing empty knowledge base")
                # Create empty vector database structure
                if not self.vector_db.create_empty_index():
                    logger.error("Failed to create empty vector index")
                    return False
                
                # Save empty index
                if not self.vector_db.save_index():
                    logger.error("Failed to save empty vector index")
                    return False
                
                logger.info("✅ Empty knowledge base initialized - ready to receive documents")
                
                return True
            
            # Get document stats
            stats = self.document_loader.get_document_stats(documents)
            logger.info(f"Loaded {stats['num_source_files']} files with {stats['total_chunks']} chunks")
            
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
            
            # Check if knowledge base has any documents
            stats = self.get_knowledge_base_stats()
            if stats.get('total_documents', 0) == 0:
                return RAGResponse(
                    answer="I don't have any documents in my knowledge base yet. Please add some documents (PDF, DOCX, PPTX, or TXT files) to the Data/Docs folder, or use the Web Import feature to import documents from URLs, then I'll be able to answer questions about them.",
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

                # Legacy approach as final fallback - now with hybrid search and re-ranking support
                try:
                    # Use the enhanced get_relevant_documents method with re-ranking
                    scored_docs = self.get_relevant_documents(
                        query=question, 
                        k=k, 
                        use_hybrid=True, 
                        use_reranking=True
                    )
                    source_docs = [doc for doc, score in scored_docs]
                    scores = [score for doc, score in scored_docs]
                    
                    # Log what retrieval method was used
                    retrieval_method = []
                    if self.vector_db.enable_hybrid_search and self.vector_db.ensemble_retriever:
                        retrieval_method.append("hybrid")
                    else:
                        retrieval_method.append("vector")
                    
                    if self.enable_reranking and self.reranker.is_available():
                        retrieval_method.append("re-ranked")
                    
                    logger.info(f"Retrieved {len(source_docs)} documents using {'+'.join(retrieval_method)} search")
                        
                except Exception as e:
                    logger.warning(f"Enhanced retrieval failed, using basic fallback: {str(e)}")
                    # Basic fallback to vector search
                    scored_docs = self.vector_db.similarity_search_with_scores(question, k=k)
                    source_docs = [doc for doc, score in scored_docs]
                    scores = [score for doc, score in scored_docs]

                if not source_docs:
                    return RAGResponse(
                        answer="I couldn't find any relevant information to answer your question.",
                        source_documents=[],
                        query=question,
                        processing_time=time.time() - start_time,
                        chat_history=chat_history
                    )

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
                
                # Build search method description
                search_methods = []
                if self.vector_db.enable_hybrid_search and self.vector_db.ensemble_retriever:
                    search_methods.append("hybrid")
                else:
                    search_methods.append("vector")
                
                if self.enable_reranking and self.reranker.is_available():
                    search_methods.append("re-ranked")
                
                search_method = "+".join(search_methods)
                logger.info(f"Question answered using legacy approach with {search_method} search in {processing_time:.2f} seconds")

                return RAGResponse(
                    answer=answer,
                    source_documents=source_docs,
                    confidence_scores=scores,  # May be None for hybrid search
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
    
    def get_relevant_documents(self, query: str, k: int = 4, use_hybrid: bool = True, use_reranking: bool = None) -> List[Tuple[Document, float]]:
        """
        Get relevant documents for a query with similarity scores and optional re-ranking
        
        Args:
            query: Search query
            k: Number of documents to retrieve (final number after re-ranking)
            use_hybrid: Whether to use hybrid search (if available)
            use_reranking: Whether to use re-ranking (defaults to instance setting)
            
        Returns:
            List of (document, score) tuples. 
            - For hybrid search without reranking: (document, None) tuples
            - For vector search: (document, similarity_score) tuples  
            - For reranked results: (document, relevance_score) tuples
        """
        try:
            logger.info(f"🔍 get_relevant_documents() called:")
            logger.info(f"   - Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            logger.info(f"   - Requested k: {k}")
            logger.info(f"   - Use hybrid: {use_hybrid}")
            logger.info(f"   - Use reranking param: {use_reranking}")
            
            # Determine if we should use re-ranking
            should_rerank = use_reranking if use_reranking is not None else (
                self.enable_reranking and self.reranker.is_available()
            )
            
            logger.info(f"🎯 Re-ranking decision:")
            logger.info(f"   - Should re-rank: {should_rerank}")
            logger.info(f"   - Instance enable_reranking: {self.enable_reranking}")
            logger.info(f"   - Reranker available: {self.reranker.is_available() if self.reranker else 'None'}")
            
            # Determine how many documents to retrieve initially
            initial_k = self.initial_retrieval_k if should_rerank else k
            logger.info(f"📊 Retrieval plan:")
            logger.info(f"   - Initial retrieval k: {initial_k}")
            logger.info(f"   - Final k after re-ranking: {k}")
            
            # Step 1: Initial retrieval using hybrid or vector search
            logger.info("🔎 Step 1: Initial document retrieval...")
            
            # Try to load vector database if not already loaded
            if not self.vector_db.vectorstore:
                logger.info("   - Vector database not loaded, attempting to load...")
                if not self.vector_db.load_index():
                    logger.warning("   - Failed to load vector database index")
                else:
                    logger.info("   - Vector database loaded successfully")
            
            if use_hybrid and self.vector_db.enable_hybrid_search and self.vector_db.ensemble_retriever:
                # Use hybrid search (scores not available)
                logger.info(f"   - Using hybrid search (BM25 + Vector)")
                initial_docs = self.vector_db.hybrid_search(query, k=initial_k)
                initial_docs_with_scores = [(doc, None) for doc in initial_docs]
                logger.info(f"   - Retrieved {len(initial_docs)} documents via hybrid search")
            else:
                # Use vector search with scores
                logger.info(f"   - Using vector search only")
                initial_docs_with_scores = self.vector_db.similarity_search_with_scores(query, k=initial_k)
                initial_docs = [doc for doc, score in initial_docs_with_scores]
                logger.info(f"   - Retrieved {len(initial_docs)} documents via vector search")
            
            # Step 2: Apply re-ranking if enabled and available
            logger.info("🎯 Step 2: Re-ranking evaluation...")
            if should_rerank and initial_docs:
                logger.info(f"✅ Proceeding with re-ranking:")
                logger.info(f"   - Input documents: {len(initial_docs)}")
                logger.info(f"   - Target output: {k}")
                
                reranked_docs, rerank_scores = self.reranker.rerank_documents(
                    query=query,
                    documents=initial_docs,
                    top_k=k,
                    initial_k=len(initial_docs)
                )
                
                logger.info(f"✅ Re-ranking completed - returning {len(reranked_docs)} documents")
                return [(doc, score) for doc, score in zip(reranked_docs, rerank_scores)]
            else:
                if not should_rerank:
                    logger.info("❌ Re-ranking skipped - not enabled or not available")
                elif not initial_docs:
                    logger.info("❌ Re-ranking skipped - no initial documents")
                
                logger.info(f"🔄 Returning initial results (limited to {k})")
                # Return initial results (limited to k)
                result = initial_docs_with_scores[:k]
                logger.info(f"   - Returning {len(result)} documents without re-ranking")
                return result
                
        except Exception as e:
            logger.error(f"Error getting relevant documents: {str(e)}")
            # Fall back to vector search without re-ranking
            try:
                return self.vector_db.similarity_search_with_scores(query, k=k)
            except Exception as fallback_e:
                logger.error(f"Fallback also failed: {str(fallback_e)}")
                return []
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with knowledge base statistics
        """
        vector_stats = self.vector_db.get_stats()
        
        # Normalize document count fields - use the most reliable source
        documents_in_corpus = vector_stats.get("documents_in_corpus", 0)
        doc_count = vector_stats.get("document_count", 0)
        
        # Use documents_in_corpus as the primary source of truth
        total_docs = max(documents_in_corpus, doc_count)
        vector_stats["total_documents"] = total_docs
        vector_stats["total_chunks"] = total_docs  # Each document in corpus represents a chunk
        
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
        # Conversational mode is enabled if we have memory (even without LangChain conversational chain)
        # We can provide conversational experience through manual memory management
        return (
            self.enable_memory and 
            self.memory is not None and 
            self.langchain_llm is not None
        )
    
    def is_hybrid_search_enabled(self) -> bool:
        """Check if hybrid search is enabled and working"""
        return (
            self.vector_db.enable_hybrid_search and 
            self.vector_db.ensemble_retriever is not None
        )
    
    def is_reranking_enabled(self) -> bool:
        """Check if re-ranking is enabled and working"""
        return (
            self.enable_reranking and 
            self.reranker is not None and 
            self.reranker.is_available()
        )
    
    def configure_hybrid_search(self, bm25_weight: float = 0.3, vector_weight: float = 0.7) -> bool:
        """
        Configure hybrid search weights
        
        Args:
            bm25_weight: Weight for BM25 retriever (0.0-1.0)
            vector_weight: Weight for vector retriever (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate weights
            if not (0.0 <= bm25_weight <= 1.0) or not (0.0 <= vector_weight <= 1.0):
                logger.error("Weights must be between 0.0 and 1.0")
                return False
            
            if abs((bm25_weight + vector_weight) - 1.0) > 0.01:
                logger.warning(f"Weights don't sum to 1.0: BM25={bm25_weight}, Vector={vector_weight}")
            
            # Update weights
            self.vector_db.bm25_weight = bm25_weight
            self.vector_db.vector_weight = vector_weight
            
            # Recreate ensemble retriever if it exists
            if self.vector_db.ensemble_retriever:
                self.vector_db._create_ensemble_retriever()
                
                # Recreate chains to use updated retriever
                self._setup_chains()
            
            logger.info(f"✅ Hybrid search weights updated: BM25={bm25_weight}, Vector={vector_weight}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure hybrid search: {str(e)}")
            return False
    
    def configure_reranking(self, enable: bool = None, top_k: int = None, initial_k: int = None) -> bool:
        """
        Configure re-ranking settings
        
        Args:
            enable: Whether to enable re-ranking
            top_k: Number of documents to return after re-ranking
            initial_k: Number of documents to retrieve before re-ranking
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if enable is not None:
                old_setting = self.enable_reranking
                self.enable_reranking = enable
                if old_setting != enable:
                    logger.info(f"✅ Re-ranking {'enabled' if enable else 'disabled'}")
            
            if top_k is not None:
                if top_k > 0:
                    self.reranking_top_k = top_k
                    logger.info(f"✅ Re-ranking top_k updated to: {top_k}")
                else:
                    logger.error("top_k must be greater than 0")
                    return False
            
            if initial_k is not None:
                if initial_k > 0:
                    self.initial_retrieval_k = initial_k
                    logger.info(f"✅ Initial retrieval k updated to: {initial_k}")
                else:
                    logger.error("initial_k must be greater than 0")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure re-ranking: {str(e)}")
            return False
    
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
            
            # Check re-ranker
            health["components"]["reranker"] = {
                "status": "available" if self.reranker.is_available() else "unavailable",
                "enabled": self.enable_reranking,
                "model_info": self.reranker.get_model_info() if self.reranker else {}
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
