"""
DIFC Legal RAG - Streamlit Web Interface
Beautiful, user-friendly web interface for the DIFC Legal RAG system
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_agent import RAGAgent
from src.nvidia_embeddings import NVIDIAEmbeddings

# Page configuration
st.set_page_config(
    page_title="RAG Assistant - NVIDIA NemoRetriever",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #7b1fa2;
    }
    
    .source-card {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #4caf50;
    }
    
    .status-offline {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_agent():
    """Initialize the RAG agent (cached for performance)"""
    try:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")

        if not api_key:
            st.error("❌ NVIDIA_API_KEY not found in environment variables")
            st.info("Please check your .env file and ensure the API key is set correctly.")
            return None

        # Test NVIDIA API connection first
        with st.spinner("🔌 Testing NVIDIA API connection..."):
            try:
                embeddings = NVIDIAEmbeddings(api_key)
                if not embeddings.test_connection():
                    st.error("❌ Failed to connect to NVIDIA API")
                    return None
            except Exception as e:
                st.error(f"❌ NVIDIA API connection failed: {str(e)}")
                return None

        # Initialize RAG agent
        with st.spinner("🤖 Initializing RAG Agent..."):
            rag_agent = RAGAgent(docs_folder, api_key)

        # Setup knowledge base
        with st.spinner("📚 Loading knowledge base..."):
            if rag_agent.setup_knowledge_base():
                st.success("✅ RAG system initialized successfully!")
                return rag_agent
            else:
                st.error("❌ Failed to setup knowledge base")
                st.info("Please ensure PDF files are in the Data/Docs folder.")
                return None

    except Exception as e:
        st.error(f"❌ Failed to initialize RAG agent: {str(e)}")
        st.exception(e)
        return None

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Assistant - NVIDIA NemoRetriever</h1>
        <p>AI-Powered Document Q&A System with Advanced Retrieval</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar(rag_agent):
    """Display the sidebar with system information"""
    st.sidebar.markdown("## 📊 System Status")
    
    if rag_agent:
        # System status
        st.sidebar.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="status-indicator status-online"></span>
            <strong>System Online</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Get knowledge base stats
        stats = rag_agent.get_knowledge_base_stats()
        
        st.sidebar.markdown("### 📚 Knowledge Base")
        st.sidebar.metric("Documents", stats.get('document_count', 0))
        st.sidebar.metric("PDF Files", stats.get('pdf_files_available', 0))
        
        # Model information
        st.sidebar.markdown("### 🤖 AI Models")
        st.sidebar.info("**Embedding**: nvidia/nv-embed-v1\n**LLM**: meta/llama-3.1-8b-instruct")
        
        # Document types supported
        st.sidebar.markdown("### 📖 Document Types Supported")
        doc_types = [
            "PDF Documents", "Research Papers", "Legal Documents",
            "Technical Manuals", "Corporate Policies", "Academic Papers",
            "Training Materials", "Compliance Documents", "Reports",
            "Contracts", "Specifications", "User Guides"
        ]

        for doc_type in doc_types:
            st.sidebar.markdown(f"• {doc_type}")
            
    else:
        st.sidebar.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="status-indicator status-offline"></span>
            <strong>System Offline</strong>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.error("RAG system not available")

def display_chat_interface(rag_agent):
    """Display the main chat interface"""
    st.markdown("## 💬 Ask Your Question")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your RAG Assistant powered by NVIDIA NemoRetriever. I can help you find information from your document collection. What would you like to know?",
            "sources": [],
            "processing_time": 0
        })
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 AI Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources if available
            if message.get("sources"):
                display_sources(message["sources"], message.get("processing_time", 0))
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {prompt}
        </div>
        """, unsafe_allow_html=True)
        
        if rag_agent:
            try:
                # Show loading spinner
                with st.spinner("🔍 Searching documents..."):
                    # Get response from RAG agent
                    response = rag_agent.ask_question(prompt)

                # Validate response
                if not response or not response.answer:
                    st.error("❌ Failed to get a response. Please try again.")
                    return

                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.source_documents,
                    "processing_time": response.processing_time
                })

                # Display assistant response
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {response.answer}
                </div>
                """, unsafe_allow_html=True)

                # Display sources
                display_sources(response.source_documents, response.processing_time)

            except Exception as e:
                st.error(f"❌ Error processing your question: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                    "sources": [],
                    "processing_time": 0
                })

        else:
            st.error("❌ RAG system not available. Please check the system status.")
            st.info("Try refreshing the page or contact support if the issue persists.")

def display_sources(source_documents, processing_time, confidence_scores=None):
    """Display source documents in an elegant format"""
    if not source_documents:
        st.info("ℹ️ No specific sources found for this response.")
        return

    # Create a container for sources
    with st.container():
        st.markdown("---")

        # Header with metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
        with col2:
            st.metric("📄 Sources Found", len(source_documents))
        with col3:
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                st.metric("🎯 Avg. Relevance", f"{avg_confidence:.2f}")

        st.markdown("### 📚 Sources & References")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📄 Document Sources", "📊 Source Analysis", "🔍 Quick Search"])

        with tab1:
            for i, doc in enumerate(source_documents, 1):
                source_file = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                chunk_id = doc.metadata.get("chunk_id", "Unknown")

                # Clean up filename for display
                display_name = source_file.replace("_", " ").replace("-", " ").title()
                if display_name.endswith(".pdf"):
                    display_name = display_name[:-4]

                # Confidence indicator
                confidence_indicator = ""
                if confidence_scores and i <= len(confidence_scores):
                    score = confidence_scores[i-1]
                    if score > 0.8:
                        confidence_indicator = "🟢 High Relevance"
                    elif score > 0.6:
                        confidence_indicator = "🟡 Medium Relevance"
                    else:
                        confidence_indicator = "🔴 Low Relevance"

                with st.expander(f"📖 Source {i}: {display_name} (Page {page}) {confidence_indicator}"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**📁 File**: {source_file}")
                        st.markdown(f"**📄 Page**: {page}")
                        st.markdown(f"**🔢 Chunk ID**: {chunk_id}")
                        if confidence_scores and i <= len(confidence_scores):
                            st.markdown(f"**🎯 Relevance Score**: {confidence_scores[i-1]:.3f}")

                    with col2:
                        # Quick actions
                        if st.button(f"📋 Copy Text {i}", key=f"copy_{i}"):
                            st.code(doc.page_content, language="text")

                    st.markdown("**📝 Content Preview**:")
                    # Better text formatting
                    preview_text = doc.page_content[:800]
                    if len(doc.page_content) > 800:
                        preview_text += "..."

                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 3px solid #2a5298;">
                        {preview_text}
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            # Create source distribution chart
            source_files = [doc.metadata.get("source_file", "Unknown") for doc in source_documents]
            file_counts = {}
            for file in source_files:
                # Clean filename for chart
                clean_name = file.replace("_", " ").replace("-", " ")
                if clean_name.endswith(".pdf"):
                    clean_name = clean_name[:-4]
                file_counts[clean_name] = file_counts.get(clean_name, 0) + 1

            if file_counts:
                # Pie chart for source distribution
                fig_pie = px.pie(
                    values=list(file_counts.values()),
                    names=list(file_counts.keys()),
                    title="📊 Source Document Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

                # Bar chart for page distribution
                pages = [doc.metadata.get("page", 0) for doc in source_documents]
                if pages:
                    fig_bar = px.histogram(
                        x=pages,
                        title="📄 Page Distribution of Sources",
                        labels={'x': 'Page Number', 'y': 'Number of Sources'},
                        color_discrete_sequence=['#2a5298']
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.markdown("#### 🔍 Search Within Sources")
            search_term = st.text_input("Search for specific terms in the source documents:")

            if search_term:
                matches = []
                for i, doc in enumerate(source_documents, 1):
                    if search_term.lower() in doc.page_content.lower():
                        # Find the context around the search term
                        content = doc.page_content.lower()
                        index = content.find(search_term.lower())
                        start = max(0, index - 100)
                        end = min(len(content), index + len(search_term) + 100)
                        context = doc.page_content[start:end]

                        matches.append({
                            'source': i,
                            'file': doc.metadata.get("source_file", "Unknown"),
                            'page': doc.metadata.get("page", "Unknown"),
                            'context': context
                        })

                if matches:
                    st.success(f"Found {len(matches)} matches for '{search_term}':")
                    for match in matches:
                        st.markdown(f"""
                        **Source {match['source']}** - {match['file']} (Page {match['page']})
                        > ...{match['context']}...
                        """)
                else:
                    st.info(f"No matches found for '{search_term}' in the source documents.")

def display_document_stats(rag_agent):
    """Display detailed document statistics"""
    if not rag_agent:
        st.error("RAG system not available")
        return

    st.markdown("## 📊 Document Statistics")

    stats = rag_agent.get_knowledge_base_stats()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Documents", stats.get('document_count', 0))
    with col2:
        st.metric("📁 PDF Files", stats.get('pdf_files_available', 0))
    with col3:
        st.metric("🔍 Searchable Chunks", stats.get('document_count', 0))
    with col4:
        st.metric("💾 Index Size", "Ready" if stats.get('index_exists') else "Not Found")

    # Document types breakdown
    st.markdown("### 📖 Document Types & Use Cases")
    doc_categories = {
        "Research Papers": "📚",
        "Technical Documentation": "🔧",
        "Legal Documents": "⚖️",
        "Corporate Policies": "🏢",
        "Training Materials": "🎓",
        "User Manuals": "📖",
        "Compliance Documents": "✅",
        "Reports & Analysis": "📊",
        "Contracts & Agreements": "📝",
        "Specifications": "🔍",
        "Academic Papers": "🎓",
        "Reference Materials": "📚"
    }

    cols = st.columns(3)
    for i, (category, emoji) in enumerate(doc_categories.items()):
        with cols[i % 3]:
            st.markdown(f"{emoji} **{category}**")

    # System health
    st.markdown("### 🔧 System Health")
    health_col1, health_col2 = st.columns(2)

    with health_col1:
        st.markdown("**API Status**")
        if rag_agent:
            st.success("🟢 NVIDIA API Connected")
            st.success("🟢 Vector Database Loaded")
            st.success("🟢 LLM Model Ready")
        else:
            st.error("🔴 System Offline")

    with health_col2:
        st.markdown("**Performance Metrics**")
        st.info("📊 Embedding Dimension: 4096")
        st.info("🚀 Average Query Time: 2-8 seconds")
        st.info("💾 Storage: Local FAISS Index")

def main():
    """Main application function"""
    # Display header
    display_header()

    # Initialize RAG agent
    rag_agent = initialize_rag_agent()

    # Display sidebar
    display_sidebar(rag_agent)

    # Create tabs for different views
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Document Statistics"])

    with tab1:
        # Main content area
        col1, col2 = st.columns([3, 1])

        with col1:
            # Chat interface
            display_chat_interface(rag_agent)
    
    with col2:
        # Quick actions and tips
        st.markdown("### 💡 Quick Tips")
        st.info("""
        **Sample Questions:**
        • What is the main topic of the documents?
        • Summarize the key points
        • What are the requirements mentioned?
        • How does [concept A] relate to [concept B]?
        • What are the benefits described?
        • Explain the process for [specific topic]
        """)

        # Advanced features
        st.markdown("### 🛠️ Actions")

        # Export chat history
        if st.button("📥 Export Chat History"):
            if st.session_state.messages:
                chat_export = []
                for msg in st.session_state.messages:
                    chat_export.append({
                        "timestamp": datetime.now().isoformat(),
                        "role": msg["role"],
                        "content": msg["content"],
                        "sources_count": len(msg.get("sources", [])),
                        "processing_time": msg.get("processing_time", 0)
                    })

                import json
                export_data = json.dumps(chat_export, indent=2)
                st.download_button(
                    label="💾 Download Chat History",
                    data=export_data,
                    file_name=f"difc_legal_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No chat history to export")

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Knowledge base rebuild
        if st.button("🔄 Rebuild Knowledge Base"):
            st.cache_resource.clear()
            st.rerun()

        # System information
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI assistant is powered by:
        - **NVIDIA** embedding models
        - **Meta LLaMA** language model
        - **FAISS** vector database
        - **1,869** legal document chunks
        """)

        # Chat statistics
        if st.session_state.messages:
            st.markdown("### 📈 Session Stats")
            user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
            assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]

            st.metric("Questions Asked", len(user_messages))
            st.metric("Responses Given", len(assistant_messages))

            if assistant_messages:
                avg_time = sum(m.get("processing_time", 0) for m in assistant_messages) / len(assistant_messages)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")

    with tab2:
        # Document statistics page
        display_document_stats(rag_agent)

if __name__ == "__main__":
    main()
