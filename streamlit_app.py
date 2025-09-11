"""
Enterprise RAG - Streamlit Web Interface
Beautiful, user-friendly web interface for the Enterprise RAG system
"""

import streamlit as st
import sys
import os
import time
import json
import uuid
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from typing import Optional

# Add project root to path, not the src folder
sys.path.append(str(Path(__file__).parent))

from src.rag_agent import RAGAgent, FeedbackAnalytics
from src.file_watcher import start_file_watcher, get_pending_notifications
from src.nvidia_embeddings import NVIDIAEmbeddings
from src.web_importer import WebImporter

def handle_feedback(message_id: str, feedback_type: str):
    """Handle user feedback on a message"""
    if "messages" not in st.session_state:
        return
    
    # Find and update the message with feedback
    for message in st.session_state.messages:
        if message.get("message_id") == message_id and message.get("role") == "assistant":
            message["feedback"] = feedback_type
            message["feedback_timestamp"] = datetime.now().isoformat()
            
            # Store the feedback confirmation to show after rerun
            st.session_state[f"feedback_confirmed_{message_id}"] = feedback_type
            
            # Trigger rerun to update UI
            st.rerun()
            break

def render_feedback_buttons(message_id: str, current_feedback: Optional[str] = None):
    """Render feedback buttons for a message"""
    # Check if feedback was just confirmed
    feedback_confirmed = st.session_state.get(f"feedback_confirmed_{message_id}")
    if feedback_confirmed:
        # Show confirmation message
        if feedback_confirmed == "like":
            st.success("👍 Thank you for the positive feedback!", icon="✅")
        else:
            st.info("👎 Thank you for the feedback. We'll use this to improve!", icon="💡")
        # Clear the confirmation
        del st.session_state[f"feedback_confirmed_{message_id}"]
    
    col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
    
    with col1:
        # Like button
        like_style = "🟢👍" if current_feedback == "like" else "👍"
        if st.button(like_style, key=f"like_{message_id}", help="Like this response"):
            handle_feedback(message_id, "like")
    
    with col2:
        # Dislike button  
        dislike_style = "🔴👎" if current_feedback == "dislike" else "👎"
        if st.button(dislike_style, key=f"dislike_{message_id}", help="Dislike this response"):
            handle_feedback(message_id, "dislike")
    
    with col3:
        # Show feedback status
        if current_feedback:
            feedback_text = "Liked ✅" if current_feedback == "like" else "Disliked ❌"
            st.caption(f"Feedback: {feedback_text}")
        else:
            st.caption("👆 Rate this response")

def add_system_notification(notification_type: str, filename: str = "", details: dict = None):
    """Add a system notification to the session state"""
    if "system_notifications" not in st.session_state:
        st.session_state.system_notifications = []
    
    notification = {
        "type": notification_type,
        "filename": filename,
        "details": details or {},
        "timestamp": time.time()
    }
    
    st.session_state.system_notifications.append(notification)

def process_file_watcher_notifications():
    """Process any pending notifications from the file watcher"""
    try:
        notifications = get_pending_notifications()
        for notification in notifications:
            # Convert file watcher notification to system notification
            add_system_notification(
                notification["type"],
                notification["filename"],
                notification["details"]
            )
            
            # Also add vector store update messages to chat timeline
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Create detailed chat message for vector store updates
            timestamp = datetime.now()
            time_str = timestamp.strftime("%H:%M:%S")
            filename = notification["filename"]
            details = notification.get("details", {})
            
            if notification["type"] == "document_added":
                chat_message = f"**📥 Vector Store Update [{time_str}]**\n\n"
                chat_message += f"**Status**: DOCUMENT ADDED\n"
                chat_message += f"**File**: `{filename}`\n"
                chat_message += f"**Details**: Document has been processed and added to the vector database."
                
            elif notification["type"] == "document_removed":
                chat_message = f"**🗑️ Vector Store Update [{time_str}]**\n\n"
                chat_message += f"**Status**: DOCUMENT REMOVED\n"
                chat_message += f"**File**: `{filename}`\n"
                chat_message += f"**Details**: Document has been removed from the vector database."
                
            elif notification["type"] == "document_updated":
                chat_message = f"**🔄 Vector Store Update [{time_str}]**\n\n"
                chat_message += f"**Status**: DOCUMENT UPDATED\n"
                chat_message += f"**File**: `{filename}`\n"
                chat_message += f"**Details**: Document has been updated in the vector database."
                
            elif notification["type"] == "vector_db_updated":
                operations = details.get("operations", 0)
                deleted = details.get("deleted", 0)
                added = details.get("added", 0)
                updated = details.get("updated", 0)
                
                chat_message = f"**💾 Vector Store Batch Update [{time_str}]**\n\n"
                chat_message += f"**Status**: BATCH OPERATION COMPLETE\n"
                chat_message += f"**Total Operations**: {operations}\n"
                if added > 0:
                    chat_message += f"**Documents Added**: {added}\n"
                if updated > 0:
                    chat_message += f"**Documents Updated**: {updated}\n"
                if deleted > 0:
                    chat_message += f"**Documents Removed**: {deleted}\n"
                chat_message += f"**Details**: Vector database has been updated with batch file operations."
            else:
                # Skip other notification types for chat timeline
                continue
            
            # Add the vector store update message to chat
            st.session_state.messages.append({
                "role": "system",
                "content": chat_message,
                "timestamp": timestamp,
                "type": "vector_store_update",
                "status_type": notification["type"]
            })
            
    except Exception as e:
        # Silently handle errors to avoid disrupting the main app
        pass

def status_callback(update):
    """Thread-safe callback function to handle vector database status updates"""
    try:
        # Add status update directly to chat messages for chronological flow
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Format status message for chat timeline
        time_str = update.timestamp.strftime("%H:%M:%S")
        file_info = f" - {Path(update.file_path).name}" if hasattr(update, 'file_path') and update.file_path else ""
        
        # Choose icon and create detailed status text based on status
        if hasattr(update, 'status'):
            status_str = str(update.status).upper()
            if "ERROR" in status_str:
                icon = "🔴"
                status_text = "ERROR"
                detail_text = "An error occurred during the vector database operation."
            elif "ADDING" in status_str:
                icon = "📥"
                status_text = "ADDING DOCUMENT"
                detail_text = "Processing and adding new document to the knowledge base."
            elif "UPDATING" in status_str:
                icon = "🔄"
                status_text = "UPDATING DOCUMENT"
                detail_text = "Updating existing document in the knowledge base."
            elif "REMOVING" in status_str:
                icon = "🗑️"
                status_text = "REMOVING DOCUMENT"
                detail_text = "Removing document from the vector database."
            elif "REBUILDING" in status_str:
                icon = "🔨"
                status_text = "REBUILDING DATABASE"
                detail_text = "Rebuilding the entire knowledge base from scratch."
            else:  # IDLE or COMPLETE
                icon = "🟢"
                status_text = "OPERATION COMPLETE"
                detail_text = "Vector database operation completed successfully."
        else:
            icon = "🟡"
            status_text = "VECTOR DB UPDATE"
            detail_text = "Vector database operation in progress."
        
        # Create comprehensive status message for chat
        status_message = f"**{icon} Vector Database Update [{time_str}]**\n\n"
        status_message += f"**Status**: {status_text}\n"
        status_message += f"**Operation**: {update.message}{file_info}\n"
        status_message += f"**Details**: {detail_text}"
        
        # Add progress info if available
        if hasattr(update, 'progress') and update.progress is not None:
            progress_percent = update.progress * 100
            progress_bar = "█" * int(update.progress * 20)
            spaces = "░" * (20 - int(update.progress * 20))
            status_message += f"\n\n**Progress**: [{progress_bar}{spaces}] {progress_percent:.1f}%"
        
        # Add file path info if available
        if hasattr(update, 'file_path') and update.file_path:
            status_message += f"\n**File**: `{Path(update.file_path).name}`"
        
        # Add as a system message in chat with vector DB status type
        st.session_state.messages.append({
            "role": "system",
            "content": status_message,
            "timestamp": update.timestamp,
            "type": "vector_db_status",
            "status_type": getattr(update, 'status', 'unknown')
        })
        
    except Exception as e:
        # Log error but don't crash the application
        print(f"Status callback error: {e}")

def display_system_message(notification_type: str, filename: str = "", details: dict = None):
    """Display system messages (file operations, indexing updates, etc.)"""
    if details is None:
        details = {}
    
    # Create content based on notification type
    if notification_type == "document_added":
        icon = "📄➕"
        bg_color = "#e8f5e8"
        border_color = "#4caf50"
        content = f"Document added: {filename}"
    elif notification_type == "document_removed":
        icon = "📄🗑️"
        bg_color = "#fff3e0"
        border_color = "#ff9800"
        content = f"Document removed: {filename}"
    elif notification_type == "document_updated":
        icon = "📄🔄"
        bg_color = "#e3f2fd"
        border_color = "#2196f3"
        content = f"Document updated: {filename}"
    elif notification_type == "vector_db_updated":
        icon = "�🔄"
        bg_color = "#f3e5f5"
        border_color = "#9c27b0"
        ops = details.get("operations", 0)
        content = f"Vector database updated ({ops} operations)"
    elif notification_type == "vector_db_rebuilt":
        icon = "🏗️🔄"
        bg_color = "#fce4ec"
        border_color = "#e91e63"
        docs = details.get("documents", 0)
        time_taken = details.get("time_taken", 0)
        content = f"Vector database rebuilt ({docs} documents, {time_taken:.1f}s)"
    elif notification_type == "index_optimized":
        icon = "⚡🔧"
        bg_color = "#f3e5f5"
        border_color = "#9c27b0"
        docs = details.get("documents", 0)
        time_taken = details.get("time_taken", 0)
        content = f"Index optimized ({docs} documents, {time_taken:.1f}s)"
    elif notification_type == "system_init":
        icon = "🚀"
        bg_color = "#e8f5e8"
        border_color = "#4caf50"
        content = "RAG Assistant initialized successfully!"
    elif notification_type == "memory_update":
        icon = "🧠🗑️"
        bg_color = "#e8f5e8"
        border_color = "#4caf50"
        status_type = details.get("status_type", "")
        if status_type == "memory_cleared":
            content = "Conversation memory has been cleared. Future questions will be treated as new conversations."
        else:
            content = "Memory updated"
    elif notification_type == "error":
        icon = "⚠️"
        bg_color = "#ffebee"
        border_color = "#f44336"
        error_msg = details.get("error", "Unknown error")
        content = f"Error: {error_msg}"
        if filename:
            content = f"Error with {filename}: {error_msg}"
    else:
        icon = "ℹ️"
        bg_color = "#f5f5f5"
        border_color = "#757575"
        content = filename if filename else "System notification"
    
    time_str = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color}; 
        border-left: 4px solid {border_color}; 
        padding: 0.8rem; 
        margin: 0.5rem 0; 
        border-radius: 8px;
        font-size: 0.9rem;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span><strong>{icon} System:</strong> {content}</span>
            <small style="color: #666;">{time_str}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        position: relative;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #7b1fa2;
        margin-right: 2rem;
    }
    
    .system-message {
        background-color: #fff8e1;
        border-left-color: #ff9800;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        padding: 0.8rem;
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
    
    .status-updating {
        background-color: #fff3cd;
        border-left-color: #ffc107 !important;
    }
    
    .status-error {
        background-color: #f8d7da;
        border-left-color: #dc3545 !important;
    }
    
    .status-offline {
        background-color: #f44336;
    }
    
    /* Chat chronology improvements */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
    }
    
    .timestamp {
        font-size: 0.8rem;
        color: #666;
        opacity: 0.8;
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
            try:
                # Try to initialize with status callback for vector DB notifications
                rag_agent = RAGAgent(docs_folder, api_key, status_callback=status_callback)
            except TypeError:
                # Fallback if status_callback parameter is not supported
                rag_agent = RAGAgent(docs_folder, api_key)

        # Setup knowledge base
        with st.spinner("📚 Loading knowledge base..."):
            if rag_agent.setup_knowledge_base():
                st.success("✅ RAG system initialized successfully!")
                
                # Add initialization message to chat timeline
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                init_message = {
                    "role": "system",
                    "content": "🚀 **System Initialization Complete**\n\nRAG Assistant has been successfully initialized and is ready to answer your questions about your document collection!",
                    "timestamp": datetime.now(),
                    "type": "system_init"
                }
                st.session_state.messages.append(init_message)
                
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

def display_feedback_analysis():
    """Display comprehensive feedback analysis"""
    st.header("📝 Feedback Analysis Dashboard")
    
    if "messages" not in st.session_state or not st.session_state.messages:
        st.info("💬 No chat messages yet. Start a conversation to see feedback analytics!")
        return
    
    # Get feedback summary using FeedbackAnalytics
    feedback_summary = FeedbackAnalytics.get_feedback_summary(st.session_state.messages)
    
    # Overview metrics
    st.markdown("### 📊 Feedback Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Responses", 
            feedback_summary["total_responses"],
            help="Total number of AI responses provided"
        )
    
    with col2:
        st.metric(
            "Feedback Received", 
            feedback_summary["feedback_received"],
            help="Number of responses that received user feedback"
        )
    
    with col3:
        st.metric(
            "Feedback Rate", 
            feedback_summary["feedback_rate"],
            help="Percentage of responses that received feedback"
        )
    
    with col4:
        st.metric(
            "Satisfaction Score", 
            feedback_summary["satisfaction_score"],
            help="Percentage of positive feedback (likes/(likes+dislikes))"
        )
    
    # Feedback distribution chart
    if feedback_summary["feedback_received"] > 0:
        st.markdown("### 📈 Feedback Distribution")
        
        # Create pie chart for feedback distribution
        feedback_data = feedback_summary["trends"]["feedback_distribution"]
        
        fig = go.Figure(data=[go.Pie(
            labels=['👍 Positive', '👎 Negative', '⚪ No Feedback'],
            values=[feedback_data["positive"], feedback_data["negative"], feedback_data["no_feedback"]],
            hole=.3,
            marker_colors=['#2E8B57', '#DC143C', '#D3D3D3']
        )])
        
        fig.update_layout(
            title="User Feedback Distribution",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed feedback breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👍 Positive Feedback")
            st.success(f"**{feedback_summary['liked_responses']}** responses received positive feedback")
            if feedback_summary["liked_responses"] > 0:
                positive_rate = (feedback_summary["liked_responses"] / feedback_summary["total_responses"]) * 100
                st.write(f"**{positive_rate:.1f}%** of all responses were liked")
        
        with col2:
            st.markdown("### 👎 Negative Feedback")
            st.error(f"**{feedback_summary['disliked_responses']}** responses received negative feedback")
            if feedback_summary["disliked_responses"] > 0:
                negative_rate = (feedback_summary["disliked_responses"] / feedback_summary["total_responses"]) * 100
                st.write(f"**{negative_rate:.1f}%** of all responses were disliked")
        
        # Recent feedback timeline
        if feedback_summary["detailed_feedback"]:
            st.markdown("### 🕒 Recent Feedback Timeline")
            
            recent_feedback = feedback_summary["trends"]["most_recent_feedback"]
            if recent_feedback:
                for i, item in enumerate(reversed(recent_feedback)):  # Show most recent first
                    feedback_icon = "👍" if item["feedback"] == "like" else "👎"
                    feedback_color = "green" if item["feedback"] == "like" else "red"
                    
                    with st.expander(f"{feedback_icon} Feedback #{len(recent_feedback)-i} - {item['feedback'].title()}", expanded=False):
                        st.write(f"**Question:** {item['question']}")
                        st.write(f"**Answer Preview:** {item['answer_preview']}")
                        st.write(f"**Processing Time:** {item['processing_time']:.2f}s")
                        if item.get('feedback_timestamp'):
                            st.write(f"**Feedback Given:** {item['feedback_timestamp']}")
            else:
                st.info("No recent feedback to display.")
        
        # Export feedback data
        st.markdown("### 📤 Export Feedback Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Feedback Summary"):
                feedback_json = json.dumps(feedback_summary, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON",
                    data=feedback_json,
                    file_name=f"feedback_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📋 View Raw Feedback Data"):
                with st.expander("Raw Feedback Data", expanded=True):
                    st.json(feedback_summary)
    
    else:
        st.info("🎯 **No feedback received yet!**")
        st.markdown("""
        **To get started with feedback analysis:**
        1. 💬 Ask questions in the Chat Assistant tab
        2. 👍👎 Use the feedback buttons on AI responses
        3. 📊 Return here to see your feedback analytics
        
        **Tips for better feedback:**
        - Give feedback on responses you find particularly helpful or unhelpful
        - Use feedback to help improve the AI's performance
        - Check back regularly to see trends in response quality
        """)

def display_sidebar(rag_agent):
    """Display the sidebar with system information"""
    st.sidebar.markdown("## 📊 System Status")
    
    if rag_agent:
        # Get system health
        health = rag_agent.get_system_health()
        overall_status = health.get("overall_status", "unknown")
        
        # System status with color coding
        if overall_status == "healthy":
            st.sidebar.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-online"></span>
                <strong>System Healthy</strong>
            </div>
            """, unsafe_allow_html=True)
        elif overall_status == "degraded":
            st.sidebar.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator" style="background-color: #ff9800;"></span>
                <strong>System Degraded</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-offline"></span>
                <strong>System Error</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Component health details
        components = health.get("components", {})
        
        st.sidebar.markdown("### 🔧 Component Health")
        
        # NVIDIA API status
        nvidia_status = components.get("nvidia_api", {})
        if nvidia_status.get("status") == "online":
            st.sidebar.success("🟢 NVIDIA API Connected")
            if "embedding_dimension" in nvidia_status:
                st.sidebar.info(
                    f"**Embedding**: nvidia/nv-embed-v1\n"
                    f"**Dimension**: {nvidia_status['embedding_dimension']}\n\n"
                    f"**LLM**: meta/llama-3.1-8b-instruct"
                )
        else:
            st.sidebar.error("🔴 NVIDIA API Offline")
            if "error" in nvidia_status:
                st.sidebar.error(f"Error: {nvidia_status['error'][:50]}...")
        
        # Model information
        # st.sidebar.markdown("### 🤖 AI Models")
        # st.sidebar.info("**Embedding**: nvidia/nv-embed-v1\n**LLM**: meta/llama-3.1-8b-instruct")
        
        # Vector database & Documents folder status
        vector_status = components.get("vector_database", {})
        docs_status = components.get("documents_folder", {})
        st.sidebar.markdown("### 📚 Knowledge Base Status")

        # Display overall status messages
        if vector_status.get("status") == "online":
            st.sidebar.success("🟢 FAISS Vector Database Online")
        else:
            st.sidebar.error("🔴 FAISS Vector Database Offline")

        if docs_status.get("status") == "accessible":
            st.sidebar.success("🟢 Documents Folder Accessible")
        else:
            st.sidebar.error("🔴 Documents Folder Issue")

        # Show metrics side by side
        kb_col1, kb_col2 = st.sidebar.columns(2)
        with kb_col1:
            st.metric("Documents", vector_status.get("document_count", 0))
        with kb_col2:
            st.metric("Files", docs_status.get("files_available", 0))
        
        # File watcher status
        watcher_status = components.get("file_watcher", {})
        if watcher_status.get("status") == "active":
            st.sidebar.success("🟢 File Watcher Active")
            st.sidebar.info("🔄 Auto-updates enabled")
        else:
            st.sidebar.warning("🟡 File Watcher Inactive")
        
        # System actions
        st.sidebar.markdown("### ⚙️ System Actions")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Optimize", help="Optimize knowledge base performance"):
                with st.spinner("Optimizing..."):
                    start_time = time.time()
                    if rag_agent.optimize_knowledge_base():
                        end_time = time.time()
                        stats = rag_agent.get_knowledge_base_stats()
                        
                        # Add system notification
                        add_system_notification("index_optimized", "", {
                            "documents": stats.get('document_count', 0),
                            "time_taken": end_time - start_time
                        })
                        
                        # Add timestamp message to chat
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        timestamp = datetime.now()
                        time_str = timestamp.strftime("%H:%M:%S")
                        chat_message = f"**⚡ Vector Store Optimization [{time_str}]**\n\n"
                        chat_message += f"**Status**: OPTIMIZATION COMPLETE\n"
                        chat_message += f"**Documents**: {stats.get('document_count', 0)}\n"
                        chat_message += f"**Time Taken**: {end_time - start_time:.1f} seconds\n"
                        chat_message += f"**Details**: Vector database has been optimized for better performance."
                        
                        st.session_state.messages.append({
                            "role": "system",
                            "content": chat_message,
                            "timestamp": timestamp,
                            "type": "vector_store_update",
                            "status_type": "optimization_complete"
                        })
                        
                        st.success("✅ Optimization completed!")
                    else:
                        add_system_notification("error", "Knowledge base optimization failed")
                        
                        # Add error message to chat
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        timestamp = datetime.now()
                        time_str = timestamp.strftime("%H:%M:%S")
                        chat_message = f"**❌ Vector Store Optimization Failed [{time_str}]**\n\n"
                        chat_message += f"**Status**: OPTIMIZATION FAILED\n"
                        chat_message += f"**Details**: Vector database optimization could not be completed."
                        
                        st.session_state.messages.append({
                            "role": "system",
                            "content": chat_message,
                            "timestamp": timestamp,
                            "type": "vector_store_update",
                            "status_type": "optimization_failed"
                        })
                        
                        st.error("❌ Optimization failed")
        
        with col2:
            if st.button("🔍 Health Check", help="Run comprehensive health check"):
                health = rag_agent.get_system_health()
                overall_status = health.get("overall_status", "unknown")
                
                # Add system notification
                add_system_notification("system", f"Health check completed - Status: {overall_status.title()}")
                st.json(health)
        
        # Conversational Memory Controls
        st.sidebar.markdown("### 🧠 Conversational Memory")
        if hasattr(rag_agent, 'is_conversational_mode_enabled') and rag_agent.is_conversational_mode_enabled():
            st.sidebar.success("🟢 Memory Active")
            
            # Show conversation history info
            if hasattr(rag_agent, 'get_conversation_history'):
                try:
                    conv_history = rag_agent.get_conversation_history()
                    if conv_history:
                        st.sidebar.info(f"💭 Remembering {len(conv_history)} conversation exchanges")
                    else:
                        st.sidebar.info("💭 No conversation history yet")
                except:
                    pass
            
            # Memory controls
            memory_col1, memory_col2 = st.sidebar.columns(2)
            with memory_col1:
                if st.button("🗑️ Clear Memory", help="Clear conversation memory"):
                    if hasattr(rag_agent, 'clear_conversation_memory'):
                        rag_agent.clear_conversation_memory()
                        st.success("✅ Memory cleared!")
                        
                        # Add notification to chat
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        timestamp = datetime.now()
                        time_str = timestamp.strftime("%H:%M:%S")
                        chat_message = f"**🧠 Conversation Memory Cleared [{time_str}]**\n\n"
                        chat_message += f"**Status**: MEMORY CLEARED\n"
                        chat_message += f"**Details**: All conversation history has been cleared. Future questions will be treated as new conversations."
                        
                        st.session_state.messages.append({
                            "role": "system",
                            "content": chat_message,
                            "timestamp": timestamp,
                            "type": "memory_update",
                            "status_type": "memory_cleared",
                            "details": {"status_type": "memory_cleared"}
                        })
                        st.rerun()
            
            with memory_col2:
                # Show memory window size
                if hasattr(rag_agent, 'memory_window_size'):
                    st.sidebar.info(f"📝 Window: {rag_agent.memory_window_size} turns")
        else:
            st.sidebar.warning("🟡 Memory Disabled")
            st.sidebar.info("💡 Basic mode - questions handled independently")
        
        # Document types supported
        st.sidebar.markdown("### 📖 Supported Documents")
        doc_types = [ "📄 PDF | 📄 Word Docs | 📽️ PowerPoint | 📝 Text Files" ]
            # "🌐 Web Pages", "📰 Articles", "📑 Reports
            # "📚 Research Papers", "🔧 Technical Docs", "⚖️ Legal Documents",
            # "🏢 Corporate Policies", "🎓 Training Materials", "📖 User Manuals",
            # "✅ Compliance Docs", "📊 Reports", "📝 Contracts"
        

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
    """Display the main chat interface with chronological order"""
    st.markdown("## 💬 Chat Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial system message
        st.session_state.messages.append({
            "role": "system",
            "content": "🤖 RAG Assistant initialized successfully!",
            "timestamp": datetime.now(),
            "type": "system_init"
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your RAG Assistant powered by NVIDIA NemoRetriever. I can help you find information from your document collection. What would you like to know?",
            "sources": [],
            "processing_time": 0,
            "timestamp": datetime.now(),
            "type": "chat"
        })

    # Create a container for the chat that can be updated
    chat_container = st.container()
    
    with chat_container:
        # Display messages in chronological order (oldest first)
        for message_idx, message in enumerate(st.session_state.messages):
            timestamp = message.get("timestamp", datetime.now())
            time_str = timestamp.strftime("%H:%M:%S")
            
            if message["role"] == "system":
                # System notifications including vector DB status updates
                message_type = message.get("type", "system")
                status_type = message.get("status_type", "")
                
                # Different styling for vector DB status messages
                if message_type == "vector_db_status":
                    # Vector database status messages with enhanced styling
                    if "error" in str(status_type).lower():
                        status_class = "status-error"
                    elif any(word in str(status_type).lower() for word in ["updating", "adding", "removing", "rebuilding"]):
                        status_class = "status-updating"
                    else:
                        status_class = "status-online"
                    
                    st.markdown(f"""
                    <div class="chat-message {status_class}" style="border-left: 4px solid #6c757d; background-color: #f8f9fa; margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong>🔧 Vector Database</strong>
                            <small style="color: #666;">{time_str}</small>
                        </div>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                elif message_type == "vector_store_update":
                    # Vector store update messages from file watcher
                    if "error" in str(status_type).lower():
                        status_class = "status-error"
                    elif any(word in str(status_type).lower() for word in ["document_added", "document_updated", "vector_db_updated"]):
                        status_class = "status-updating"
                    elif "document_removed" in str(status_type).lower():
                        status_class = "status-error"
                    else:
                        status_class = "status-online"
                    
                    st.markdown(f"""
                    <div class="chat-message {status_class}" style="border-left: 4px solid #9c27b0; background-color: #f3e5f5; margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong>💾 Vector Store</strong>
                            <small style="color: #666;">{time_str}</small>
                        </div>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Regular system notifications
                    display_system_message(
                        message.get("type", "system"),
                        message.get("filename", ""),
                        message.get("details", {})
                    )
            elif message["role"] == "user":
                # User messages
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>👤 You</strong>
                        <small style="color: #666;">{time_str}</small>
                    </div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Assistant messages
                conversational_indicator = ""
                if message.get("conversational"):
                    history_count = message.get("chat_history_used", 0)
                    conversational_indicator = f" 🧠 (Using {history_count} previous exchanges)"
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>🤖 AI Assistant{conversational_indicator}</strong>
                        <small style="color: #666;">{time_str}</small>
                    </div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Add feedback buttons for assistant messages
                if message.get("type") == "chat":  # Only for actual chat responses, not system messages
                    message_id = message.get("message_id", f"msg_{message_idx}")
                    current_feedback = message.get("feedback")
                    render_feedback_buttons(message_id, current_feedback)
                
                # Display sources if available
                if message.get("sources"):
                    display_sources(message["sources"], message.get("processing_time", 0), message_idx=message_idx)

    # Chat input at the bottom
    st.markdown("---")
    
    # Add conversational memory status indicator if available
    if rag_agent and hasattr(rag_agent, 'is_conversational_mode_enabled'):
        if rag_agent.is_conversational_mode_enabled():
            st.info("🧠 **Conversational Memory Active** - I can remember our previous conversation and answer follow-up questions!")
        else:
            st.warning("📝 **Basic Mode** - Each question is handled independently")
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message with timestamp
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now(),
            "type": "chat"
        }
        st.session_state.messages.append(user_message)
        
        if rag_agent:
            try:
                # Extract chat history for conversational context
                chat_history = []
                if hasattr(rag_agent, 'is_conversational_mode_enabled') and rag_agent.is_conversational_mode_enabled():
                    # Find the last memory clear message to determine valid conversation history
                    last_memory_clear_index = -1
                    for i, msg in enumerate(st.session_state.messages):
                        if (msg["role"] == "system" and 
                            msg.get("type") == "memory_update" and 
                            msg.get("status_type") == "memory_cleared"):
                            last_memory_clear_index = i
                    
                    # Extract only user-assistant conversation pairs for context
                    # Only consider messages after the last memory clear
                    user_messages = []
                    assistant_messages = []
                    
                    for i, msg in enumerate(st.session_state.messages):
                        # Only include messages after the last memory clear
                        if i > last_memory_clear_index:
                            if msg["role"] == "user" and msg.get("type") == "chat":
                                user_messages.append(msg["content"])
                            elif msg["role"] == "assistant" and msg.get("type") == "chat" and msg.get("content"):
                                assistant_messages.append(msg["content"])
                    
                    # Pair up the messages (excluding the current question)
                    min_len = min(len(user_messages) - 1, len(assistant_messages))  # -1 to exclude current question
                    chat_history = [(user_messages[i], assistant_messages[i]) for i in range(min_len)]
                
                # Get response from RAG agent with chat history
                with st.spinner("🔍 Searching documents..."):
                    if chat_history and hasattr(rag_agent, 'ask_question'):
                        # Try to call the enhanced ask_question method with chat history
                        try:
                            response = rag_agent.ask_question(prompt, chat_history=chat_history)
                        except TypeError:
                            # Fallback to basic method if chat_history parameter not supported
                            response = rag_agent.ask_question(prompt)
                    else:
                        response = rag_agent.ask_question(prompt)

                # Validate response
                if not response or not response.answer:
                    error_message = {
                        "role": "assistant",
                        "content": "❌ I apologize, but I couldn't generate a response. Please try rephrasing your question.",
                        "sources": [],
                        "processing_time": 0,
                        "timestamp": datetime.now(),
                        "type": "error"
                    }
                    st.session_state.messages.append(error_message)
                else:
                    # Add assistant response with timestamp and conversational context
                    assistant_message = {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.source_documents,
                        "processing_time": response.processing_time,
                        "timestamp": datetime.now(),
                        "type": "chat",
                        "conversational": bool(chat_history),  # Mark if this was a conversational response
                        "chat_history_used": len(chat_history) if chat_history else 0,
                        "message_id": response.message_id,  # Add unique message ID for feedback
                        "question": prompt,  # Store the original question for feedback analysis
                        "feedback": None,  # Initialize feedback as None
                        "feedback_timestamp": None  # Initialize feedback timestamp
                    }
                    st.session_state.messages.append(assistant_message)

                # Force refresh to show new messages
                st.rerun()

            except Exception as e:
                error_message = {
                    "role": "assistant",
                    "content": f"❌ I encountered an error while processing your question: {str(e)}",
                    "sources": [],
                    "processing_time": 0,
                    "timestamp": datetime.now(),
                    "type": "error"
                }
                st.session_state.messages.append(error_message)
                st.rerun()

        else:
            # System unavailable message
            system_message = {
                "role": "system",
                "content": "❌ RAG system not available. Please check the system status.",
                "timestamp": datetime.now(),
                "type": "error"
            }
            st.session_state.messages.append(system_message)
            st.rerun()

def display_sources(source_documents, processing_time, confidence_scores=None, message_idx=0):
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
                        # if st.button(f"📋 Copy Text {i}", key=f"copy_{i}"):
                        if st.button(f"📋 Copy Text {i}", key=f"copy_{message_idx}_{i}"):
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
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{message_idx}")

                # Bar chart for page distribution
                pages = [doc.metadata.get("page", 0) for doc in source_documents]
                if pages:
                    fig_bar = px.histogram(
                        x=pages,
                        title="📄 Page Distribution of Sources",
                        labels={'x': 'Page Number', 'y': 'Number of Sources'},
                        color_discrete_sequence=['#2a5298']
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_{message_idx}")

        with tab3:
            st.markdown("#### 🔍 Search Within Sources")
            search_term = st.text_input("Search for specific terms in the source documents:", key=f"search_{message_idx}")

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

    st.markdown("## 📊 Document Statistics & System Health")

    # Get comprehensive system health
    health = rag_agent.get_system_health()
    stats = rag_agent.get_knowledge_base_stats()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Documents", stats.get('document_count', 0))
    with col2:
        st.metric("📁 Supported Files", stats.get('files_available', 0))
    with col3:
        st.metric("🔍 Searchable Chunks", stats.get('document_count', 0))
    with col4:
        overall_status = health.get('overall_status', 'unknown')
        status_emoji = {"healthy": "🟢", "degraded": "🟡", "error": "🔴"}.get(overall_status, "❓")
        st.metric("🏥 System Health", f"{status_emoji} {overall_status.title()}")

    # System health details
    st.markdown("### � System Health Dashboard")
    
    health_col1, health_col2 = st.columns(2)
    
    with health_col1:
        st.markdown("**Component Status**")
        components = health.get("components", {})
        
        for component_name, component_data in components.items():
            status = component_data.get("status", "unknown")
            display_name = component_name.replace("_", " ").title()
            
            if status in ["online", "active", "accessible", "healthy"]:
                st.success(f"� {display_name}: {status.title()}")
            elif status in ["degraded", "inactive"]:
                st.warning(f"🟡 {display_name}: {status.title()}")
            else:
                st.error(f"🔴 {display_name}: {status.title()}")
                if "error" in component_data:
                    st.error(f"   Error: {component_data['error'][:100]}...")

    with health_col2:
        st.markdown("**Performance Metrics**")
        
        # NVIDIA API metrics
        nvidia_component = components.get("nvidia_api", {})
        if "embedding_dimension" in nvidia_component:
            st.info(f"📊 Embedding Dimension: {nvidia_component['embedding_dimension']}")
        
        # Vector DB metrics
        vector_component = components.get("vector_database", {})
        if "document_count" in vector_component:
            st.info(f"💾 Indexed Documents: {vector_component['document_count']}")
        
        # File system metrics
        docs_component = components.get("documents_folder", {})
        if "files_available" in docs_component:
            st.info(f"📁 Available Files: {docs_component['files_available']}")
        
        st.info("🚀 Average Query Time: 2-8 seconds")

    # Enhanced system actions
    st.markdown("### ⚙️ System Management")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Optimize Knowledge Base", help="Optimize vector database for better performance"):
            with st.spinner("🔧 Optimizing knowledge base..."):
                try:
                    start_time = time.time()
                    if rag_agent.optimize_knowledge_base():
                        end_time = time.time()
                        stats = rag_agent.get_knowledge_base_stats()
                        
                        # Add timestamp message to chat
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        timestamp = datetime.now()
                        time_str = timestamp.strftime("%H:%M:%S")
                        chat_message = f"**⚡ Vector Store Optimization [{time_str}]**\n\n"
                        chat_message += f"**Status**: OPTIMIZATION COMPLETE\n"
                        chat_message += f"**Documents**: {stats.get('document_count', 0)}\n"
                        chat_message += f"**Time Taken**: {end_time - start_time:.1f} seconds\n"
                        chat_message += f"**Details**: Vector database optimized from Document Statistics page."
                        
                        st.session_state.messages.append({
                            "role": "system",
                            "content": chat_message,
                            "timestamp": timestamp,
                            "type": "vector_store_update",
                            "status_type": "optimization_complete"
                        })
                        
                        st.success("✅ Knowledge base optimized successfully!")
                        st.balloons()
                    else:
                        # Add error message to chat
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        timestamp = datetime.now()
                        time_str = timestamp.strftime("%H:%M:%S")
                        chat_message = f"**❌ Vector Store Optimization Failed [{time_str}]**\n\n"
                        chat_message += f"**Status**: OPTIMIZATION FAILED\n"
                        chat_message += f"**Details**: Vector database optimization could not be completed."
                        
                        st.session_state.messages.append({
                            "role": "system",
                            "content": chat_message,
                            "timestamp": timestamp,
                            "type": "vector_store_update",
                            "status_type": "optimization_failed"
                        })
                        
                        st.error("❌ Optimization failed. Check system logs.")
                except Exception as e:
                    # Add error message to chat
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    
                    timestamp = datetime.now()
                    time_str = timestamp.strftime("%H:%M:%S")
                    chat_message = f"**❌ Vector Store Optimization Error [{time_str}]**\n\n"
                    chat_message += f"**Status**: OPTIMIZATION ERROR\n"
                    chat_message += f"**Error**: {str(e)}\n"
                    chat_message += f"**Details**: An error occurred during vector database optimization."
                    
                    st.session_state.messages.append({
                        "role": "system",
                        "content": chat_message,
                        "timestamp": timestamp,
                        "type": "vector_store_update",
                        "status_type": "optimization_error"
                    })
                    
                    st.error(f"❌ Optimization error: {str(e)}")
    
    with action_col2:
        if st.button("🔍 Run Health Check", help="Perform comprehensive system diagnostics"):
            with st.spinner("🩺 Running health diagnostics..."):
                fresh_health = rag_agent.get_system_health()
                overall = fresh_health.get("overall_status", "unknown")
                
                if overall == "healthy":
                    st.success("✅ All systems healthy!")
                elif overall == "degraded":
                    st.warning("⚠️ Some components need attention")
                else:
                    st.error("❌ System issues detected")
                
                with st.expander("📋 Detailed Health Report"):
                    st.json(fresh_health)
    
    with action_col3:
        if st.button("🔧 Rebuild Index", help="Force rebuild of the entire vector index"):
            if st.checkbox("⚠️ Confirm rebuild (this may take time)", key="confirm_rebuild"):
                with st.spinner("🏗️ Rebuilding knowledge base..."):
                    try:
                        start_time = time.time()
                        if rag_agent.setup_knowledge_base(force_rebuild=True):
                            end_time = time.time()
                            stats = rag_agent.get_knowledge_base_stats()
                            
                            # Add system notification
                            add_system_notification("vector_db_rebuilt", "", {
                                "documents": stats.get('document_count', 0),
                                "time_taken": end_time - start_time
                            })
                            
                            # Add timestamp message to chat
                            if "messages" not in st.session_state:
                                st.session_state.messages = []
                            
                            timestamp = datetime.now()
                            time_str = timestamp.strftime("%H:%M:%S")
                            chat_message = f"**🔨 Vector Store Rebuild [{time_str}]**\n\n"
                            chat_message += f"**Status**: REBUILD COMPLETE\n"
                            chat_message += f"**Documents**: {stats.get('document_count', 0)}\n"
                            chat_message += f"**Time Taken**: {end_time - start_time:.1f} seconds\n"
                            chat_message += f"**Details**: Vector database has been completely rebuilt from scratch."
                            
                            st.session_state.messages.append({
                                "role": "system",
                                "content": chat_message,
                                "timestamp": timestamp,
                                "type": "vector_store_update",
                                "status_type": "rebuild_complete"
                            })
                            
                            st.success("✅ Knowledge base rebuilt successfully!")
                            st.balloons()
                        else:
                            add_system_notification("error", "Knowledge base rebuild failed")
                            
                            # Add error message to chat
                            if "messages" not in st.session_state:
                                st.session_state.messages = []
                            
                            timestamp = datetime.now()
                            time_str = timestamp.strftime("%H:%M:%S")
                            chat_message = f"**❌ Vector Store Rebuild Failed [{time_str}]**\n\n"
                            chat_message += f"**Status**: REBUILD FAILED\n"
                            chat_message += f"**Details**: Vector database rebuild could not be completed."
                            
                            st.session_state.messages.append({
                                "role": "system",
                                "content": chat_message,
                                "timestamp": timestamp,
                                "type": "vector_store_update",
                                "status_type": "rebuild_failed"
                            })
                            
                            st.error("❌ Rebuild failed. Check system logs.")
                    except Exception as e:
                        add_system_notification("error", f"Rebuild error: {str(e)}")
                        
                        # Add error message to chat
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        timestamp = datetime.now()
                        time_str = timestamp.strftime("%H:%M:%S")
                        chat_message = f"**❌ Vector Store Rebuild Error [{time_str}]**\n\n"
                        chat_message += f"**Status**: REBUILD ERROR\n"
                        chat_message += f"**Error**: {str(e)}\n"
                        chat_message += f"**Details**: An error occurred during vector database rebuild."
                        
                        st.session_state.messages.append({
                            "role": "system",
                            "content": chat_message,
                            "timestamp": timestamp,
                            "type": "vector_store_update",
                            "status_type": "rebuild_error"
                        })
                        
                        st.error(f"❌ Rebuild error: {str(e)}")
    
    # Real-time file monitoring status
    if health.get("components", {}).get("file_watcher", {}).get("status") == "active":
        st.markdown("### 👀 Real-time File Monitoring")
        st.success("🔄 **Auto-updates enabled** - The system will automatically detect and process new PDF files added to your documents folder")
        st.info("� **Monitored folder**: " + str(docs_component.get("path", "Unknown")))
        st.info("⚡ **Processing mode**: Batch processing with 5-second delay for optimal performance")
    else:
        st.markdown("### 👀 File Monitoring")
        st.warning("🔄 Auto-updates currently disabled")
        st.info("� Restart the system to enable automatic file monitoring")
        
        
    # Document types breakdown with enhanced visualization
    st.markdown("### 📖 Supported Document Types & Use Cases")
    
    doc_categories = {
        "Research & Academic": ["📚 Research Papers", "🎓 Academic Papers", "📊 Reports & Analysis"],
        "Technical & Engineering": ["🔧 Technical Documentation", "📖 User Manuals", "🔍 Specifications"],
        "Legal & Compliance": ["⚖️ Legal Documents", "📝 Contracts & Agreements", "✅ Compliance Documents"],
        "Corporate & Business": ["🏢 Corporate Policies", "🎓 Training Materials", "� Business Reports"]
    }

    for category, doc_types in doc_categories.items():
        with st.expander(f"📂 {category}"):
            for doc_type in doc_types:
                st.markdown(f"• {doc_type}")

def main():
    """Main application function"""
    # Process any pending file watcher notifications
    process_file_watcher_notifications()
    
    # Auto-refresh every 5 seconds to show new vector DB updates
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
    
    if st.session_state.auto_refresh:
        time.sleep(0.1)  # Small delay for UI responsiveness
    
    # Display header
    display_header()

    # Initialize RAG agent
    rag_agent = initialize_rag_agent()

    # Display sidebar
    display_sidebar(rag_agent)

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Assistant", "📊 Document Statistics", "📝 Feedback Analysis", "🌐 Web Import"])

    with tab1:
        # Main content area
        col1, col2 = st.columns([3, 1])

        with col1:
            # Chat interface
            display_chat_interface(rag_agent)
    
    with col2:
        # Quick actions and tips with conversational examples
        st.markdown("### 💡 Quick Tips")
        
        # Check if conversational mode is available
        if rag_agent and hasattr(rag_agent, 'is_conversational_mode_enabled') and rag_agent.is_conversational_mode_enabled():
            st.success("🧠 **Conversational Mode Active!**")
            st.info("""
            **Try these conversational sequences:**
            
            **First ask:** "What is the main topic of the documents?"
            **Then follow up:** "Can you explain that in more detail?"
            **Or ask:** "What are the key benefits of this?"
            
            **Other follow-up examples:**
            • "What else should I know about this?"
            • "Are there any requirements mentioned?"
            • "How does this compare to [other topic]?"
            • "Can you give me more examples?"
            • "What are the implications of this?"
            """)
        else:
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
                feedback_summary = FeedbackAnalytics.get_feedback_summary(st.session_state.messages)
                
                for msg in st.session_state.messages:
                    export_item = {
                        "timestamp": msg.get("timestamp", datetime.now()).isoformat() if hasattr(msg.get("timestamp", datetime.now()), 'isoformat') else str(msg.get("timestamp", datetime.now())),
                        "role": msg["role"],
                        "content": msg["content"],
                        "sources_count": len(msg.get("sources", [])),
                        "processing_time": msg.get("processing_time", 0)
                    }
                    
                    # Add feedback-related fields for assistant messages
                    if msg["role"] == "assistant" and msg.get("type") == "chat":
                        export_item.update({
                            "message_id": msg.get("message_id", ""),
                            "question": msg.get("question", ""),
                            "feedback": msg.get("feedback"),
                            "feedback_timestamp": msg.get("feedback_timestamp"),
                            "conversational": msg.get("conversational", False),
                            "chat_history_used": msg.get("chat_history_used", 0)
                        })
                    
                    chat_export.append(export_item)

                # Create comprehensive export with session summary
                comprehensive_export = {
                    "export_metadata": {
                        "export_timestamp": datetime.now().isoformat(),
                        "total_messages": len(st.session_state.messages),
                        "session_start": st.session_state.messages[0].get("timestamp", datetime.now()).isoformat() if st.session_state.messages else None
                    },
                    "feedback_analytics": feedback_summary,
                    "messages": chat_export
                }

                export_data = json.dumps(comprehensive_export, indent=2, default=str)
                st.download_button(
                    label="💾 Download Chat History (with Feedback)",
                    data=export_data,
                    file_name=f"RAG_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
            add_system_notification("system", "Clearing cache and rebuilding knowledge base")
            
            # Add timestamp message to chat
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            timestamp = datetime.now()
            time_str = timestamp.strftime("%H:%M:%S")
            chat_message = f"**🔄 Vector Store Cache Clear [{time_str}]**\n\n"
            chat_message += f"**Status**: CACHE CLEARED\n"
            chat_message += f"**Details**: System cache cleared and knowledge base will be reloaded."
            
            st.session_state.messages.append({
                "role": "system",
                "content": chat_message,
                "timestamp": timestamp,
                "type": "vector_store_update",
                "status_type": "cache_cleared"
            })
            
            st.cache_resource.clear()
            st.rerun()

        # # System information
        # st.markdown("### ℹ️ About")
        # st.markdown("""
        # This AI assistant is powered by:
        # - **NVIDIA** embedding models
        # - **Meta LLaMA** language model
        # - **FAISS** vector database
        # - **1,869** legal document chunks
        # """)

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
    
    with tab3:
        # Feedback analysis page
        display_feedback_analysis()
    
    with tab4:
        # Web Import page
        display_web_import()

def display_web_import():
    """Display the Web Import interface for downloading files from URLs"""
    st.header("🌐 Web Import")
    st.markdown("Import documents directly from web URLs into your knowledge base")
    
    # Initialize web importer - use same folder as RAG system
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    data_folder = Path(docs_folder)
    web_importer = WebImporter(data_folder)
    
    # Show download location info
    st.info(f"📁 **Download Location:** Files will be saved to `{docs_folder}/` and automatically processed by the RAG system")
    
    # Single URL import section
    st.subheader("📄 Single URL Import")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "Enter URL:", 
            placeholder="https://example.com/document.pdf",
            help="Supported formats: PDF, DOCX, PPTX, TXT, images (JPG, PNG, etc.)"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        if st.button("📥 Import File", type="primary", disabled=not url_input.strip()):
            if url_input.strip():
                # Progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Import the file
                    with st.spinner("Downloading and importing file..."):
                        success, message, import_record = web_importer.download_from_url(url_input.strip())
                    
                    if success:
                        progress_bar.progress(1.0)
                        status_text.success(f"✅ Successfully imported: {import_record.filename}")
                        st.balloons()
                        
                        # Show file details
                        size_mb = import_record.file_size / (1024 * 1024) if import_record.file_size else 0
                        st.info(f"**Saved to:** {import_record.local_path}\n**File size:** {size_mb:.2f} MB")
                        
                        # Clear input after successful import
                        time.sleep(2)
                        st.rerun()
                    else:
                        progress_bar.progress(0)
                        status_text.error(f"❌ Import failed: {message}")
                        
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.error(f"❌ Unexpected error: {str(e)}")
    
    st.markdown("---")
    
    # Batch import section
    st.subheader("📚 Batch URL Import")
    st.markdown("Import multiple files at once by providing URLs (one per line)")
    
    # Text area for multiple URLs
    urls_text = st.text_area(
        "Enter URLs (one per line):", 
        height=150,
        placeholder="https://example.com/doc1.pdf\nhttps://example.com/doc2.docx\nhttps://example.com/doc3.txt"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📥 Import All", type="primary", disabled=not urls_text.strip()):
            urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
            
            if urls:
                st.markdown("### Import Progress")
                
                # Create progress tracking
                overall_progress = st.progress(0)
                results = []
                
                for i, url in enumerate(urls):
                    with st.expander(f"Importing {i+1}/{len(urls)}: {url}", expanded=True):
                        file_progress = st.progress(0)
                        file_status = st.empty()
                        
                        try:
                            success, message, import_record = web_importer.download_from_url(url)
                            
                            if success:
                                file_progress.progress(1.0)
                                file_status.success(f"✅ {import_record.filename}")
                            else:
                                file_progress.progress(0)
                                file_status.error(f"❌ {message}")
                                
                            results.append(import_record)
                            
                        except Exception as e:
                            file_progress.progress(0)
                            file_status.error(f"❌ Error: {str(e)}")
                            # Create a failed import record
                            failed_record = type('ImportRecord', (), {
                                'status': 'failed', 
                                'url': url, 
                                'error_message': str(e),
                                'filename': 'failed'
                            })()
                            results.append(failed_record)
                    
                    # Update overall progress
                    overall_progress.progress((i + 1) / len(urls))
                
                # Show summary
                successful = sum(1 for r in results if r.status == "success")
                st.markdown("### Import Summary")
                
                if successful > 0:
                    st.success(f"✅ Successfully imported {successful}/{len(urls)} files")
                    if successful < len(urls):
                        st.warning(f"⚠️ {len(urls) - successful} files failed to import")
                else:
                    st.error("❌ No files were successfully imported")
    
    with col2:
        if st.button("🔄 Validate URLs"):
            urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
            
            if urls:
                st.markdown("### URL Validation")
                
                for url in urls:
                    is_valid, message, detected_ext = web_importer.is_valid_url(url)
                    if is_valid:
                        ext_info = f" (detected: {detected_ext})" if detected_ext else ""
                        st.success(f"✅ {url}{ext_info}")
                    else:
                        st.error(f"❌ {url}: {message}")
    
    with col3:
        st.markdown("**Supported Formats:**")
        st.markdown("""
        - 📄 PDF documents
        - 📝 Word documents (DOCX)
        - 📊 PowerPoint presentations (PPTX)
        - 📃 Text files (TXT)
        - 🖼️ Images (JPG, PNG, GIF, etc.)
        """)
    
    st.markdown("---")
    
    # Recent imports section
    st.subheader("📋 Recent Imports")
    
    # Check for recent files in Data folder
    recent_files = []
    if data_folder.exists():
        for file_path in data_folder.glob("**/*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                recent_files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime)
                })
    
    # Sort by modification time (newest first)
    recent_files.sort(key=lambda x: x["modified"], reverse=True)
    
    if recent_files:
        # Show only the 10 most recent files
        for file_info in recent_files[:10]:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.text(file_info["name"])
            
            with col2:
                size_mb = file_info["size"] / (1024 * 1024)
                st.text(f"{size_mb:.2f} MB")
            
            with col3:
                st.text(file_info["modified"].strftime("%m/%d %H:%M"))
        
        st.caption(f"Showing {min(10, len(recent_files))} of {len(recent_files)} files in knowledge base")
    else:
        st.info("No files found in the knowledge base. Import some files to get started!")

if __name__ == "__main__":
    main()
