"""
File Watcher for RAG System
Monitors the document directory for changes and updates the knowledge base.
"""
import time
import logging
from pathlib import Path
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# Forward reference to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag_agent import RAGAgent

logger = logging.getLogger(__name__)

class DocumentChangeHandler(FileSystemEventHandler):
    """Handles file system events for the document folder."""

    def __init__(self, rag_agent: "RAGAgent"):
        """
        Initializes the handler with a RAGAgent instance.
        
        Args:
            rag_agent: The RAG agent that will process document changes.
        """
        self.rag_agent = rag_agent
        # A set to track recently handled files to avoid duplicate processing
        self.debounce_set = set()

    def _is_valid_file(self, event: FileSystemEvent) -> bool:
        """Checks if the event corresponds to a PDF file we should process."""
        if event.is_directory:
            return False
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() != ".pdf":
            return False
            
        # Debounce check
        if file_path in self.debounce_set:
            return False
            
        return True

    def _add_to_debounce(self, file_path: Path):
        """Adds a file to the debounce set and removes it after a short period."""
        self.debounce_set.add(file_path)
        
        def remove_from_set():
            time.sleep(2) # Debounce for 2 seconds
            self.debounce_set.discard(file_path)
            
        Thread(target=remove_from_set).start()

    def on_created(self, event: FileSystemEvent):
        """Called when a file is created."""
        if not self._is_valid_file(event):
            return
        
        file_path = Path(event.src_path)
        logger.info(f"📄 New document detected: {file_path.name}. Adding to knowledge base.")
        self._add_to_debounce(file_path)
        self.rag_agent.add_document(str(file_path))

    def on_modified(self, event: FileSystemEvent):
        """Called when a file is modified."""
        if not self._is_valid_file(event):
            return
            
        file_path = Path(event.src_path)
        logger.info(f"🔄 Document modified: {file_path.name}. Updating knowledge base.")
        self._add_to_debounce(file_path)
        self.rag_agent.update_document(str(file_path))

    def on_deleted(self, event: FileSystemEvent):
        """Called when a file is deleted."""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # For deleted files, check extension from the path (file doesn't exist anymore)
        if file_path.suffix.lower() != ".pdf":
            return
            
        # Skip debounce check for deletions since file is already gone
        logger.info(f"🗑️ Document deleted: {file_path.name}. Removing from knowledge base.")
        self.rag_agent.remove_document(str(file_path))


def start_file_watcher(rag_agent: "RAGAgent", watch_path: str):
    """
    Starts the file watcher in a background thread.

    Args:
        rag_agent: The RAGAgent instance to use for processing.
        watch_path: The directory path to monitor.
    """
    path = Path(watch_path)
    if not path.exists() or not path.is_dir():
        logger.error(f"Watch path '{watch_path}' is not a valid directory. File watcher not started.")
        return

    event_handler = DocumentChangeHandler(rag_agent)
    observer = Observer()
    observer.schedule(event_handler, str(path), recursive=False)
    
    # Start the observer
    observer.start()
    
    logger.info(f"👀 File watcher started on directory: '{watch_path}'")
    return observer
