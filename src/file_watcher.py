"""
File Watcher for RAG System
Monitors the document directory for changes and updates the knowledge base.
"""
import time
import logging
from pathlib import Path
from threading import Thread, Timer
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import queue
import threading

# Forward reference to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag_agent import RAGAgent

logger = logging.getLogger(__name__)

# Global notification queue for cross-thread communication
notification_queue = queue.Queue()

def get_pending_notifications():
    """Get all pending notifications from the queue"""
    notifications = []
    try:
        while True:
            notification = notification_queue.get_nowait()
            notifications.append(notification)
    except queue.Empty:
        pass
    return notifications

class BatchDocumentChangeHandler(FileSystemEventHandler):
    """
    Enhanced file handler that batches operations to avoid excessive processing.
    """

    def __init__(self, rag_agent: "RAGAgent", batch_delay: float = 5.0):
        """
        Initializes the handler with a RAGAgent instance.
        
        Args:
            rag_agent: The RAG agent that will process document changes.
            batch_delay: Time to wait before processing batched operations (seconds).
        """
        self.rag_agent = rag_agent
        self.batch_delay = batch_delay
        self.pending_operations = defaultdict(set)  # Use set to avoid duplicates
        self.timer = None
        
        # Track recently processed files to avoid duplicate processing
        self.debounce_set = set()

    def _is_valid_file(self, event: FileSystemEvent) -> bool:
        """Checks if the event corresponds to a supported file we should process."""
        if event.is_directory:
            return False
        
        file_path = Path(event.src_path)
        supported_extensions = self.rag_agent.document_loader.SUPPORTED_EXTENSIONS.keys()
        if file_path.suffix.lower() not in supported_extensions:
            return False
            
        # Debounce check - avoid processing the same file too quickly
        if file_path in self.debounce_set:
            return False
            
        return True

    def _add_to_debounce(self, file_path: Path):
        """Adds a file to the debounce set and removes it after a short period."""
        self.debounce_set.add(file_path)
        
        def remove_from_set():
            time.sleep(2)  # Debounce for 2 seconds
            self.debounce_set.discard(file_path)
            
        Thread(target=remove_from_set, daemon=True).start()

    def _schedule_operation(self, operation: str, file_path: str):
        """Schedule operation with batching"""
        path_obj = Path(file_path)
        
        # Add to pending operations
        self.pending_operations[operation].add(file_path)
        self._add_to_debounce(path_obj)
        
        # Reset timer
        if self.timer:
            self.timer.cancel()
        
        self.timer = Timer(self.batch_delay, self._process_batch)
        self.timer.start()
        
        logger.debug(f"Scheduled {operation} operation for {path_obj.name}")

    def _add_notification(self, notification_type: str, file_name: str, details: dict = None):
        """Add a notification to the queue for the main app to display"""
        notification = {
            "type": notification_type,
            "filename": file_name,
            "details": details or {},
            "timestamp": time.time()
        }
        notification_queue.put(notification)

    def _process_batch(self):
        """Process all pending operations in batch"""
        if not any(self.pending_operations.values()):
            return
        
        logger.info("🔄 Processing batch of file operations...")
        
        try:
            operations_count = sum(len(ops) for ops in self.pending_operations.values())
            logger.info(f"Processing {operations_count} file operations")
            
            # Process deletions first (order matters)
            deleted_files = self.pending_operations.get('deleted', set())
            for file_path in deleted_files:
                try:
                    file_name = Path(file_path).name
                    logger.info(f"🗑️ Removing document: {file_name}")
                    self.rag_agent.remove_document(file_path)
                    self._add_notification("document_removed", file_name)
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {str(e)}")
                    self._add_notification("error", Path(file_path).name, {"error": str(e)})
            
            # Then process additions and modifications
            modified_files = self.pending_operations.get('modified', set())
            created_files = self.pending_operations.get('created', set())
            all_changes = modified_files.union(created_files)
            
            for file_path in all_changes:
                try:
                    path_obj = Path(file_path)
                    if path_obj.exists():  # File might have been deleted after scheduling
                        if file_path in created_files:
                            logger.info(f"📄 Adding new document: {path_obj.name}")
                            self.rag_agent.add_document(file_path)
                            self._add_notification("document_added", path_obj.name)
                        else:
                            logger.info(f"🔄 Updating document: {path_obj.name}")
                            self.rag_agent.update_document(file_path)
                            self._add_notification("document_updated", path_obj.name)
                    else:
                        logger.warning(f"File no longer exists: {path_obj.name}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    self._add_notification("error", Path(file_path).name, {"error": str(e)})
            
            # Save the updated index once after all operations
            if operations_count > 0:
                try:
                    self.rag_agent.vector_db.save_index()
                    logger.info(f"✅ Batch processing completed successfully ({operations_count} operations)")
                    
                    # Add batch completion notification
                    self._add_notification("vector_db_updated", "", {
                        "operations": operations_count,
                        "deleted": len(deleted_files),
                        "added": len(created_files),
                        "updated": len(modified_files)
                    })
                except Exception as e:
                    logger.error(f"Failed to save index after batch processing: {str(e)}")
                    self._add_notification("error", "Index save failed", {"error": str(e)})
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            self._add_notification("error", "Batch processing failed", {"error": str(e)})
        finally:
            # Clear pending operations
            self.pending_operations.clear()

    def on_created(self, event: FileSystemEvent):
        """Called when a file is created."""
        if not self._is_valid_file(event):
            return
        
        self._schedule_operation('created', event.src_path)

    def on_modified(self, event: FileSystemEvent):
        """Called when a file is modified."""
        if not self._is_valid_file(event):
            return
            
        self._schedule_operation('modified', event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        """Called when a file is deleted."""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # For deleted files, check extension from the path (file doesn't exist anymore)
        supported_extensions = self.rag_agent.document_loader.SUPPORTED_EXTENSIONS.keys()
        if file_path.suffix.lower() not in supported_extensions:
            return
            
        self._schedule_operation('deleted', event.src_path)


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

    event_handler = BatchDocumentChangeHandler(rag_agent)
    observer = Observer()
    observer.schedule(event_handler, str(path), recursive=False)
    
    # Start the observer
    observer.start()
    
    logger.info(f"👀 Enhanced file watcher started on directory: '{watch_path}'")
    logger.info(f"🔄 Batch processing enabled with 5-second delay for optimal performance")
    return observer


# Legacy handler for backward compatibility
class DocumentChangeHandler(BatchDocumentChangeHandler):
    """Legacy handler - now just an alias for BatchDocumentChangeHandler"""
    
    def __init__(self, rag_agent: "RAGAgent"):
        super().__init__(rag_agent, batch_delay=5.0)
