"""
Web Importer for RAG System
Handles downloading files from web URLs to local document folder
"""

import os
import requests
import mimetypes
import tempfile
import shutil
import logging
from urllib.parse import urlparse, unquote
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImportRecord:
    """Record of a web import attempt"""
    timestamp: datetime
    url: str
    filename: str
    status: str  # "success", "failed", "processing"
    local_path: Optional[str] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    import_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class WebImporter:
    """Handle downloading files from web URLs to local document folder"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.pptx', '.txt', '.doc', '.rtf',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'
    }
    
    # Common headers to mimic browser requests
    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    def __init__(self, docs_folder: str):
        """
        Initialize WebImporter
        
        Args:
            docs_folder: Path to the documents folder where files will be saved
        """
        self.docs_folder = Path(docs_folder)
        self.docs_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"WebImporter initialized with docs folder: {self.docs_folder}")
    
    def is_valid_url(self, url: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate URL and detect file type
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_valid, message, detected_extension)
        """
        try:
            parsed = urlparse(url.strip())
            
            if not parsed.scheme in ['http', 'https']:
                return False, "Only HTTP/HTTPS URLs are supported", None
            
            if not parsed.netloc:
                return False, "Invalid URL format", None
            
            # Try to detect file extension from URL path
            path = unquote(parsed.path)
            ext = Path(path).suffix.lower()
            
            if ext in self.SUPPORTED_EXTENSIONS:
                return True, f"Valid {ext.upper()} file detected from URL", ext
            
            # If no extension in URL, we'll try to detect from headers later
            return True, "URL appears valid, will detect file type during download", None
            
        except Exception as e:
            return False, f"Invalid URL: {str(e)}", None
    
    def get_filename_from_url(self, url: str, custom_filename: Optional[str] = None) -> str:
        """
        Extract or generate filename from URL
        
        Args:
            url: Source URL
            custom_filename: Optional custom filename
            
        Returns:
            Final filename to use
        """
        if custom_filename:
            # Ensure custom filename has an extension
            if not Path(custom_filename).suffix:
                # Try to get extension from URL
                parsed_path = unquote(urlparse(url).path)
                url_ext = Path(parsed_path).suffix
                if url_ext:
                    custom_filename += url_ext
                else:
                    custom_filename += '.txt'  # Default extension
            return custom_filename
        
        # Extract filename from URL
        parsed_path = unquote(urlparse(url).path)
        filename = Path(parsed_path).name
        
        if not filename or filename == '/':
            # Generate filename from domain and timestamp
            domain = urlparse(url).netloc.replace('www.', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{domain}_{timestamp}.txt"
        
        return filename
    
    def detect_content_type(self, response: requests.Response, url: str) -> Optional[str]:
        """
        Detect file extension from HTTP response headers
        
        Args:
            response: HTTP response object
            url: Original URL
            
        Returns:
            Detected file extension or None
        """
        # Try Content-Type header
        content_type = response.headers.get('content-type', '').lower()
        
        # Map common content types to extensions
        content_type_map = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            'application/msword': '.doc',
            'text/plain': '.txt',
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/gif': '.gif',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff'
        }
        
        for mime_type, extension in content_type_map.items():
            if mime_type in content_type:
                return extension
        
        # Try Content-Disposition header
        content_disposition = response.headers.get('content-disposition', '')
        if 'filename=' in content_disposition:
            try:
                filename_part = content_disposition.split('filename=')[1]
                filename_part = filename_part.strip('"\'')
                detected_ext = Path(filename_part).suffix.lower()
                if detected_ext in self.SUPPORTED_EXTENSIONS:
                    return detected_ext
            except:
                pass
        
        return None
    
    def download_from_url(self, url: str, custom_filename: Optional[str] = None) -> Tuple[bool, str, ImportRecord]:
        """
        Download file from URL to docs folder
        
        Args:
            url: URL to download from
            custom_filename: Optional custom filename
            
        Returns:
            Tuple of (success, message, import_record)
        """
        import_record = ImportRecord(
            timestamp=datetime.now(),
            url=url,
            filename="",
            status="processing"
        )
        
        try:
            logger.info(f"Starting download from: {url}")
            
            # Validate URL
            is_valid, validation_message, detected_ext = self.is_valid_url(url)
            if not is_valid:
                import_record.status = "failed"
                import_record.error_message = validation_message
                return False, validation_message, import_record
            
            # Make HEAD request first to check file info
            try:
                head_response = requests.head(url, headers=self.DEFAULT_HEADERS, timeout=10, allow_redirects=True)
                content_length = head_response.headers.get('content-length')
                if content_length:
                    file_size = int(content_length)
                    # Check if file is too large (limit to 100MB)
                    if file_size > 100 * 1024 * 1024:
                        error_msg = f"File too large: {file_size / (1024*1024):.1f}MB (max 100MB)"
                        import_record.status = "failed"
                        import_record.error_message = error_msg
                        return False, error_msg, import_record
            except:
                # HEAD request failed, proceed with GET
                pass
            
            # Download the file
            response = requests.get(url, headers=self.DEFAULT_HEADERS, timeout=30, stream=True)
            response.raise_for_status()
            
            # Detect file type if not detected from URL
            if not detected_ext:
                detected_ext = self.detect_content_type(response, url)
            
            # Generate filename
            base_filename = self.get_filename_from_url(url, custom_filename)
            
            # Ensure correct extension
            if detected_ext and not base_filename.lower().endswith(detected_ext):
                # Remove any existing extension and add detected one
                base_filename = Path(base_filename).stem + detected_ext
            
            # Ensure filename is unique
            final_filename = self.get_unique_filename(base_filename)
            local_path = self.docs_folder / final_filename
            
            # Download file content
            total_size = 0
            with open(local_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        total_size += len(chunk)
                        
                        # Safety check for file size during download
                        if total_size > 100 * 1024 * 1024:
                            file.close()
                            local_path.unlink()  # Delete partial file
                            error_msg = "File too large (>100MB), download stopped"
                            import_record.status = "failed"
                            import_record.error_message = error_msg
                            return False, error_msg, import_record
            
            # Verify file was downloaded and is not empty
            if not local_path.exists() or local_path.stat().st_size == 0:
                error_msg = "Downloaded file is empty or failed to save"
                import_record.status = "failed"
                import_record.error_message = error_msg
                return False, error_msg, import_record
            
            # Final validation - check if it's a supported file type
            if not self.is_supported_file(local_path):
                local_path.unlink()  # Delete unsupported file
                error_msg = f"Unsupported file type detected: {local_path.suffix}"
                import_record.status = "failed"
                import_record.error_message = error_msg
                return False, error_msg, import_record
            
            # Update import record with success info
            import_record.status = "success"
            import_record.filename = final_filename
            import_record.local_path = str(local_path)
            import_record.file_size = local_path.stat().st_size
            
            success_msg = f"Successfully downloaded '{final_filename}' ({self.format_file_size(import_record.file_size)})"
            logger.info(f"Download completed: {local_path}")
            
            return True, success_msg, import_record
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            import_record.status = "failed"
            import_record.error_message = error_msg
            logger.error(f"Download failed: {error_msg}")
            return False, error_msg, import_record
            
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            import_record.status = "failed"
            import_record.error_message = error_msg
            logger.error(f"Unexpected error: {error_msg}")
            return False, error_msg, import_record
    
    def get_unique_filename(self, desired_filename: str) -> str:
        """
        Generate a unique filename if the desired one already exists
        
        Args:
            desired_filename: The preferred filename
            
        Returns:
            Unique filename
        """
        base_path = self.docs_folder / desired_filename
        
        if not base_path.exists():
            return desired_filename
        
        # File exists, generate unique name
        stem = Path(desired_filename).stem
        suffix = Path(desired_filename).suffix
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter}{suffix}"
            new_path = self.docs_folder / new_filename
            if not new_path.exists():
                return new_filename
            counter += 1
    
    def is_supported_file(self, file_path: Path) -> bool:
        """
        Check if the downloaded file is of a supported type
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file type is supported
        """
        extension = file_path.suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human readable format
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def batch_download(self, urls: List[str]) -> List[ImportRecord]:
        """
        Download multiple files from a list of URLs
        
        Args:
            urls: List of URLs to download
            
        Returns:
            List of import records
        """
        records = []
        
        for url in urls:
            url = url.strip()
            if not url:
                continue
                
            success, message, record = self.download_from_url(url)
            records.append(record)
            
            # Small delay between downloads to be respectful
            import time
            time.sleep(0.5)
        
        return records


class WebImportAnalytics:
    """Analytics for web import functionality"""
    
    @staticmethod
    def get_import_summary(import_history: List[ImportRecord]) -> Dict[str, Any]:
        """
        Generate comprehensive import analytics
        
        Args:
            import_history: List of import records
            
        Returns:
            Analytics summary
        """
        if not import_history:
            return {
                "total_import_attempts": 0,
                "successful_imports": 0,
                "success_rate": "N/A",
                "total_files_imported": 0,
                "total_size_downloaded": 0,
                "most_imported_domains": [],
                "recent_imports": []
            }
        
        successful_imports = [r for r in import_history if r.status == "success"]
        failed_imports = [r for r in import_history if r.status == "failed"]
        
        total_size = sum(r.file_size for r in successful_imports if r.file_size)
        
        # Get popular domains
        domains = {}
        for record in import_history:
            try:
                domain = urlparse(record.url).netloc.replace('www.', '')
                domains[domain] = domains.get(domain, 0) + 1
            except:
                continue
        
        popular_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Recent imports (last 10)
        recent_imports = sorted(import_history, key=lambda x: x.timestamp, reverse=True)[:10]
        
        return {
            "total_import_attempts": len(import_history),
            "successful_imports": len(successful_imports),
            "failed_imports": len(failed_imports),
            "success_rate": f"{(len(successful_imports)/len(import_history)*100):.1f}%" if import_history else "N/A",
            "total_files_imported": len(successful_imports),
            "total_size_downloaded": WebImporter.format_file_size(total_size),
            "most_imported_domains": popular_domains,
            "recent_imports": recent_imports
        }


def main():
    """Test the WebImporter"""
    import tempfile
    
    # Create temporary test folder
    with tempfile.TemporaryDirectory() as temp_dir:
        importer = WebImporter(temp_dir)
        
        # Test URL validation
        test_urls = [
            "https://www.example.com/document.pdf",
            "invalid-url",
            "http://example.com/file.docx"
        ]
        
        for url in test_urls:
            is_valid, message, ext = importer.is_valid_url(url)
            print(f"URL: {url}")
            print(f"  Valid: {is_valid}, Message: {message}, Extension: {ext}")
        
        print("\nWebImporter test completed successfully!")


if __name__ == "__main__":
    main()
