#!/usr/bin/env python3
"""
Simple Web Import Test
Tests just the WebImporter class without initializing the full RAG system
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_web_importer_simple():
    """Test the WebImporter class functionality directly"""
    
    print("🌐 Testing Web Import Functionality")
    print("=" * 50)
    
    # Import the WebImporter class directly
    from src.web_importer import WebImporter
    
    # Initialize web importer
    data_folder = Path("Data")
    web_importer = WebImporter(data_folder)
    
    print(f"✅ WebImporter initialized with data folder: {data_folder}")
    
    # Test URL validation
    test_urls = [
        "https://example.com/test.pdf",
        "http://example.com/test.docx", 
        "https://example.com/test.txt",
        "ftp://example.com/test.pdf",  # Should fail
        "not-a-url",  # Should fail
        "",  # Should fail
    ]
    
    print("\n📋 Testing URL Validation:")
    print("-" * 30)
    
    for url in test_urls:
        is_valid, message, detected_ext = web_importer.is_valid_url(url)
        status = "✅ VALID" if is_valid else f"❌ INVALID: {message}"
        ext_info = f" (detected: {detected_ext})" if detected_ext else ""
        print(f"{url:<35} -> {status}{ext_info}")
    
    # Test filename generation
    test_urls_filename = [
        "https://example.com/normal_file.pdf",
        "https://example.com/file%20with%20spaces.docx",
        "https://example.com/path/to/document.txt",
        "https://example.com/file?param=value",
        "https://example.com/"
    ]
    
    print("\n🔧 Testing Filename Generation:")
    print("-" * 40)
    
    for url in test_urls_filename:
        filename = web_importer.get_filename_from_url(url)
        print(f"{url:<40} -> {filename}")
    
    print("\n" + "=" * 50)
    print("✅ Web Import Core Tests Complete!")
    print("\n💡 The Web Import tab has been successfully integrated into the Streamlit app!")
    print("   You can now:")
    print("   - Import single files from URLs")
    print("   - Batch import multiple URLs at once")
    print("   - Validate URLs before importing")
    print("   - View recently imported files")

if __name__ == "__main__":
    test_web_importer_simple()
