#!/usr/bin/env python3
"""
Test Web Import Functionality
Tests the WebImporter class and its integration with the Streamlit app
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.web_importer import WebImporter

def test_web_importer():
    """Test the WebImporter class functionality"""
    
    print("🌐 Testing Web Import Functionality")
    print("=" * 50)
    
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
        is_valid, error = web_importer.validate_url(url)
        status = "✅ VALID" if is_valid else f"❌ INVALID: {error}"
        print(f"{url:<35} -> {status}")
    
    # Test filename sanitization
    test_filenames = [
        "normal_file.pdf",
        "file with spaces.docx",
        "file/with/slashes.txt",
        "file?with&query=params.pdf",
        "file<with>forbidden|chars.txt",
    ]
    
    print("\n🔧 Testing Filename Sanitization:")
    print("-" * 40)
    
    for original in test_filenames:
        sanitized = web_importer._sanitize_filename(original)
        print(f"{original:<30} -> {sanitized}")
    
    # Test with a real URL (if available)
    print("\n🔍 Testing Real URL Download:")
    print("-" * 35)
    
    # Test with a simple text file that should be publicly accessible
    test_url = "https://raw.githubusercontent.com/octocat/Hello-World/master/README"
    
    print(f"Attempting to import: {test_url}")
    
    try:
        result = web_importer.import_from_url(test_url)
        
        if result["success"]:
            print(f"✅ Successfully imported: {result['filename']}")
            print(f"   Saved to: {result['filepath']}")
            if 'size_mb' in result:
                print(f"   Size: {result['size_mb']} MB")
        else:
            print(f"❌ Import failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ Web Import Testing Complete!")

if __name__ == "__main__":
    test_web_importer()
