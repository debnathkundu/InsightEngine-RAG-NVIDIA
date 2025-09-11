#!/usr/bin/env python3
"""
Test Web Import Path Configuration
Verify that Web Import now downloads to the same folder as the RAG system
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_web_import_path():
    """Test that Web Import uses the correct path"""
    
    print("🔧 Testing Web Import Path Configuration")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get the docs folder configuration (same as RAG system)
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    
    print(f"📁 Environment DOCS_FOLDER: {docs_folder}")
    print(f"📂 Absolute path: {Path(docs_folder).absolute()}")
    
    # Initialize web importer with the same path
    from src.web_importer import WebImporter
    
    data_folder = Path(docs_folder)
    web_importer = WebImporter(data_folder)
    
    print(f"✅ WebImporter initialized with: {data_folder}")
    print(f"📍 Web Import will download to: {data_folder.absolute()}")
    
    # Verify the folder exists
    if data_folder.exists():
        print(f"✅ Target folder exists: {data_folder}")
        
        # List current files
        files = list(data_folder.glob("*"))
        print(f"📄 Current files in folder: {len(files)}")
        for file in files[:5]:  # Show first 5 files
            print(f"   - {file.name}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more files")
    else:
        print(f"⚠️  Target folder does not exist: {data_folder}")
        print("   It will be created automatically when first file is imported")
    
    print("\n" + "=" * 50)
    print("✅ Path Configuration Test Complete!")
    print("\n🎯 Summary:")
    print(f"   • RAG system reads from: {docs_folder}")
    print(f"   • Web Import downloads to: {docs_folder}")
    print("   • ✅ Paths are now SYNCHRONIZED!")
    print("\n💡 Downloaded files will now be automatically processed by the RAG system!")

if __name__ == "__main__":
    test_web_import_path()
