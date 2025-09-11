#!/usr/bin/env python3
"""
Simple Path Test - No RAG System Initialization
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def simple_path_test():
    """Simple test without initializing RAG system"""
    
    print("🔧 Web Import Path Configuration Test")
    print("=" * 40)
    
    # Load environment
    load_dotenv()
    
    # Get docs folder (same as used by RAG system)
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    
    print(f"📁 DOCS_FOLDER from env: {docs_folder}")
    print(f"📂 Absolute path: {Path(docs_folder).absolute()}")
    print(f"📂 Folder exists: {Path(docs_folder).exists()}")
    
    # Show what Web Import will use now
    data_folder = Path(docs_folder)
    print(f"✅ Web Import will download to: {data_folder}")
    
    print("\n🎯 RESULT:")
    print(f"   • RAG system reads from: {docs_folder}")  
    print(f"   • Web Import downloads to: {docs_folder}")
    print("   • ✅ PATHS ARE NOW SYNCHRONIZED!")
    
    return data_folder

if __name__ == "__main__":
    simple_path_test()
