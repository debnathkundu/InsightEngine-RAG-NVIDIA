"""
Quick Start Script for DIFC Legal RAG Web Interface
Launches the Streamlit web interface with proper configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'langchain',
        'faiss-cpu',
        'requests',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_environment():
    """Check if environment is properly configured"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("📝 Please create a .env file with your NVIDIA_API_KEY")
        return False
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in .env file")
        print("🔑 Please add your NVIDIA API key to the .env file")
        return False
    
    # Check if docs folder exists
    docs_folder = Path("Data/Docs")
    if not docs_folder.exists():
        print("❌ Data/Docs folder not found")
        print("📁 Please create the Data/Docs folder and add your files")
        return False
    
    all_files = list(docs_folder.rglob("*"))
    supported_extensions = ['.pdf', '.docx', '.pptx', '.txt']
    supported_files = [f for f in all_files if f.is_file() and f.suffix.lower() in supported_extensions]

    if not supported_files:
        print("⚠️  No supported files found in Data/Docs folder")
        print("📄 Please add your files to the Data/Docs folder")
    else:
        print(f"✅ Found {len(supported_files)} supported files in Data/Docs")
    
    return True

def launch_interface():
    """Launch the Streamlit web interface"""
    print("🚀 Launching RAG Web Interface...")
    print("🌐 The interface will open in your default browser")
    print("📱 Access URLs:")
    print("   - Local: http://localhost:8501")
    print("   - Network: http://[your-ip]:8501")
    print("\n💡 Tips:")
    print("   - Wait for the knowledge base to load completely")
    print("   - Try sample questions from the sidebar")
    print("   - Explore the Document Statistics tab")
    print("   - Use Ctrl+C to stop the server")
    print("\n" + "="*50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 RAG Web Interface stopped.")
        print("Thank you for using the system!")
    except Exception as e:
        print(f"\n❌ Error launching interface: {str(e)}")
        print("Please check the error message and try again.")

def main():
    """Main function"""
    print("🤖 RAG Template - Web Interface Launcher")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        return
    
    print("✅ All required packages are installed")
    
    # Check environment
    print("\n🔍 Checking environment configuration...")
    if not check_environment():
        return
    
    print("✅ Environment configuration is valid")
    
    # Launch interface
    print("\n🎯 All checks passed! Ready to launch...")
    input("Press Enter to start the web interface...")
    
    launch_interface()

if __name__ == "__main__":
    main()
