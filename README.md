# 🏛️ RAG Template for NVIDIA NemoRetriever

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-API-green)](https://build.nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A powerful **Retrieval-Augmented Generation (RAG)** template built with **NVIDIA's embedding models** and **LangChain**. This template provides a complete solution for building AI-powered document Q&A systems with a beautiful web interface.

## 🌟 **Features**

- 🤖 **NVIDIA AI Integration**: Uses NVIDIA's high-quality embedding models
- 📄 **PDF Document Processing**: Automatic loading and intelligent chunking
- 🔍 **Vector Search**: FAISS-based similarity search with persistence
- 💬 **Interactive Web UI**: Beautiful Streamlit interface with chat functionality
- 📊 **Advanced Analytics**: Document statistics and source visualization
- 🔒 **Secure**: Environment-based API key management
- 📱 **Responsive**: Mobile-friendly design
- 🚀 **Production Ready**: Comprehensive error handling and logging

## 🎯 **Perfect For**

- Legal document analysis
- Research paper Q&A systems
- Corporate knowledge bases
- Educational content exploration
- Technical documentation search
- Any domain-specific document collection

## 📋 **Prerequisites**

### **System Requirements**
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space

### **Required Accounts**
- **NVIDIA Developer Account**: For API access to embedding models
- **Git**: For cloning the repository

## 🔑 **Getting NVIDIA API Key**

### **Step 1: Create NVIDIA Developer Account**
1. Visit [build.nvidia.com](https://build.nvidia.com)
2. Click **"Sign Up"** or **"Log In"** if you have an account
3. Complete the registration process
4. Verify your email address

### **Step 2: Generate API Key**
1. After logging in, navigate to your **Dashboard**
2. Click on **"API Keys"** or **"Credentials"**
3. Click **"Generate New API Key"**
4. Give your key a descriptive name (e.g., "RAG-Template-Key")
5. **Copy and save** the API key securely
6. ⚠️ **Important**: Save this key immediately - you won't be able to see it again!

### **Step 3: Verify API Access**
1. Ensure you have access to:
   - **Embedding Models**: `nvidia/nv-embed-v1`
   - **LLM Models**: `meta/llama-3.1-8b-instruct`
2. Check the [NVIDIA API documentation](https://docs.api.nvidia.com) for current model availability

## 🚀 **Quick Start Guide**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever.git
cd RAG-Template-for-NVIDIA-nemoretriever
```

### **Step 2: Set Up Python Environment**

#### **Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv rag_env

# Activate virtual environment
# On Windows:
rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate
```

#### **Option B: Using Conda**
```bash
# Create conda environment
conda create -n rag_env python=3.9
conda activate rag_env
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Configure Environment**
1. **Copy the environment template**:
   ```bash
   cp .env.template .env
   ```

2. **Edit the `.env` file** with your details:
   ```bash
   # Open .env file in your preferred editor
   notepad .env  # Windows
   nano .env     # Linux/macOS
   ```

3. **Add your NVIDIA API key**:
   ```env
   # NVIDIA API Configuration
   NVIDIA_API_KEY=your_nvidia_api_key_here
   
   # Configuration (optional - defaults provided)
   DOCS_FOLDER=Data/Docs
   VECTOR_DB_PATH=./vector_db
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

### **Step 5: Add Your Documents**
1. **Create the documents folder** (if not exists):
   ```bash
   mkdir -p Data/Docs
   ```

2. **Add your PDF files** to the `Data/Docs` folder:
   - Copy your PDF documents into this folder
   - The system will automatically process all PDF files
   - Supported formats: `.pdf`

### **Step 6: Test the System**
```bash
# Run the test suite
python test_rag_system.py
```

### **Step 7: Launch the Web Interface**
```bash
# Start the Streamlit web interface
streamlit run streamlit_app.py
```

Or use the convenient launcher:
```bash
python start_web_interface.py
```

### **Step 8: Access Your RAG System**
1. **Open your browser** and navigate to:
   - **Local**: `http://localhost:8501`
   - **Network**: `http://[your-ip]:8501`

2. **Wait for initialization**:
   - The system will load and process your documents
   - This may take a few minutes for large document collections

3. **Start asking questions**!
   - Type your questions in the chat interface
   - Explore the document statistics
   - View detailed source references

## 📁 **Project Structure**

```
RAG-Template-for-NVIDIA-nemoretriever/
├── 📄 README.md                     # This comprehensive guide
├── 📄 requirements.txt              # Python dependencies
├── 📄 .env.template                 # Environment variables template
├── 📄 .gitignore                   # Git ignore rules
├── 📄 streamlit_app.py             # Main web interface
├── 📄 main.py                      # CLI interface
├── 📄 test_rag_system.py           # System tests
├── 📄 start_web_interface.py       # Web interface launcher
├── 📁 src/                         # Core source code
│   ├── 📄 __init__.py
│   ├── 📄 document_loader.py       # PDF processing
│   ├── 📄 nvidia_embeddings.py     # NVIDIA API integration
│   ├── 📄 vector_database.py       # FAISS vector storage
│   └── 📄 rag_agent.py             # Main RAG pipeline
├── 📁 .streamlit/                  # Streamlit configuration
│   └── 📄 config.toml
├── 📁 Data/                        # Document storage
│   └── 📁 Docs/                    # Place your PDF files here
└── 📁 vector_db/                   # Vector database (auto-created)
```

## 🎮 **Usage Examples**

### **Command Line Interface**
```bash
# Use the CLI version
python main.py
```

### **Web Interface**
1. **Start the web interface**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Sample questions to try**:
   - "What is the main topic of the documents?"
   - "Summarize the key points from [specific document]"
   - "What are the requirements for [specific process]?"
   - "How does [concept A] relate to [concept B]?"

### **Programmatic Usage**
```python
from src.rag_agent import RAGAgent
import os

# Initialize RAG agent
api_key = os.getenv("NVIDIA_API_KEY")
rag_agent = RAGAgent("Data/Docs", api_key)

# Setup knowledge base
rag_agent.setup_knowledge_base()

# Ask questions
response = rag_agent.ask_question("Your question here")
print(response.answer)
print(f"Sources: {len(response.source_documents)}")
```

## 🔧 **Configuration Options**

### **Environment Variables**
| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | Your NVIDIA API key | **Required** |
| `DOCS_FOLDER` | Path to PDF documents | `Data/Docs` |
| `VECTOR_DB_PATH` | Vector database storage | `./vector_db` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` |

### **Customization Options**
- **Chunk Size**: Adjust for different document types
- **Model Selection**: Switch between available NVIDIA models
- **UI Styling**: Modify Streamlit interface in `streamlit_app.py`
- **Processing Logic**: Customize RAG pipeline in `src/rag_agent.py`

## 🧪 **Testing**

### **Run All Tests**
```bash
python test_rag_system.py
```

### **Test Individual Components**
```bash
# Test NVIDIA embeddings
python src/nvidia_embeddings.py

# Test document loader
python src/document_loader.py

# Test vector database
python src/vector_database.py

# Test RAG agent
python src/rag_agent.py
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. API Connection Failed**
```
❌ NVIDIA API connection failed
```
**Solutions**:
- Verify your API key is correct in `.env`
- Check internet connection
- Ensure API key has proper permissions
- Visit [NVIDIA API status page](https://status.nvidia.com)

#### **2. No Documents Found**
```
❌ No PDF files found in Data/Docs
```
**Solutions**:
- Ensure PDF files are in `Data/Docs` folder
- Check file permissions
- Verify files are valid PDFs

#### **3. Memory Issues**
```
❌ Out of memory during processing
```
**Solutions**:
- Reduce `CHUNK_SIZE` in `.env`
- Process fewer documents at once
- Increase system RAM

#### **4. Import Errors**
```
❌ ModuleNotFoundError: No module named 'xyz'
```
**Solutions**:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version compatibility

### **Getting Help**
1. **Check the logs**: Look for detailed error messages in console output
2. **Run tests**: Use `python test_rag_system.py` to diagnose issues
3. **Verify setup**: Ensure all prerequisites are met
4. **Check documentation**: Review NVIDIA API documentation

## 🚀 **Deployment Options**

### **Local Development**
- Use the provided scripts for local testing and development

### **Docker Deployment**
```dockerfile
# Example Dockerfile (create as needed)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

### **Cloud Deployment**
- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Use the provided configuration
- **AWS/GCP/Azure**: Deploy using container services

## 🤝 **Contributing**

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Development Setup**
```bash
# Clone your fork
git clone https://github.com/yourusername/RAG-Template-for-NVIDIA-nemoretriever.git

# Install development dependencies
pip install -r requirements.txt

# Run tests before committing
python test_rag_system.py
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **NVIDIA** for providing excellent AI models and APIs
- **LangChain** for the RAG framework
- **Streamlit** for the beautiful web interface framework
- **FAISS** for efficient vector search capabilities

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever/discussions)
- **Documentation**: Check the `docs/` folder for additional guides

---

**🎉 Happy Building! Create amazing RAG applications with NVIDIA's powerful AI models! 🚀**
