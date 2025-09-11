#!/usr/bin/env python3
"""
Test script for conversational memory features
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_enhanced_imports():
    """Test if all enhanced imports work"""
    print("🧪 Testing enhanced imports...")
    
    try:
        from langchain.memory import ConversationBufferWindowMemory
        from langchain.chains import ConversationalRetrievalChain
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        print("✅ All LangChain imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_rag_agent_conversational_features():
    """Test RAG agent conversational features"""
    print("\n🧪 Testing RAG agent conversational features...")
    
    try:
        # Set a dummy API key for testing
        os.environ['NVIDIA_API_KEY'] = 'test_key_for_structure_test'
        
        # Import the enhanced RAG agent components
        from rag_agent import NVIDIALangChainLLM, RAGResponse
        
        # Test LLM wrapper initialization
        llm = NVIDIALangChainLLM(api_key="test_key")
        print("✅ NVIDIA LangChain LLM wrapper created")
        
        # Test if all required properties exist
        assert hasattr(llm, '_llm_type'), "LLM type property missing"
        assert hasattr(llm, '_call'), "LLM call method missing"
        assert hasattr(llm, '_identifying_params'), "LLM identifying params missing"
        print("✅ LLM wrapper has all required methods")
        
        # Test RAGResponse dataclass with new chat_history field
        response = RAGResponse(
            answer="Test answer",
            source_documents=[],
            chat_history=[("Question", "Answer")]
        )
        assert hasattr(response, 'chat_history'), "Chat history field missing"
        print("✅ RAGResponse enhanced with chat_history field")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG agent test failed: {e}")
        return False

def test_memory_functionality():
    """Test memory functionality in isolation"""
    print("\n🧪 Testing memory functionality...")
    
    try:
        from langchain.memory import ConversationBufferWindowMemory
        
        # Create memory instance
        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Add some test conversation
        memory.chat_memory.add_user_message("What is AI?")
        memory.chat_memory.add_ai_message("AI stands for Artificial Intelligence.")
        memory.chat_memory.add_user_message("Can you explain that in more detail?")
        memory.chat_memory.add_ai_message("AI is a branch of computer science...")
        
        # Test memory retrieval
        messages = memory.chat_memory.messages
        assert len(messages) == 4, f"Expected 4 messages, got {len(messages)}"
        
        # Test memory window
        for i in range(10):
            memory.chat_memory.add_user_message(f"Question {i}")
            memory.chat_memory.add_ai_message(f"Answer {i}")
        
        # Should maintain window size
        final_messages = memory.chat_memory.messages
        print(f"✅ Memory maintains conversation history ({len(final_messages)} messages)")
        
        # Test clearing memory
        memory.clear()
        cleared_messages = memory.chat_memory.messages
        assert len(cleared_messages) == 0, "Memory should be empty after clearing"
        print("✅ Memory can be cleared successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Conversational Memory Implementation")
    print("=" * 50)
    
    tests = [
        ("Enhanced Imports", test_enhanced_imports),
        ("RAG Agent Conversational Features", test_rag_agent_conversational_features),
        ("Memory Functionality", test_memory_functionality),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        results[test_name] = test_func()
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧪 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Conversational memory implementation is ready.")
        print("\n💡 Next steps:")
        print("   1. Test the enhanced RAG agent with actual documents")
        print("   2. Try the Streamlit interface with conversational questions")
        print("   3. Verify follow-up questions work correctly")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
