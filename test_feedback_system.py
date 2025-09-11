#!/usr/bin/env python3
"""
Test script for the feedback system implementation
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.rag_agent import FeedbackAnalytics

def test_feedback_analytics():
    """Test the FeedbackAnalytics functionality"""
    print("🧪 Testing Feedback Analytics System...")
    
    # Create sample chat messages
    sample_messages = [
        {
            "role": "user",
            "content": "What is machine learning?",
            "timestamp": datetime.now()
        },
        {
            "role": "assistant",
            "content": "Machine learning is a subset of artificial intelligence...",
            "message_id": str(uuid.uuid4()),
            "question": "What is machine learning?",
            "feedback": "like",
            "feedback_timestamp": datetime.now().isoformat(),
            "type": "chat",
            "processing_time": 2.3
        },
        {
            "role": "user",
            "content": "Can you explain neural networks?",
            "timestamp": datetime.now()
        },
        {
            "role": "assistant",
            "content": "Neural networks are computing systems inspired by biological neural networks...",
            "message_id": str(uuid.uuid4()),
            "question": "Can you explain neural networks?",
            "feedback": "dislike",
            "feedback_timestamp": datetime.now().isoformat(),
            "type": "chat",
            "processing_time": 1.8
        },
        {
            "role": "user",
            "content": "What about deep learning?",
            "timestamp": datetime.now()
        },
        {
            "role": "assistant",
            "content": "Deep learning is a subset of machine learning...",
            "message_id": str(uuid.uuid4()),
            "question": "What about deep learning?",
            "feedback": None,  # No feedback yet
            "feedback_timestamp": None,
            "type": "chat",
            "processing_time": 2.1
        }
    ]
    
    # Test feedback summary
    feedback_summary = FeedbackAnalytics.get_feedback_summary(sample_messages)
    
    print("✅ Feedback Summary Generated:")
    print(f"  Total Responses: {feedback_summary['total_responses']}")
    print(f"  Feedback Received: {feedback_summary['feedback_received']}")
    print(f"  Feedback Rate: {feedback_summary['feedback_rate']}")
    print(f"  Liked Responses: {feedback_summary['liked_responses']}")
    print(f"  Disliked Responses: {feedback_summary['disliked_responses']}")
    print(f"  Satisfaction Score: {feedback_summary['satisfaction_score']}")
    
    # Validate results
    assert feedback_summary['total_responses'] == 3
    assert feedback_summary['feedback_received'] == 2
    assert feedback_summary['liked_responses'] == 1
    assert feedback_summary['disliked_responses'] == 1
    assert len(feedback_summary['detailed_feedback']) == 2
    
    print("✅ All tests passed!")
    return True

def test_edge_cases():
    """Test edge cases for feedback analytics"""
    print("\n🧪 Testing Edge Cases...")
    
    # Test with no messages
    empty_summary = FeedbackAnalytics.get_feedback_summary([])
    assert empty_summary['total_responses'] == 0
    print("✅ Empty messages test passed")
    
    # Test with only user messages
    user_only = [{"role": "user", "content": "Hello"}]
    user_summary = FeedbackAnalytics.get_feedback_summary(user_only)
    assert user_summary['total_responses'] == 0
    print("✅ User-only messages test passed")
    
    # Test with assistant messages but no feedback
    no_feedback = [
        {
            "role": "assistant",
            "content": "Hello",
            "type": "chat",
            "feedback": None
        }
    ]
    no_feedback_summary = FeedbackAnalytics.get_feedback_summary(no_feedback)
    assert no_feedback_summary['feedback_received'] == 0
    assert no_feedback_summary['satisfaction_score'] == "N/A"
    print("✅ No feedback test passed")
    
    print("✅ All edge case tests passed!")
    return True

def main():
    """Run all feedback system tests"""
    print("🚀 Testing Feedback System Implementation")
    print("=" * 50)
    
    try:
        test_feedback_analytics()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Feedback system is working correctly")
        print("✅ Ready for production use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
