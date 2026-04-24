#!/usr/bin/env python3
"""Test to verify the experiment_manager fix"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_experiment_manager_fix():
    """Test that RAGChatbot has experiment_manager"""
    try:
        from rag_core import RAGChatbot
        
        print("🔍 Testing experiment_manager fix...")
        
        # Create bot instance
        bot = RAGChatbot(docs_path="docs", strategy="fixed", top_k=3)
        
        # Check for required attributes
        required_attrs = ['log_manager', 'experiment_manager', 'chat_history']
        
        for attr in required_attrs:
            if hasattr(bot, attr):
                print(f"✅ {attr}: exists")
            else:
                print(f"❌ {attr}: missing")
                return False
        
        # Test experiment manager functionality
        experiments = bot.experiment_manager.list_experiments()
        print(f"✅ experiment_manager.list_experiments(): works ({len(experiments)} experiments)")
        
        # Test log manager functionality
        logs = bot.log_manager.get_logs()
        print(f"✅ log_manager.get_logs(): works ({len(logs)} logs)")
        
        # Test chat history
        history = bot.chat_history
        print(f"✅ chat_history: works ({len(history)} messages)")
        
        print("\n🎉 Fix verified! The experiment_manager error should be resolved.")
        print("\n📋 To test in Streamlit:")
        print("1. Close any running Streamlit app")
        print("2. Run: streamlit run app.py")
        print("3. Open the '📊 Manual Logs & Experiments' section")
        print("4. Test the Experiments tab")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_experiment_manager_fix()
    sys.exit(0 if success else 1)
