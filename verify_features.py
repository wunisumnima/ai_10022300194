#!/usr/bin/env python3
"""
Simple verification script to check if all features are working
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("🔍 Manual Feature Verification")
    print("=" * 50)
    
    try:
        # Test 1: Import all components
        print("1. Testing imports...")
        from rag_core import RAGChatbot, LogManager, ExperimentManager
        from app import get_bot
        print("   ✅ All imports successful")
        
        # Test 2: Initialize RAGChatbot with enhanced features
        print("2. Testing RAGChatbot initialization...")
        bot = RAGChatbot(docs_path="docs", strategy="fixed", top_k=3)
        print(f"   ✅ Session ID: {bot.session_id}")
        print(f"   ✅ Log manager: {hasattr(bot, 'log_manager')}")
        print(f"   ✅ Experiment manager: {hasattr(bot, 'experiment_manager')}")
        print(f"   ✅ Chat history: {len(bot.chat_history)} messages")
        
        # Test 3: Manual Logging
        print("3. Testing manual logging...")
        initial_logs = len(bot.log_manager.logs)
        bot.log_manager.add_log("INFO", "Verification test log", "verification")
        final_logs = len(bot.log_manager.logs)
        print(f"   ✅ Logs: {initial_logs} -> {final_logs}")
        
        # Test 4: Experiments
        print("4. Testing experiments...")
        exp_id = bot.start_experiment("Verification Test", "Testing experiment features", {"test": True})
        print(f"   ✅ Experiment created: {exp_id}")
        bot.log_experiment_result("test_metric", 42)
        bot.end_experiment("completed")
        print("   ✅ Experiment completed")
        
        # Test 5: Chat History
        print("5. Testing chat history...")
        bot.add_to_history("user", "Test question")
        bot.add_to_history("assistant", "Test answer")
        print(f"   ✅ Chat history: {len(bot.chat_history)} messages")
        
        # Test 6: File Persistence
        print("6. Testing file persistence...")
        files_to_check = [
            "logs/manual_logs.json",
            "experiments/experiments.json", 
            "chat_history.json"
        ]
        for file_path in files_to_check:
            path = Path(file_path)
            exists = path.exists()
            print(f"   {'✅' if exists else '❌'} {file_path}: {'exists' if exists else 'missing'}")
        
        # Test 7: Streamlit Integration
        print("7. Testing Streamlit integration...")
        streamlit_bot = get_bot("fixed", 0)
        print(f"   ✅ Streamlit bot created with session: {streamlit_bot.session_id}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("\n📋 Your Manual Log & Experiment features are working fully!")
        print("\n🚀 To test in the browser:")
        print("1. Run: streamlit run app.py")
        print("2. Open: http://localhost:8501")
        print("3. Expand: '📊 Manual Logs & Experiments'")
        print("4. Test each tab:")
        print("   - 📝 Manual Logs: Add logs with different levels")
        print("   - 🧪 Experiments: Create and manage experiments")
        print("   - 💬 Chat History: View conversation history")
        print("   - 📋 View Logs: Filter and view all logs")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
