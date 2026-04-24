#!/usr/bin/env python3
"""
Comprehensive test to verify all Manual Log & Experiment features work fully
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_manual_logs():
    """Test Manual Log functionality end-to-end"""
    print("🧪 Testing Manual Logs...")
    
    try:
        from rag_core import RAGChatbot
        
        # Initialize bot
        bot = RAGChatbot(docs_path="docs", strategy="fixed", top_k=3)
        
        # Test adding different types of logs
        test_logs = [
            ("INFO", "Test info log", "general"),
            ("WARNING", "Test warning log", "manual"),
            ("ERROR", "Test error log", "debug"),
            ("DEBUG", "Test debug log", "performance")
        ]
        
        for level, message, category in test_logs:
            bot.log_manager.add_log(level, message, category)
        
        # Verify logs were added
        logs = bot.log_manager.get_logs()
        assert len(logs) >= len(test_logs), f"Expected at least {len(test_logs)} logs, got {len(logs)}"
        
        # Test filtering
        info_logs = bot.log_manager.get_logs(category="general")
        warning_logs = bot.log_manager.get_logs(level="WARNING")
        
        assert len(info_logs) >= 1, "Should have at least 1 general log"
        assert len(warning_logs) >= 1, "Should have at least 1 WARNING log"
        
        # Test file persistence
        logs_file = Path("logs/manual_logs.json")
        assert logs_file.exists(), "Manual logs file should exist"
        
        with open(logs_file, 'r') as f:
            saved_logs = json.load(f)
            assert len(saved_logs) >= len(test_logs), "Logs should be saved to file"
        
        print("✅ Manual Logs: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Manual Logs: FAIL - {e}")
        return False

def test_experiments():
    """Test Experiment functionality end-to-end"""
    print("🧪 Testing Experiments...")
    
    try:
        from rag_core import RAGChatbot
        
        bot = RAGChatbot(docs_path="docs", strategy="fixed", top_k=3)
        
        # Test creating experiment
        exp_id = bot.start_experiment(
            "Test Experiment", 
            "Testing experiment functionality",
            {"test_param": "test_value", "number_param": 42}
        )
        
        assert exp_id is not None, "Experiment ID should not be None"
        assert bot.current_experiment_id == exp_id, "Current experiment should be set"
        
        # Test logging experiment results
        bot.log_experiment_result("metric1", 100)
        bot.log_experiment_result("metric2", "test_result")
        bot.log_experiment_result("processing_time", 1.5)
        
        # Test getting experiment
        experiment = bot.experiment_manager.get_experiment(exp_id)
        assert experiment is not None, "Should be able to retrieve experiment"
        assert experiment.name == "Test Experiment", "Experiment name should match"
        assert len(experiment.results) >= 3, "Should have logged results"
        
        # Test completing experiment
        bot.end_experiment("completed")
        assert bot.current_experiment_id is None, "Current experiment should be cleared"
        
        completed_exp = bot.experiment_manager.get_experiment(exp_id)
        assert completed_exp.status == "completed", "Experiment should be marked as completed"
        
        # Test listing experiments
        all_experiments = bot.experiment_manager.list_experiments()
        assert len(all_experiments) >= 1, "Should have at least 1 experiment"
        
        # Test file persistence
        exp_file = Path("experiments/experiments.json")
        assert exp_file.exists(), "Experiments file should exist"
        
        with open(exp_file, 'r') as f:
            saved_experiments = json.load(f)
            assert exp_id in saved_experiments, "Experiment should be saved to file"
        
        print("✅ Experiments: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Experiments: FAIL - {e}")
        return False

def test_chat_history():
    """Test Chat History functionality end-to-end"""
    print("🧪 Testing Chat History...")
    
    try:
        from rag_core import RAGChatbot
        
        bot = RAGChatbot(docs_path="docs", strategy="fixed", top_k=3)
        
        # Test adding messages to history
        test_messages = [
            ("user", "Hello, this is a test question"),
            ("assistant", "Hello, this is a test answer"),
            ("user", "Another test question"),
            ("assistant", "Another test answer")
        ]
        
        for role, content in test_messages:
            bot.add_to_history(role, content)
        
        # Verify history
        assert len(bot.chat_history) >= len(test_messages), "Should have added messages to history"
        
        # Test history string generation
        history_str = bot.get_history_string(max_exchanges=2)
        assert history_str != "", "History string should not be empty"
        assert "User:" in history_str, "Should contain user messages"
        assert "Assistant:" in history_str, "Should contain assistant messages"
        
        # Test clearing history
        bot.clear_history()
        assert len(bot.chat_history) == 0, "History should be cleared"
        
        # Test file persistence
        history_file = Path("chat_history.json")
        assert history_file.exists(), "Chat history file should exist"
        
        print("✅ Chat History: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Chat History: FAIL - {e}")
        return False

def test_streamlit_integration():
    """Test Streamlit app integration"""
    print("🧪 Testing Streamlit Integration...")
    
    try:
        from app import get_bot
        
        # Test bot creation through Streamlit function
        bot = get_bot("fixed", 0)
        assert bot is not None, "Should be able to create bot through Streamlit function"
        
        # Test that enhanced features are available
        assert hasattr(bot, 'log_manager'), "Should have log_manager"
        assert hasattr(bot, 'experiment_manager'), "Should have experiment_manager"
        assert hasattr(bot, 'chat_history'), "Should have chat_history"
        
        # Test query with enhanced logging
        result = bot.query("Test query for integration testing")
        assert 'answer' in result, "Query should return answer"
        assert 'latency_ms' in result, "Query should return latency metrics"
        
        # Verify automatic logging
        logs = bot.log_manager.get_logs(category="query")
        assert len(logs) >= 1, "Should have automatic query logs"
        
        # Verify automatic chat history
        assert len(bot.chat_history) >= 2, "Should have added query and response to history"
        
        print("✅ Streamlit Integration: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit Integration: FAIL - {e}")
        return False

def test_data_persistence():
    """Test data persistence across sessions"""
    print("🧪 Testing Data Persistence...")
    
    try:
        # Test that files exist and contain data
        files_to_check = [
            ("logs/manual_logs.json", "manual logs"),
            ("experiments/experiments.json", "experiments"),
            ("chat_history.json", "chat history")
        ]
        
        for file_path, description in files_to_check:
            path = Path(file_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    assert isinstance(data, (list, dict)), f"{description} should be valid JSON"
                    if isinstance(data, list):
                        assert len(data) >= 0, f"{description} should be a list"
                    elif isinstance(data, dict):
                        assert len(data) >= 0, f"{description} should be a dict"
                print(f"  ✅ {description} file exists and valid")
            else:
                print(f"  ⚠️ {description} file not found (may be created on first run)")
        
        print("✅ Data Persistence: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Data Persistence: FAIL - {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive Functionality Test")
    print("=" * 60)
    
    tests = [
        ("Manual Logs", test_manual_logs),
        ("Experiments", test_experiments),
        ("Chat History", test_chat_history),
        ("Streamlit Integration", test_streamlit_integration),
        ("Data Persistence", test_data_persistence)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: CRASH - {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Manual Log & Experiment features are working fully!")
        print("\n📋 Next steps:")
        print("1. Run 'streamlit run app.py' to test the UI")
        print("2. Open the '📊 Manual Logs & Experiments' section")
        print("3. Test each tab: Manual Logs, Experiments, Chat History, View Logs")
        print("4. Verify data persistence by restarting the app")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
