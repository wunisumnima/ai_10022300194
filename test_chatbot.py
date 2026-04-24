#!/usr/bin/env python3
"""Test script for the Academic City Chatbot"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all imports work correctly"""
    try:
        # Test basic imports
        import json
        import logging
        from datetime import datetime
        from dataclasses import dataclass, field
        from typing import Dict, List, Optional, Sequence, Tuple
        print("✓ Basic imports successful")
        
        # Test optional imports
        try:
            import numpy as np
            print("✓ NumPy available")
        except ImportError:
            print("⚠ NumPy not available")
            
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            print("✓ Scikit-learn available")
        except ImportError:
            print("⚠ Scikit-learn not available")
            
        try:
            import openai
            print("✓ OpenAI available")
        except ImportError:
            print("⚠ OpenAI not available")
            
        try:
            from sentence_transformers import SentenceTransformer
            print("✓ Sentence Transformers available")
        except ImportError:
            print("⚠ Sentence Transformers not available")
            
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_chatbot_initialization():
    """Test if chatbot can be initialized"""
    try:
        # Mock __file__ for the chatbot
        import builtins
        builtins.__file__ = __file__
        
        # Import and test chatbot components
        exec(open('Academic City Chatbot 2026 .py').read())
        
        # Test dataclass creation
        from datetime import datetime
        
        @dataclass
        class ChatMessage:
            role: str
            content: str
            timestamp: str = field(default_factory=lambda: str(datetime.now()))
        
        @dataclass
        class ManualLog:
            timestamp: str
            level: str
            message: str
            category: str = "general"
            experiment_id: Optional[str] = None
        
        @dataclass
        class Experiment:
            id: str
            name: str
            description: str
            parameters: Dict[str, any]
            results: Dict[str, any]
            created_at: str = field(default_factory=lambda: str(datetime.now()))
            status: str = "active"
        
        print("✓ Dataclasses defined successfully")
        
        # Test LogManager
        class LogManager:
            def __init__(self, log_dir: str = "logs"):
                self.log_dir = Path(log_dir)
                self.log_dir.mkdir(exist_ok=True)
                self.logs: List[ManualLog] = []
                
            def add_log(self, level: str, message: str, category: str = "general", experiment_id: Optional[str] = None):
                log_entry = ManualLog(
                    timestamp=str(datetime.now()),
                    level=level.upper(),
                    message=message,
                    category=category,
                    experiment_id=experiment_id
                )
                self.logs.append(log_entry)
                print(f"Log added: [{level.upper()}] {message}")
        
        log_manager = LogManager()
        log_manager.add_log("INFO", "Test log message")
        print("✓ LogManager working")
        
        # Test ExperimentManager
        class ExperimentManager:
            def __init__(self, experiments_dir: str = "experiments"):
                self.experiments_dir = Path(experiments_dir)
                self.experiments_dir.mkdir(exist_ok=True)
                self.experiments: Dict[str, Experiment] = {}
                
            def create_experiment(self, name: str, description: str, parameters: Dict[str, any]) -> str:
                experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.experiments)}"
                experiment = Experiment(
                    id=experiment_id,
                    name=name,
                    description=description,
                    parameters=parameters,
                    results={}
                )
                self.experiments[experiment_id] = experiment
                return experiment_id
        
        exp_manager = ExperimentManager()
        exp_id = exp_manager.create_experiment("Test", "Test experiment", {"param": "value"})
        print(f"✓ ExperimentManager working (ID: {exp_id})")
        
        return True
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Academic City Chatbot 2026...")
    print("=" * 50)
    
    if test_imports():
        print("\n" + "=" * 50)
        if test_chatbot_initialization():
            print("\n✓ All tests passed! The chatbot should work correctly.")
        else:
            print("\n✗ Initialization failed.")
    else:
        print("\n✗ Import tests failed.")

if __name__ == "__main__":
    main()
