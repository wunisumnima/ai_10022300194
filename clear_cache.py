#!/usr/bin/env python3
"""Clear Streamlit cache to fix the experiment_manager error"""

import shutil
from pathlib import Path

def clear_streamlit_cache():
    """Clear Streamlit cache directory"""
    cache_path = Path(".streamlit")
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print("✅ Streamlit cache cleared")
    else:
        print("ℹ️ No Streamlit cache found")
    
    print("\n🚀 Now restart your Streamlit app:")
    print("   streamlit run app.py")
    print("\n📋 The experiment_manager error should be fixed!")

if __name__ == "__main__":
    clear_streamlit_cache()
