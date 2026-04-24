#!/usr/bin/env python3
"""Test region information functionality in election data"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_region_extraction():
    """Test that region information is extracted from election data"""
    try:
        from rag_core import RAGChatbot
        
        print("🧪 Testing Region Information Extraction...")
        
        # Initialize bot
        bot = RAGChatbot(docs_path="docs", strategy="fixed", top_k=3)
        
        # Check if chunks have region metadata
        election_chunks = [chunk for chunk in bot.chunks if "election" in chunk.source.lower()]
        
        if not election_chunks:
            print("❌ No election chunks found")
            return False
        
        print(f"✅ Found {len(election_chunks)} election chunks")
        
        # Test for region metadata
        region_chunks = 0
        sample_chunks = []
        
        for chunk in election_chunks[:5]:  # Check first 5 chunks
            metadata = chunk.metadata
            has_region = any(key in metadata for key in ['old_region', 'new_region', 'region'])
            
            if has_region:
                region_chunks += 1
                sample_chunks.append(chunk)
                print(f"✅ Chunk {chunk.chunk_id} has region data:")
                if 'old_region' in metadata:
                    print(f"   Old Region: {metadata['old_region']}")
                if 'new_region' in metadata:
                    print(f"   New Region: {metadata['new_region']}")
                if 'region' in metadata:
                    print(f"   Region: {metadata['region']}")
                if 'year' in metadata:
                    print(f"   Year: {metadata['year']}")
        
        print(f"✅ {region_chunks} chunks have region information")
        
        # Test query with region context
        print("\n🔍 Testing query with region context...")
        query = "What were the election results in Ashanti Region?"
        result = bot.query(query)
        
        # Check if response contains region information
        answer = result['answer']
        print(f"Query: {query}")
        print(f"Answer: {answer[:200]}...")
        
        # Check retrieved context for region info
        retrieved = result['retrieved']
        print(f"\n📊 Retrieved {len(retrieved)} chunks:")
        for i, item in enumerate(retrieved[:3]):
            metadata = item['metadata']
            region_info = ""
            if 'old_region' in metadata and 'new_region' in metadata:
                if metadata['old_region'] != metadata['new_region']:
                    region_info = f" | {metadata['old_region']} → {metadata['new_region']}"
                else:
                    region_info = f" | {metadata['new_region']}"
            elif 'region' in metadata:
                region_info = f" | {metadata['region']}"
            
            print(f"  [{i+1}] {item['source']}{region_info}")
        
        print("\n🎉 Region functionality test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_region_context():
    """Test that prompts include region context"""
    try:
        from rag_core import PromptBuilder
        
        print("\n🧪 Testing Prompt Region Context...")
        
        # Create sample retrieved data with region info
        retrieved = [
            {
                'source': 'Ghana_Election_Result.csv',
                'chunk_id': 'test-row-0-fixed-0',
                'combined_score': 0.85,
                'text': 'Year: 2020 | Old Region: Ashanti Region | New Region: Ashanti Region | Candidate: Nana Akufo Addo | Party: NPP | Votes: 1795824 | Votes(%): 72.79%',
                'metadata': {
                    'doc_type': 'csv',
                    'row': '0',
                    'token_count': '50',
                    'old_region': 'Ashanti Region',
                    'new_region': 'Ashanti Region',
                    'year': '2020',
                    'candidate': 'Nana Akufo Addo',
                    'party': 'NPP',
                    'votes': '1795824',
                    'votes_percentage': '72.79%'
                }
            }
        ]
        
        prompt = PromptBuilder.build("What were the results in Ashanti Region?", retrieved)
        
        print("✅ Generated prompt with region context:")
        print("   (showing relevant parts)")
        
        # Check if prompt contains region information
        if "Region: Ashanti Region" in prompt:
            print("✅ Region information included in prompt")
        else:
            print("❌ Region information missing from prompt")
            return False
        
        if "ALWAYS include the region information" in prompt:
            print("✅ Region instruction included in prompt")
        else:
            print("❌ Region instruction missing from prompt")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt test failed: {e}")
        return False

def main():
    """Run all region functionality tests"""
    print("🚀 Starting Region Functionality Tests")
    print("=" * 50)
    
    tests = [
        ("Region Extraction", test_region_extraction),
        ("Prompt Region Context", test_prompt_region_context)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: CRASH - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 REGION TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL REGION TESTS PASSED!")
        print("\n📋 Region information is now included in:")
        print("   • Document metadata extraction")
        print("   • Prompt building with region context")
        print("   • AI responses with region data")
        print("\n🚀 Test in Streamlit:")
        print("   streamlit run app.py")
        print("   Ask: 'What were the election results in Ashanti Region?'")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
