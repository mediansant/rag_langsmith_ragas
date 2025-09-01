#!/usr/bin/env python3
"""
Basic test script for the RAG pipeline project.
Run this to verify that all components are working correctly.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        from config.config import Config
        print("  ✅ Config module imported")
        
        from src.utils import create_sample_documents
        print("  ✅ Utils module imported")
        
        print("  ✅ All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        config = Config()
        print("  ✅ Config object created")
        
        # Check if required fields exist
        required_fields = [
            'OPENAI_API_KEY', 'LANGSMITH_API_KEY', 'CHUNK_SIZE', 
            'TOP_K_RETRIEVAL', 'EMBEDDING_MODEL'
        ]
        
        for field in required_fields:
            if hasattr(config, field):
                print(f"  ✅ {field} exists")
            else:
                print(f"  ❌ {field} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\n🛠️ Testing utility functions...")
    
    try:
        from src.utils import create_sample_documents, get_document_statistics
        
        # Test document creation
        docs = create_sample_documents()
        print(f"  ✅ Created {len(docs)} sample documents")
        
        # Test statistics
        stats = get_document_statistics(docs)
        print(f"  ✅ Generated statistics: {len(stats)} metrics")
        
        # Test document validation
        from src.utils import validate_document_structure
        is_valid = validate_document_structure(docs)
        print(f"  ✅ Document validation: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utils error: {e}")
        return False

def test_sample_data():
    """Test sample data generation."""
    print("\n📚 Testing sample data...")
    
    try:
        from config.config import Config
        config = Config()
        
        sample_docs = config.SAMPLE_DOCUMENTS
        print(f"  ✅ Sample documents: {len(sample_docs)}")
        
        for i, doc in enumerate(sample_docs):
            print(f"    {i+1}. {doc['title']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sample data error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RAG Pipeline Project - Basic Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_utils,
        test_sample_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {total - passed}")
    print(f"  📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! The project is ready to use.")
        print("\n💡 Next steps:")
        print("  • Set your API keys in environment variables")
        print("  • Run 'python demo.py' for a full demonstration")
        print("  • Run 'streamlit run src/streamlit_app.py' for the web interface")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
