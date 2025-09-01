#!/usr/bin/env python3
"""
Setup script for the RAG Pipeline with LangSmith & Ragas project.
This script helps users get started by setting up the environment and running initial tests.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print the project header."""
    print("🚀 RAG Pipeline with LangSmith & Ragas")
    print("=" * 50)
    print("Educational Project: Retrieval-Augmented Generation")
    print("with Monitoring and Evaluation")
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ❌ Python {version.major}.{version.minor} detected")
        print("  💡 This project requires Python 3.8 or higher")
        return False
    
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'langchain', 'openai', 'chromadb', 'sentence-transformers',
        'langsmith', 'ragas', 'datasets', 'streamlit', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  ⚠️ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("  ✅ All required packages are available")
    return True

def install_dependencies():
    """Install project dependencies."""
    print("\n📥 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("  ✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error installing dependencies: {e}")
        return False

def create_env_file():
    """Create a .env file template."""
    print("\n🔑 Creating environment file...")
    
    env_template = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# LangSmith Configuration
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=rag-educational-project
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING_V2=true

# RAG Pipeline Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
TEMPERATURE=0.7
MAX_TOKENS=500

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Evaluation Configuration
RAGAS_DATASET_SIZE=100
EVALUATION_METRICS=answer_relevancy,context_relevancy,faithfulness,answer_correctness
"""
    
    env_path = Path(".env")
    if env_path.exists():
        print("  ℹ️ .env file already exists")
        return True
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_template)
        print("  ✅ .env file created")
        print("  💡 Please edit .env with your actual API keys")
        return True
        
    except Exception as e:
        print(f"  ❌ Error creating .env file: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "data/chroma_db",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}/")
    
    return True

def run_basic_tests():
    """Run basic tests to verify setup."""
    print("\n🧪 Running basic tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_basic.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Basic tests passed")
            return True
        else:
            print("  ❌ Basic tests failed")
            print(f"  📝 Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error running tests: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("  1. Edit .env file with your API keys:")
    print("     • OPENAI_API_KEY: Get from https://platform.openai.com/")
    print("     • LANGSMITH_API_KEY: Get from https://smith.langchain.com/")
    print("\n  2. Test the setup:")
    print("     • python test_basic.py")
    print("\n  3. Run the demo:")
    print("     • python demo.py")
    print("\n  4. Start the web interface:")
    print("     • streamlit run src/streamlit_app.py")
    print("\n  5. Explore with Jupyter:")
    print("     • jupyter notebook notebooks/rag_exploration.ipynb")
    print("\n📚 Documentation:")
    print("  • README.md - Project overview and usage")
    print("  • LangSmith: https://docs.smith.langchain.com/")
    print("  • Ragas: https://docs.ragas.io/")
    print("  • LangChain: https://python.langchain.com/")

def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        print("❌ Setup cannot continue. Please upgrade Python.")
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\n📥 Installing missing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install manually:")
            print("   pip install -r requirements.txt")
            return False
    
    # Create environment file
    if not create_env_file():
        print("❌ Failed to create environment file.")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories.")
        return False
    
    # Run tests
    if not run_basic_tests():
        print("❌ Basic tests failed. Please check your setup.")
        return False
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
