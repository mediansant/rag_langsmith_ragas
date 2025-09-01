"""
Configuration settings for the RAG pipeline with LangSmith and Ragas.
"""
import os
from typing import List

class Config:
    """Configuration class for the RAG pipeline."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # LangSmith Configuration
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-educational-project")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2", "true").lower() == "true"
    
    # RAG Pipeline Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
    
    # Vector Store Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Evaluation Configuration
    RAGAS_DATASET_SIZE = int(os.getenv("RAGAS_DATASET_SIZE", "100"))
    EVALUATION_METRICS = os.getenv("EVALUATION_METRICS", "answer_relevancy,context_relevancy,faithfulness,answer_correctness").split(",")
    
    # Sample Documents (for educational purposes)
    SAMPLE_DOCUMENTS = [
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns."
        },
        {
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. These neural networks are inspired by the human brain and can automatically learn representations from data such as images, text, or sound."
        },
        {
            "title": "Natural Language Processing",
            "content": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a way that is both meaningful and useful."
        },
        {
            "title": "Retrieval-Augmented Generation",
            "content": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It first retrieves relevant documents or information from a knowledge base, then uses that context to generate more accurate and informative responses."
        },
        {
            "title": "Vector Databases and Embeddings",
            "content": "Vector databases store and search high-dimensional vector representations of data, enabling semantic similarity search. Embeddings are numerical representations of text, images, or other data that capture their meaning in a way that similar items have similar vectors."
        }
    ]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        required_vars = [
            "OPENAI_API_KEY",
            "LANGSMITH_API_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("Please set these variables in your .env file")
            return False
        
        print("✅ Configuration validated successfully")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)."""
        print("\n🔧 Current Configuration:")
        print(f"  OpenAI Model: {cls.OPENAI_MODEL}")
        print(f"  LangSmith Project: {cls.LANGSMITH_PROJECT}")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Top-K Retrieval: {cls.TOP_K_RETRIEVAL}")
        print(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"  Evaluation Metrics: {', '.join(cls.EVALUATION_METRICS)}")
        print()
