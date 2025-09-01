"""
Utility functions for the RAG pipeline.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize text for better processing.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def split_text_into_chunks(text: str, chunk_size: int = 1000, 
                          chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to find a good break point (sentence boundary)
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks

def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Extract basic metadata from text content.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of metadata
    """
    metadata = {
        "length": len(text),
        "word_count": len(text.split()),
        "sentence_count": len(re.split(r'[.!?]+', text)),
        "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
        "has_numbers": bool(re.search(r'\d', text)),
        "has_urls": bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        "language_indicators": {
            "english": bool(re.search(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', text.lower())),
            "technical": bool(re.search(r'\b(algorithm|function|method|class|object|variable|parameter)\b', text.lower())),
            "scientific": bool(re.search(r'\b(research|study|analysis|experiment|hypothesis|conclusion)\b', text.lower()))
        }
    }
    
    return metadata

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def generate_document_hash(content: str, title: str = "") -> str:
    """
    Generate a hash for document identification.
    
    Args:
        content: Document content
        title: Document title
        
    Returns:
        MD5 hash string
    """
    text_to_hash = f"{title}:{content}"
    return hashlib.md5(text_to_hash.encode('utf-8')).hexdigest()

def save_documents_to_json(documents: List[Dict[str, str]], output_path: str):
    """
    Save documents to a JSON file.
    
    Args:
        documents: List of documents with 'title' and 'content' keys
        output_path: Path to save the JSON file
    """
    try:
        # Ensure directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare documents with metadata
        docs_with_metadata = []
        for doc in documents:
            doc_data = {
                "title": doc["title"],
                "content": doc["content"],
                "metadata": {
                    "hash": generate_document_hash(doc["content"], doc["title"]),
                    "created_at": datetime.now().isoformat(),
                    "text_analysis": extract_metadata_from_text(doc["content"])
                }
            }
            docs_with_metadata.append(doc_data)
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs_with_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(documents)} documents to {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error saving documents: {e}")
        raise

def load_documents_from_json(input_path: str) -> List[Dict[str, str]]:
    """
    Load documents from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        List of documents with 'title' and 'content' keys
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract basic document structure
        documents = []
        for item in data:
            if isinstance(item, dict) and "title" in item and "content" in item:
                documents.append({
                    "title": item["title"],
                    "content": item["content"]
                })
        
        logger.info(f"✅ Loaded {len(documents)} documents from {input_path}")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Error loading documents: {e}")
        raise

def create_sample_documents() -> List[Dict[str, str]]:
    """
    Create sample documents for testing and demonstration.
    
    Returns:
        List of sample documents
    """
    sample_docs = [
        {
            "title": "Introduction to Artificial Intelligence",
            "content": """
            Artificial Intelligence (AI) is a broad field of computer science that aims to create systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.
            
            AI can be categorized into several types:
            - Narrow AI: Designed for specific tasks (e.g., facial recognition, language translation)
            - General AI: Possesses human-like intelligence across all domains
            - Superintelligent AI: Surpasses human intelligence in all areas
            
            The field has seen remarkable progress in recent years, particularly in machine learning, deep learning, and natural language processing.
            """
        },
        {
            "title": "Machine Learning Fundamentals",
            "content": """
            Machine Learning is a subset of AI that focuses on algorithms and statistical models that enable computers to improve their performance on a specific task through experience.
            
            Key concepts include:
            - Supervised Learning: Learning from labeled training data
            - Unsupervised Learning: Finding patterns in unlabeled data
            - Reinforcement Learning: Learning through interaction with an environment
            
            Popular algorithms include linear regression, decision trees, support vector machines, and neural networks. The choice of algorithm depends on the nature of the problem, available data, and desired outcomes.
            """
        },
        {
            "title": "Deep Learning and Neural Networks",
            "content": """
            Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.
            
            Neural networks are inspired by biological neurons and consist of:
            - Input Layer: Receives the initial data
            - Hidden Layers: Process information through weighted connections
            - Output Layer: Produces the final result
            
            Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition. Popular architectures include Convolutional Neural Networks (CNNs) for image processing and Recurrent Neural Networks (RNNs) for sequential data.
            """
        },
        {
            "title": "Natural Language Processing",
            "content": """
            Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in meaningful ways.
            
            NLP applications include:
            - Text Classification: Categorizing documents by topic or sentiment
            - Named Entity Recognition: Identifying people, places, and organizations
            - Machine Translation: Converting text between languages
            - Question Answering: Understanding and responding to questions
            - Text Generation: Creating human-like text
            
            Modern NLP heavily relies on transformer models like BERT, GPT, and T5, which have achieved remarkable performance on various language tasks.
            """
        },
        {
            "title": "Vector Databases and Embeddings",
            "content": """
            Vector databases are specialized databases designed to store and search high-dimensional vector representations of data. These vectors, called embeddings, capture the semantic meaning of text, images, or other data types.
            
            Key concepts:
            - Embeddings: Numerical representations that capture meaning
            - Similarity Search: Finding vectors that are close to each other
            - Indexing: Efficient data structures for fast retrieval
            
            Popular vector databases include:
            - ChromaDB: Open-source, embeddable vector database
            - Pinecone: Managed vector database service
            - Weaviate: Vector database with GraphQL interface
            - Qdrant: High-performance vector database
            
            Vector databases are essential for building semantic search systems, recommendation engines, and AI applications that need to understand content similarity.
            """
        }
    ]
    
    # Clean the content
    for doc in sample_docs:
        doc["content"] = clean_text(doc["content"])
    
    return sample_docs

def validate_document_structure(documents: List[Dict[str, str]]) -> bool:
    """
    Validate that documents have the correct structure.
    
    Args:
        documents: List of documents to validate
        
    Returns:
        True if all documents are valid, False otherwise
    """
    if not isinstance(documents, list):
        logger.error("Documents must be a list")
        return False
    
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            logger.error(f"Document {i} must be a dictionary")
            return False
        
        if "title" not in doc or "content" not in doc:
            logger.error(f"Document {i} must have 'title' and 'content' keys")
            return False
        
        if not isinstance(doc["title"], str) or not isinstance(doc["content"], str):
            logger.error(f"Document {i} title and content must be strings")
            return False
        
        if not doc["title"].strip() or not doc["content"].strip():
            logger.error(f"Document {i} title and content cannot be empty")
            return False
    
    logger.info(f"✅ Validated {len(documents)} documents")
    return True

def get_document_statistics(documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Get statistics about a collection of documents.
    
    Args:
        documents: List of documents
        
    Returns:
        Dictionary of statistics
    """
    if not documents:
        return {}
    
    total_chars = sum(len(doc["content"]) for doc in documents)
    total_words = sum(len(doc["content"].split()) for doc in documents)
    
    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_words": total_words,
        "average_chars_per_doc": total_chars / len(documents),
        "average_words_per_doc": total_words / len(documents),
        "shortest_doc_chars": min(len(doc["content"]) for doc in documents),
        "longest_doc_chars": max(len(doc["content"]) for doc in documents),
        "shortest_doc_words": min(len(doc["content"].split()) for doc in documents),
        "longest_doc_words": max(len(doc["content"].split()) for doc in documents)
    }
    
    return stats
