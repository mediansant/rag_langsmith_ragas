"""
Core RAG Pipeline implementation with LangSmith integration.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    A complete RAG pipeline with document processing, vector search, and generation.
    """
    
    def __init__(self, documents: Optional[List[Dict[str, str]]] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            documents: List of documents with 'title' and 'content' keys
        """
        self.config = Config()
        self.documents = documents or self.config.SAMPLE_DOCUMENTS
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.OPENAI_API_KEY,
            openai_api_base=self.config.OPENAI_BASE_URL,
            model="text-embedding-ada-002"
        )
        
        self.llm = ChatOpenAI(
            openai_api_key=self.config.OPENAI_API_KEY,
            openai_api_base=self.config.OPENAI_BASE_URL,
            model=self.config.OPENAI_MODEL,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS
        )
        
        # Initialize vector store
        self.vector_store = None
        self.retriever = None
        
        # Build the pipeline
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build the RAG pipeline components."""
        try:
            # Process documents
            processed_docs = self._process_documents()
            
            # Create vector store
            self._create_vector_store(processed_docs)
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.TOP_K_RETRIEVAL}
            )
            
            # Create the RAG chain
            self._create_rag_chain()
            
            logger.info("✅ RAG pipeline built successfully")
            
        except Exception as e:
            logger.error(f"❌ Error building RAG pipeline: {e}")
            raise
    
    def _process_documents(self) -> List[Document]:
        """Process and chunk documents."""
        logger.info(f"Processing {len(self.documents)} documents...")
        
        processed_docs = []
        for doc in self.documents:
            # Create LangChain Document objects
            langchain_doc = Document(
                page_content=doc["content"],
                metadata={"title": doc["title"], "source": "sample_data"}
            )
            processed_docs.append(langchain_doc)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(processed_docs)
        logger.info(f"Created {len(chunks)} chunks from documents")
        
        return chunks
    
    def _create_vector_store(self, documents: List[Document]):
        """Create and populate the vector store."""
        logger.info("Creating vector store...")
        
        # Ensure directory exists
        persist_dir = Path(self.config.CHROMA_PERSIST_DIRECTORY)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(persist_dir)
        )
        
        # Persist the vector store
        self.vector_store.persist()
        logger.info(f"Vector store created and persisted to {persist_dir}")
    
    def _create_rag_chain(self):
        """Create the RAG chain with prompt template."""
        # Define the prompt template
        template = """You are a helpful AI assistant. Use the following context to answer the question at the end.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question based on the context provided. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate.
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            The generated answer
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Get the answer using the RAG chain
            answer = self.rag_chain.invoke(question)
            
            logger.info("✅ Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_relevant_context(self, question: str) -> List[Document]:
        """
        Get relevant context for a question without generating an answer.
        
        Args:
            question: The question to find context for
            
        Returns:
            List of relevant documents
        """
        try:
            return self.retriever.get_relevant_documents(question)
        except Exception as e:
            logger.error(f"❌ Error retrieving context: {e}")
            return []
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add new documents to the vector store.
        
        Args:
            documents: List of documents with 'title' and 'content' keys
        """
        try:
            logger.info(f"Adding {len(documents)} new documents...")
            
            # Process new documents
            processed_docs = []
            for doc in documents:
                langchain_doc = Document(
                    page_content=doc["content"],
                    metadata={"title": doc["title"], "source": "user_added"}
                )
                processed_docs.append(langchain_doc)
            
            # Split and add to vector store
            chunks = self.text_splitter.split_documents(processed_docs)
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
            
            logger.info(f"✅ Added {len(chunks)} new chunks to vector store")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            raise
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "persist_directory": self.config.CHROMA_PERSIST_DIRECTORY,
                "embedding_model": "text-embedding-ada-002",
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP
            }
        except Exception as e:
            logger.error(f"❌ Error getting vector store info: {e}")
            return {}
