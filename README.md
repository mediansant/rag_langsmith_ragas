# RAG Pipeline with LangSmith & Ragas

This educational project demonstrates how to implement a Retrieval-Augmented Generation (RAG) pipeline with comprehensive monitoring using LangSmith and evaluation using Ragas.

## 🎯 Project Overview

This project showcases:
- **RAG Pipeline**: A complete retrieval-augmented generation system
- **LangSmith Integration**: For monitoring, tracing, and debugging LLM applications
- **Ragas Evaluation**: For comprehensive RAG system evaluation
- **Best Practices**: Production-ready patterns for RAG systems

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources │    │   Vector Store  │    │   LLM (OpenAI)  │
│   (Documents)  │───▶│   (ChromaDB)    │───▶│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   LangSmith     │    │     Ragas       │
                       │   (Monitoring)  │    │  (Evaluation)  │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Features

- **Document Processing**: Automatic text chunking and embedding
- **Vector Search**: Semantic similarity search using ChromaDB
- **LLM Integration**: OpenAI GPT models for generation
- **LangSmith Tracing**: Complete request/response monitoring
- **Ragas Evaluation**: Multi-dimensional RAG quality assessment
- **Streamlit UI**: Interactive demo interface
- **Configuration Management**: Environment-based settings

## 📁 Project Structure

```
rag_langsmith_ragas/
├── src/                    # Source code
│   ├── __init__.py
│   ├── rag_pipeline.py    # Core RAG implementation
│   ├── langsmith_client.py # LangSmith integration
│   ├── ragas_evaluator.py # Ragas evaluation
│   └── utils.py           # Utility functions
├── config/                 # Configuration files
│   └── config.py          # Settings and constants
├── data/                   # Sample data and documents
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd rag_langsmith_ragas
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 🔑 Environment Variables

Create a `.env` file with the following variables:

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith API
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=rag-educational-project
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Optional: Custom OpenAI endpoint
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 🚀 Quick Start

1. **Run the RAG pipeline**:
   ```python
   from src.rag_pipeline import RAGPipeline
   
   pipeline = RAGPipeline()
   response = pipeline.query("What is machine learning?")
   print(response)
   ```

2. **Start the Streamlit demo**:
   ```bash
   streamlit run src/streamlit_app.py
   ```

3. **Run evaluation with Ragas**:
   ```python
   from src.ragas_evaluator import RagasEvaluator
   
   evaluator = RagasEvaluator()
   results = evaluator.evaluate_pipeline()
   print(results)
   ```

## 📊 LangSmith Integration

LangSmith provides:
- **Request/Response Tracing**: Complete visibility into LLM calls
- **Performance Monitoring**: Latency, token usage, and cost tracking
- **Debugging Tools**: Step-by-step execution analysis
- **Dataset Management**: Version control for prompts and responses

## 📈 Ragas Evaluation

Ragas evaluates RAG systems across multiple dimensions:
- **Answer Relevancy**: How well the answer matches the question
- **Context Relevancy**: How relevant the retrieved context is
- **Faithfulness**: How faithful the answer is to the context
- **Answer Correctness**: Factual accuracy of responses

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📚 Learning Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Ragas Documentation](https://docs.ragas.io/)
- [LangChain RAG Tutorials](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different configurations
- Add new evaluation metrics
- Improve the documentation
- Share your learnings

## 📄 License

This project is for educational purposes. Please respect the licenses of all included libraries and tools.

## 🆘 Support

If you encounter issues:
1. Check the [Issues](../../issues) section
2. Review the LangSmith and Ragas documentation
3. Ensure all environment variables are properly set
4. Verify your API keys have sufficient credits/permissions
