# 🚀 Quick Start Guide

Get up and running with the RAG Pipeline project in minutes!

## ⚡ Super Quick Start

1. **Clone and setup:**
   ```bash
   cd rag_langsmith_ragas
   python setup.py
   ```

2. **Set your API keys in `.env`:**
   ```bash
   # Edit .env file with your keys
   OPENAI_API_KEY=your_key_here
   LANGSMITH_API_KEY=your_key_here
   ```

3. **Run the demo:**
   ```bash
   python demo.py
   ```

4. **Start the web interface:**
   ```bash
   streamlit run src/streamlit_app.py
   ```

## 🔑 Getting API Keys

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy and paste into `.env` file

### LangSmith API Key
1. Go to [LangSmith](https://smith.langchain.com/)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy and paste into `.env` file

## 🧪 Test Your Setup

```bash
# Run basic tests
python test_basic.py

# Run full demo
python demo.py
```

## 🌐 Web Interface

The Streamlit app provides:
- **RAG Pipeline**: Ask questions and get answers
- **LangSmith**: Monitor performance and traces
- **Ragas**: Evaluate RAG quality
- **Document Management**: Add/edit documents

## 📚 What You'll Learn

- **RAG Implementation**: Complete pipeline with document processing
- **Vector Search**: Semantic similarity using ChromaDB
- **LangSmith Monitoring**: Request/response tracing and metrics
- **Ragas Evaluation**: Multi-dimensional quality assessment
- **Best Practices**: Production-ready patterns

## 🆘 Common Issues

### "Configuration validation failed"
- Check your `.env` file exists
- Verify API keys are set correctly
- Ensure no extra spaces in values

### "Module not found" errors
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### "API key invalid" errors
- Verify API keys in `.env` file
- Check API key permissions
- Ensure sufficient credits

## 📖 Next Steps

1. **Explore the code**: Check `src/` directory
2. **Customize documents**: Add your own content
3. **Experiment with parameters**: Modify `config/config.py`
4. **Run evaluations**: Test different metrics
5. **Monitor performance**: Check LangSmith dashboard

## 🎯 Learning Path

1. **Start**: Run `demo.py` to see the basics
2. **Explore**: Use Streamlit interface
3. **Understand**: Read the code in `src/`
4. **Customize**: Modify configuration and prompts
5. **Evaluate**: Run Ragas evaluations
6. **Monitor**: Check LangSmith traces
7. **Deploy**: Adapt for your own use case

## 📞 Need Help?

- Check the [README.md](README.md) for detailed documentation
- Review [LangSmith docs](https://docs.smith.langchain.com/)
- Check [Ragas docs](https://docs.ragas.io/)
- Look at the example code in `src/`

---

**Happy Learning! 🎉**
