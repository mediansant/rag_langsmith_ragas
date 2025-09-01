"""
Streamlit web application for the RAG pipeline with LangSmith and Ragas.
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.rag_pipeline import RAGPipeline
from src.langsmith_client import LangSmithClient
from src.ragas_evaluator import RagasEvaluator
from src.utils import create_sample_documents, save_documents_to_json, load_documents_from_json
from config.config import Config

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline with LangSmith & Ragas",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Pipeline with LangSmith & Ragas</h1>', unsafe_allow_html=True)
    st.markdown("### Educational Project: Retrieval-Augmented Generation with Monitoring and Evaluation")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Check configuration
        if st.button("Check Configuration"):
            config = Config()
            if config.validate():
                st.success("✅ Configuration is valid!")
                config.print_config()
            else:
                st.error("❌ Configuration validation failed!")
        
        st.markdown("---")
        st.header("📚 Quick Actions")
        
        if st.button("Create Sample Documents"):
            create_sample_docs_action()
        
        if st.button("Run Quick Evaluation"):
            run_quick_evaluation_action()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home", 
        "🔍 RAG Pipeline", 
        "📊 LangSmith", 
        "📈 Ragas Evaluation", 
        "📁 Document Management"
    ])
    
    with tab1:
        show_home_tab()
    
    with tab2:
        show_rag_tab()
    
    with tab3:
        show_langsmith_tab()
    
    with tab4:
        show_ragas_tab()
    
    with tab5:
        show_documents_tab()

def show_home_tab():
    """Display the home tab with project overview."""
    st.header("🏠 Welcome to the RAG Pipeline Project!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This educational project demonstrates how to build a complete **Retrieval-Augmented Generation (RAG)** pipeline 
        with comprehensive monitoring using **LangSmith** and evaluation using **Ragas**.
        
        ### 🎯 What You'll Learn
        
        - **RAG Pipeline Implementation**: Complete document processing, vector search, and generation
        - **LangSmith Integration**: Monitor, trace, and debug your LLM applications
        - **Ragas Evaluation**: Assess RAG quality across multiple dimensions
        - **Best Practices**: Production-ready patterns for RAG systems
        
        ### 🚀 Getting Started
        
        1. **Check Configuration**: Ensure your API keys are set in the sidebar
        2. **Create Documents**: Add sample documents or your own content
        3. **Test the Pipeline**: Try asking questions in the RAG Pipeline tab
        4. **Monitor Performance**: View traces and metrics in LangSmith
        5. **Evaluate Quality**: Run comprehensive evaluations with Ragas
        """)
    
    with col2:
        st.markdown("### 📊 Project Stats")
        
        # Try to get pipeline info
        try:
            pipeline = RAGPipeline()
            info = pipeline.get_vector_store_info()
            
            if info:
                st.metric("Documents", info.get("total_documents", "N/A"))
                st.metric("Chunk Size", info.get("chunk_size", "N/A"))
                st.metric("Embedding Model", "text-embedding-ada-002")
            else:
                st.info("Pipeline not initialized yet")
        except Exception as e:
            st.warning(f"Pipeline not ready: {str(e)}")
    
    st.markdown("---")
    
    # Quick demo section
    st.header("🎮 Quick Demo")
    
    if st.button("🚀 Initialize Pipeline"):
        with st.spinner("Initializing RAG pipeline..."):
            try:
                pipeline = RAGPipeline()
                st.success("✅ RAG pipeline initialized successfully!")
                
                # Show sample question
                st.markdown("### Try asking a question:")
                sample_question = "What is machine learning?"
                st.code(sample_question)
                
                if st.button("Ask Sample Question"):
                    with st.spinner("Generating answer..."):
                        answer = pipeline.query(sample_question)
                        st.markdown("**Answer:**")
                        st.write(answer)
                        
            except Exception as e:
                st.error(f"❌ Error initializing pipeline: {str(e)}")

def show_rag_tab():
    """Display the RAG pipeline tab."""
    st.header("🔍 RAG Pipeline")
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline()
        st.success("✅ RAG pipeline ready!")
        
        # Query interface
        st.subheader("Ask Questions")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is machine learning?",
                key="rag_question"
            )
        
        with col2:
            if st.button("🔍 Search", type="primary"):
                if question:
                    process_rag_query(pipeline, question)
                else:
                    st.warning("Please enter a question")
        
        # Show recent questions
        if "recent_questions" not in st.session_state:
            st.session_state.recent_questions = []
        
        if st.session_state.recent_questions:
            st.subheader("📝 Recent Questions")
            for i, q in enumerate(st.session_state.recent_questions[-5:]):
                if st.button(f"🔍 {q[:50]}...", key=f"recent_{i}"):
                    process_rag_query(pipeline, q)
        
        # Pipeline information
        st.subheader("📊 Pipeline Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            info = pipeline.get_vector_store_info()
            if info:
                st.metric("Total Documents", info.get("total_documents", "N/A"))
        
        with col2:
            st.metric("Chunk Size", Config().CHUNK_SIZE)
        
        with col3:
            st.metric("Top-K Retrieval", Config().TOP_K_RETRIEVAL)
        
    except Exception as e:
        st.error(f"❌ Error initializing RAG pipeline: {str(e)}")
        st.info("Please check your configuration and try again.")

def process_rag_query(pipeline, question):
    """Process a RAG query and display results."""
    with st.spinner("Processing your question..."):
        try:
            # Get relevant context
            context_docs = pipeline.get_relevant_context(question)
            context_texts = [doc.page_content for doc in context_docs]
            
            # Generate answer
            answer = pipeline.query(question)
            
            # Store question in recent questions
            if question not in st.session_state.recent_questions:
                st.session_state.recent_questions.append(question)
            
            # Display results
            st.markdown("### 📋 Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**🤖 Generated Answer:**")
                st.markdown(f"<div class='success-box'>{answer}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("**📚 Retrieved Context:**")
                for i, context in enumerate(context_texts[:3]):  # Show top 3 contexts
                    st.markdown(f"**Context {i+1}:**")
                    st.markdown(f"<div class='info-box'>{context[:200]}...</div>", unsafe_allow_html=True)
            
            # Show full context if requested
            if st.button("Show Full Context"):
                st.markdown("**📚 Full Retrieved Context:**")
                for i, context in enumerate(context_texts):
                    st.markdown(f"**Context {i+1}:**")
                    st.text_area(f"Context {i+1}", context, height=150, key=f"context_{i}")
            
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")

def show_langsmith_tab():
    """Display the LangSmith monitoring tab."""
    st.header("📊 LangSmith Monitoring")
    
    try:
        langsmith_client = LangSmithClient()
        st.success("✅ LangSmith client connected!")
        
        # Project metrics
        st.subheader("📈 Project Metrics")
        
        if st.button("🔄 Refresh Metrics"):
            with st.spinner("Fetching metrics..."):
                metrics = langsmith_client.get_project_metrics()
                
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Runs", metrics.get("total_runs", 0))
                    
                    with col2:
                        st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
                    
                    with col3:
                        st.metric("Avg Latency", f"{metrics.get('average_latency_seconds', 0):.2f}s")
                    
                    with col4:
                        st.metric("Total Cost", f"${metrics.get('total_cost_usd', 0):.4f}")
                else:
                    st.info("No metrics available yet. Run some queries first!")
        
        # Recent runs
        st.subheader("🔄 Recent Runs")
        
        if st.button("📋 Get Recent Runs"):
            with st.spinner("Fetching recent runs..."):
                runs = langsmith_client.get_project_runs(limit=10)
                
                if runs:
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(runs)
                    
                    # Format timestamps
                    if 'start_time' in df.columns:
                        df['start_time'] = pd.to_datetime(df['start_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No runs found yet. Run some queries first!")
        
        # Dataset management
        st.subheader("📚 Dataset Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input("Dataset Name:", placeholder="e.g., rag-evaluation-dataset")
            dataset_description = st.text_area("Description:", placeholder="Dataset for RAG evaluation")
            
            if st.button("📁 Create Dataset"):
                if dataset_name:
                    try:
                        dataset_id = langsmith_client.create_dataset(dataset_name, dataset_description)
                        st.success(f"✅ Dataset created with ID: {dataset_id}")
                    except Exception as e:
                        st.error(f"❌ Error creating dataset: {str(e)}")
                else:
                    st.warning("Please enter a dataset name")
        
        with col2:
            st.markdown("**Export Options:**")
            if st.button("📊 Export Traces to CSV"):
                try:
                    output_path = "data/langsmith_traces.csv"
                    langsmith_client.export_traces_to_csv(output_path)
                    st.success(f"✅ Traces exported to {output_path}")
                except Exception as e:
                    st.error(f"❌ Error exporting traces: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Error connecting to LangSmith: {str(e)}")
        st.info("Please check your LangSmith API key and configuration.")

def show_ragas_tab():
    """Display the Ragas evaluation tab."""
    st.header("📈 Ragas Evaluation")
    
    try:
        evaluator = RagasEvaluator()
        st.success("✅ Ragas evaluator ready!")
        
        # Evaluation configuration
        st.subheader("⚙️ Evaluation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_questions = st.slider("Number of Test Questions:", 5, 15, 10)
            include_ground_truth = st.checkbox("Include Ground Truth", value=True)
        
        with col2:
            st.markdown("**Evaluation Metrics:**")
            for metric in evaluator.config.EVALUATION_METRICS:
                st.markdown(f"• {metric.replace('_', ' ').title()}")
        
        # Generate test questions
        if st.button("🎯 Generate Test Questions"):
            test_questions = evaluator.generate_test_questions(num_questions)
            st.session_state.test_questions = test_questions
            
            if include_ground_truth:
                ground_truths = evaluator.generate_ground_truths(test_questions)
                st.session_state.ground_truths = ground_truths
            
            st.success(f"✅ Generated {len(test_questions)} test questions!")
            
            # Display questions
            st.markdown("**📝 Test Questions:**")
            for i, question in enumerate(test_questions):
                st.markdown(f"{i+1}. {question}")
        
        # Run evaluation
        if st.button("🚀 Run Evaluation") and hasattr(st.session_state, 'test_questions'):
            with st.spinner("Running evaluation..."):
                try:
                    pipeline = RAGPipeline()
                    
                    ground_truths = st.session_state.get('ground_truths', None)
                    
                    results = evaluator.evaluate_rag_pipeline(
                        pipeline, 
                        st.session_state.test_questions,
                        ground_truths
                    )
                    
                    st.session_state.evaluation_results = results
                    st.success("✅ Evaluation completed!")
                    
                    # Display results
                    display_evaluation_results(results)
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
        
        # Show previous results
        if hasattr(st.session_state, 'evaluation_results'):
            st.subheader("📊 Previous Evaluation Results")
            display_evaluation_results(st.session_state.evaluation_results)
        
        # Load/save results
        st.subheader("💾 Save/Load Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if hasattr(st.session_state, 'evaluation_results'):
                if st.button("💾 Save Results"):
                    try:
                        output_path = "data/evaluation_results.json"
                        evaluator.save_evaluation_results(
                            st.session_state.evaluation_results, 
                            output_path
                        )
                        st.success(f"✅ Results saved to {output_path}")
                    except Exception as e:
                        st.error(f"❌ Error saving results: {str(e)}")
        
        with col2:
            uploaded_file = st.file_uploader("Load Results", type=['json'])
            if uploaded_file is not None:
                try:
                    # Save uploaded file temporarily
                    temp_path = "data/temp_upload.json"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Load results
                    results = evaluator.load_evaluation_results(temp_path)
                    st.session_state.evaluation_results = results
                    st.success("✅ Results loaded successfully!")
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"❌ Error loading results: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Error initializing Ragas evaluator: {str(e)}")

def display_evaluation_results(results):
    """Display evaluation results in a formatted way."""
    scores = results.get("evaluation_scores", {})
    
    if not scores:
        st.warning("No evaluation scores available")
        return
    
    # Display scores
    st.markdown("**🎯 Evaluation Scores:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for metric, score in scores.items():
            st.metric(
                metric.replace('_', ' ').title(),
                f"{score:.3f}",
                delta=None
            )
    
    with col2:
        # Calculate overall score
        avg_score = sum(scores.values()) / len(scores)
        st.metric("Overall Score", f"{avg_score:.3f}")
        
        # Performance indicator
        if avg_score >= 0.8:
            st.success("🎉 Excellent performance!")
        elif avg_score >= 0.6:
            st.info("👍 Good performance")
        else:
            st.warning("⚠️ Room for improvement")
    
    # Show detailed results
    if st.button("📋 Show Detailed Results"):
        st.markdown("**📊 Detailed Results:**")
        
        # Questions and answers
        for i, (question, answer) in enumerate(zip(
            results.get("test_questions", []),
            results.get("answers", [])
        )):
            with st.expander(f"Question {i+1}: {question[:50]}..."):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                
                if results.get("ground_truths"):
                    st.markdown(f"**Ground Truth:** {results['ground_truths'][i]}")

def show_documents_tab():
    """Display the document management tab."""
    st.header("📁 Document Management")
    
    # Document operations
    st.subheader("📚 Document Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Create Sample Documents**")
        if st.button("🎲 Generate Sample Docs"):
            sample_docs = create_sample_documents()
            st.session_state.documents = sample_docs
            st.success(f"✅ Created {len(sample_docs)} sample documents!")
        
        st.markdown("**Upload Documents**")
        uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = "data/temp_upload.json"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Load documents
                documents = load_documents_from_json(temp_path)
                st.session_state.documents = documents
                st.success(f"✅ Loaded {len(documents)} documents!")
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                st.error(f"❌ Error loading documents: {str(e)}")
    
    with col2:
        st.markdown("**Save Documents**")
        if hasattr(st.session_state, 'documents'):
            if st.button("💾 Save to JSON"):
                try:
                    output_path = "data/documents.json"
                    save_documents_to_json(st.session_state.documents, output_path)
                    st.success(f"✅ Documents saved to {output_path}")
                except Exception as e:
                    st.error(f"❌ Error saving documents: {str(e)}")
        
        st.markdown("**Document Statistics**")
        if hasattr(st.session_state, 'documents'):
            from src.utils import get_document_statistics
            stats = get_document_statistics(st.session_state.documents)
            
            if stats:
                st.metric("Total Documents", stats["total_documents"])
                st.metric("Total Words", stats["total_words"])
                st.metric("Avg Words/Doc", f"{stats['average_words_per_doc']:.1f}")
    
    # Display documents
    if hasattr(st.session_state, 'documents'):
        st.subheader("📖 Current Documents")
        
        # Document list
        for i, doc in enumerate(st.session_state.documents):
            with st.expander(f"📄 {doc['title']}"):
                st.markdown(f"**Title:** {doc['title']}")
                st.markdown(f"**Content:** {doc['content'][:300]}...")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.documents.pop(i)
                        st.rerun()
                
                with col2:
                    if st.button(f"✏️ Edit", key=f"edit_{i}"):
                        st.session_state.editing_doc = i
        
        # Edit document
        if hasattr(st.session_state, 'editing_doc'):
            doc_index = st.session_state.editing_doc
            doc = st.session_state.documents[doc_index]
            
            st.markdown("**✏️ Edit Document**")
            
            new_title = st.text_input("Title:", value=doc['title'], key=f"edit_title_{doc_index}")
            new_content = st.text_area("Content:", value=doc['content'], height=200, key=f"edit_content_{doc_index}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Changes"):
                    st.session_state.documents[doc_index]['title'] = new_title
                    st.session_state.documents[doc_index]['content'] = new_content
                    del st.session_state.editing_doc
                    st.success("✅ Document updated!")
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel"):
                    del st.session_state.editing_doc
                    st.rerun()
    
    else:
        st.info("No documents loaded yet. Create sample documents or upload a file to get started!")

def create_sample_docs_action():
    """Action for creating sample documents."""
    try:
        sample_docs = create_sample_documents()
        st.session_state.documents = sample_docs
        st.success(f"✅ Created {len(sample_docs)} sample documents!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error creating sample documents: {str(e)}")

def run_quick_evaluation_action():
    """Action for running a quick evaluation."""
    try:
        evaluator = RagasEvaluator()
        pipeline = RAGPipeline()
        
        # Generate a few test questions
        test_questions = evaluator.generate_test_questions(5)
        
        with st.spinner("Running quick evaluation..."):
            results = evaluator.evaluate_rag_pipeline(pipeline, test_questions)
            st.session_state.evaluation_results = results
            st.success("✅ Quick evaluation completed!")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error running quick evaluation: {str(e)}")

if __name__ == "__main__":
    main()
