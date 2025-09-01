#!/usr/bin/env python3
"""
Simple demo script for the RAG pipeline with LangSmith and Ragas.
Run this script to see the pipeline in action without the Streamlit interface.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.rag_pipeline import RAGPipeline
from src.langsmith_client import LangSmithClient
from src.ragas_evaluator import RagasEvaluator
from config.config import Config

def main():
    """Main demo function."""
    print("🚀 RAG Pipeline with LangSmith & Ragas - Demo")
    print("=" * 60)
    
    # Check configuration
    print("\n🔧 Checking configuration...")
    config = Config()
    if not config.validate():
        print("❌ Configuration validation failed!")
        print("Please set your API keys in environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - LANGSMITH_API_KEY")
        return
    
    print("✅ Configuration is valid!")
    config.print_config()
    
    try:
        # Initialize components
        print("\n🏗️ Initializing components...")
        
        # Initialize RAG pipeline
        print("  📚 Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        print("  ✅ RAG pipeline ready!")
        
        # Initialize LangSmith client
        print("  📊 Initializing LangSmith client...")
        langsmith_client = LangSmithClient()
        print("  ✅ LangSmith client ready!")
        
        # Initialize Ragas evaluator
        print("  📈 Initializing Ragas evaluator...")
        evaluator = RagasEvaluator()
        print("  ✅ Ragas evaluator ready!")
        
        # Demo queries
        print("\n🔍 Running demo queries...")
        demo_questions = [
            "What is machine learning?",
            "How does deep learning work?",
            "What is natural language processing?",
            "Explain retrieval-augmented generation",
            "What are vector databases?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n  📝 Question {i}: {question}")
            
            # Get context
            context_docs = pipeline.get_relevant_context(question)
            print(f"    📚 Retrieved {len(context_docs)} context documents")
            
            # Generate answer
            answer = pipeline.query(question)
            print(f"    🤖 Answer: {answer[:100]}...")
            
            # Trace with LangSmith
            context_texts = [doc.page_content for doc in context_docs]
            trace_result = langsmith_client.trace_rag_query(
                question=question,
                context=context_texts,
                answer=answer
            )
            print(f"    📊 Traced: {trace_result['status']}")
        
        # Run evaluation
        print("\n📈 Running evaluation...")
        test_questions = evaluator.generate_test_questions(5)
        ground_truths = evaluator.generate_ground_truths(test_questions)
        
        print(f"  🎯 Evaluating with {len(test_questions)} test questions...")
        results = evaluator.evaluate_rag_pipeline(
            pipeline, 
            test_questions, 
            ground_truths
        )
        
        # Display evaluation results
        print("\n📊 Evaluation Results:")
        scores = results.get("evaluation_scores", {})
        for metric, score in scores.items():
            print(f"  • {metric.replace('_', ' ').title()}: {score:.3f}")
        
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            print(f"\n🏆 Overall Score: {avg_score:.3f}")
        
        # Get LangSmith metrics
        print("\n📊 LangSmith Project Metrics:")
        metrics = langsmith_client.get_project_metrics()
        if metrics:
            print(f"  • Total Runs: {metrics.get('total_runs', 0)}")
            print(f"  • Success Rate: {metrics.get('success_rate', 0):.1f}%")
            print(f"  • Average Latency: {metrics.get('average_latency_seconds', 0):.2f}s")
            print(f"  • Total Cost: ${metrics.get('total_cost_usd', 0):.4f}")
        else:
            print("  ℹ️ No metrics available yet")
        
        # Pipeline information
        print("\n📚 Pipeline Information:")
        info = pipeline.get_vector_store_info()
        if info:
            print(f"  • Total Documents: {info.get('total_documents', 'N/A')}")
            print(f"  • Chunk Size: {info.get('chunk_size', 'N/A')}")
            print(f"  • Embedding Model: {info.get('embedding_model', 'N/A')}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Next steps:")
        print("  • Run 'streamlit run src/streamlit_app.py' for the web interface")
        print("  • Check your LangSmith dashboard for detailed traces")
        print("  • Experiment with different questions and documents")
        print("  • Run evaluations with different metrics")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("  • Check your API keys are set correctly")
        print("  • Ensure you have sufficient API credits")
        print("  • Check your internet connection")
        print("  • Review the error message above")

if __name__ == "__main__":
    main()
