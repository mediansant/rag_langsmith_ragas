"""
Ragas evaluator for comprehensive RAG system evaluation.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_relevancy,
    faithfulness,
    answer_correctness
)
from datasets import Dataset
import pandas as pd

from config.config import Config

logger = logging.getLogger(__name__)

class RagasEvaluator:
    """
    Ragas evaluator for RAG system quality assessment.
    """
    
    def __init__(self):
        """Initialize the Ragas evaluator."""
        self.config = Config()
        self.metrics = self._get_metrics()
        
        logger.info("✅ Ragas evaluator initialized")
    
    def _get_metrics(self) -> List:
        """Get the evaluation metrics based on configuration."""
        metric_map = {
            "answer_relevancy": answer_relevancy,
            "context_relevancy": context_relevancy,
            "faithfulness": faithfulness,
            "answer_correctness": answer_correctness
        }
        
        metrics = []
        for metric_name in self.config.EVALUATION_METRICS:
            if metric_name in metric_map:
                metrics.append(metric_map[metric_name])
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        
        logger.info(f"Using metrics: {[m.__name__ for m in metrics]}")
        return metrics
    
    def create_evaluation_dataset(self, questions: List[str], 
                                contexts: List[List[str]], 
                                answers: List[str],
                                ground_truths: Optional[List[str]] = None) -> Dataset:
        """
        Create a dataset for evaluation.
        
        Args:
            questions: List of questions
            contexts: List of context lists for each question
            answers: List of generated answers
            ground_truths: List of ground truth answers (optional)
            
        Returns:
            HuggingFace Dataset for evaluation
        """
        try:
            # Prepare dataset
            dataset_dict = {
                "question": questions,
                "contexts": contexts,
                "answer": answers
            }
            
            if ground_truths:
                dataset_dict["ground_truth"] = ground_truths
            
            # Create dataset
            dataset = Dataset.from_dict(dataset_dict)
            
            logger.info(f"✅ Created evaluation dataset with {len(questions)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Error creating evaluation dataset: {e}")
            raise
    
    def evaluate_dataset(self, dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate a dataset using Ragas metrics.
        
        Args:
            dataset: Dataset to evaluate
            
        Returns:
            Dictionary of metric scores
        """
        try:
            logger.info(f"Evaluating dataset with {len(dataset)} samples...")
            
            # Run evaluation
            results = evaluate(
                dataset,
                metrics=self.metrics,
                verbose=True
            )
            
            # Extract scores
            scores = {}
            for metric_name, score in results.items():
                if hasattr(score, 'score'):
                    scores[metric_name] = float(score.score)
                else:
                    scores[metric_name] = float(score)
            
            logger.info("✅ Evaluation completed successfully")
            return scores
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}")
            raise
    
    def evaluate_rag_pipeline(self, pipeline, test_questions: List[str],
                            ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a RAG pipeline with test questions.
        
        Args:
            pipeline: RAG pipeline instance
            test_questions: List of test questions
            ground_truths: List of ground truth answers (optional)
            
        Returns:
            Evaluation results and pipeline performance
        """
        try:
            logger.info(f"Evaluating RAG pipeline with {len(test_questions)} questions...")
            
            # Generate responses
            contexts = []
            answers = []
            
            for i, question in enumerate(test_questions):
                logger.info(f"Processing question {i+1}/{len(test_questions)}: {question[:50]}...")
                
                # Get context
                context_docs = pipeline.get_relevant_context(question)
                context_texts = [doc.page_content for doc in context_docs]
                contexts.append(context_texts)
                
                # Generate answer
                answer = pipeline.query(question)
                answers.append(answer)
                
                logger.info(f"Generated answer: {answer[:100]}...")
            
            # Create evaluation dataset
            dataset = self.create_evaluation_dataset(
                questions=test_questions,
                contexts=contexts,
                answers=answers,
                ground_truths=ground_truths
            )
            
            # Run evaluation
            scores = self.evaluate_dataset(dataset)
            
            # Compile results
            results = {
                "evaluation_scores": scores,
                "test_questions": test_questions,
                "contexts": contexts,
                "answers": answers,
                "ground_truths": ground_truths,
                "dataset_size": len(test_questions),
                "evaluation_metrics": [m.__name__ for m in self.metrics]
            }
            
            logger.info("✅ RAG pipeline evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error evaluating RAG pipeline: {e}")
            raise
    
    def generate_test_questions(self, num_questions: int = 10) -> List[str]:
        """
        Generate test questions for evaluation.
        
        Args:
            num_questions: Number of questions to generate
            
        Returns:
            List of test questions
        """
        # Pre-defined test questions for educational purposes
        test_questions = [
            "What is machine learning?",
            "How does deep learning work?",
            "What is natural language processing?",
            "Explain retrieval-augmented generation",
            "What are vector databases?",
            "How do embeddings work?",
            "What is the difference between AI and ML?",
            "How do neural networks learn?",
            "What is supervised learning?",
            "Explain unsupervised learning",
            "What is reinforcement learning?",
            "How do transformers work?",
            "What is the attention mechanism?",
            "Explain the concept of overfitting",
            "What is cross-validation?"
        ]
        
        # Return requested number of questions
        return test_questions[:min(num_questions, len(test_questions))]
    
    def generate_ground_truths(self, questions: List[str]) -> List[str]:
        """
        Generate ground truth answers for test questions.
        
        Args:
            questions: List of questions
            
        Returns:
            List of ground truth answers
        """
        # Pre-defined ground truth answers (simplified for educational purposes)
        ground_truths = [
            "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "Deep learning uses neural networks with multiple layers to automatically learn complex patterns from data.",
            "NLP is a field of AI focused on computer understanding and generation of human language.",
            "RAG combines information retrieval with text generation for more accurate responses.",
            "Vector databases store high-dimensional vector representations for semantic similarity search.",
            "Embeddings are numerical representations that capture meaning, with similar items having similar vectors.",
            "AI is broader, while ML is a specific approach to achieving AI through learning from data.",
            "Neural networks learn by adjusting weights through backpropagation based on training data.",
            "Supervised learning uses labeled training data to learn input-output mappings.",
            "Unsupervised learning finds patterns in data without predefined labels.",
            "Reinforcement learning learns through trial and error with rewards and penalties.",
            "Transformers use attention mechanisms to process sequential data in parallel.",
            "Attention mechanisms allow models to focus on relevant parts of input data.",
            "Overfitting occurs when a model learns training data too well but generalizes poorly.",
            "Cross-validation splits data into folds to assess model performance more reliably."
        ]
        
        # Map questions to answers
        question_answer_map = dict(zip(self.generate_test_questions(), ground_truths))
        
        # Return answers for the provided questions
        return [question_answer_map.get(q, "No ground truth available") for q in questions]
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results
            output_path: Path to save the results
        """
        try:
            # Ensure directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving evaluation results: {e}")
            raise
    
    def load_evaluation_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from a JSON file.
        
        Args:
            input_path: Path to the results file
            
        Returns:
            Loaded evaluation results
        """
        try:
            with open(input_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"✅ Evaluation results loaded from {input_path}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error loading evaluation results: {e}")
            raise
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Create a human-readable evaluation report.
        
        Args:
            results: Evaluation results
            
        Returns:
            Formatted report string
        """
        try:
            scores = results.get("evaluation_scores", {})
            
            report = "📊 RAG Pipeline Evaluation Report\n"
            report += "=" * 50 + "\n\n"
            
            # Summary
            report += f"📈 Evaluation Summary:\n"
            report += f"  • Dataset Size: {results.get('dataset_size', 'N/A')}\n"
            report += f"  • Metrics Used: {', '.join(results.get('evaluation_metrics', []))}\n\n"
            
            # Scores
            report += "🎯 Metric Scores:\n"
            for metric, score in scores.items():
                report += f"  • {metric.replace('_', ' ').title()}: {score:.3f}\n"
            
            # Overall score
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                report += f"\n🏆 Overall Score: {avg_score:.3f}\n"
            
            # Recommendations
            report += "\n💡 Recommendations:\n"
            if scores:
                low_metrics = [m for m, s in scores.items() if s < 0.7]
                if low_metrics:
                    report += f"  • Focus on improving: {', '.join(low_metrics)}\n"
                else:
                    report += "  • All metrics are performing well!\n"
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error creating evaluation report: {e}")
            return f"Error creating report: {str(e)}"
    
    def compare_evaluations(self, results1: Dict[str, Any], 
                          results2: Dict[str, Any], 
                          names: Tuple[str, str] = ("Baseline", "Improved")) -> str:
        """
        Compare two evaluation results.
        
        Args:
            results1: First evaluation results
            results2: Second evaluation results
            names: Names for the two evaluations
            
        Returns:
            Comparison report
        """
        try:
            scores1 = results1.get("evaluation_scores", {})
            scores2 = results2.get("evaluation_scores", {})
            
            report = f"📊 Evaluation Comparison: {names[0]} vs {names[1]}\n"
            report += "=" * 60 + "\n\n"
            
            # Compare scores
            report += "🎯 Metric Comparison:\n"
            all_metrics = set(scores1.keys()) | set(scores2.keys())
            
            for metric in sorted(all_metrics):
                score1 = scores1.get(metric, "N/A")
                score2 = scores2.get(metric, "N/A")
                
                if score1 != "N/A" and score2 != "N/A":
                    diff = score2 - score1
                    change = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                    report += f"  • {metric.replace('_', ' ').title()}:\n"
                    report += f"    {names[0]}: {score1:.3f} | {names[1]}: {score2:.3f} | Change: {diff:+.3f} {change}\n"
                else:
                    report += f"  • {metric.replace('_', ' ').title()}: {names[0]}: {score1} | {names[1]}: {score2}\n"
            
            # Overall comparison
            if scores1 and scores2:
                avg1 = sum(scores1.values()) / len(scores1)
                avg2 = sum(scores2.values()) / len(scores2)
                overall_diff = avg2 - avg1
                overall_change = "📈" if overall_diff > 0 else "📉" if overall_diff < 0 else "➡️"
                
                report += f"\n🏆 Overall Comparison:\n"
                report += f"  • {names[0]} Average: {avg1:.3f}\n"
                report += f"  • {names[1]} Average: {avg2:.3f}\n"
                report += f"  • Overall Change: {overall_diff:+.3f} {overall_change}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error creating comparison report: {e}")
            return f"Error creating comparison: {str(e)}"
