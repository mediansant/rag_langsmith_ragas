"""
LangSmith integration for monitoring and tracing the RAG pipeline.
"""
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from langsmith import Client
from langsmith.run_helpers import traceable
from langchain.callbacks import LangChainTracer
from langchain.callbacks.manager import CallbackManager

from config.config import Config

logger = logging.getLogger(__name__)

class LangSmithClient:
    """
    LangSmith client for monitoring and tracing RAG pipeline operations.
    """
    
    def __init__(self):
        """Initialize the LangSmith client."""
        self.config = Config()
        self.client = None
        self.tracer = None
        
        self._setup_langsmith()
    
    def _setup_langsmith(self):
        """Set up LangSmith client and tracer."""
        try:
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.config.LANGSMITH_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = self.config.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.config.LANGSMITH_PROJECT
            
            # Initialize LangSmith client
            self.client = Client(
                api_url=self.config.LANGSMITH_ENDPOINT,
                api_key=self.config.LANGSMITH_API_KEY
            )
            
            # Initialize tracer
            self.tracer = LangChainTracer(
                project_name=self.config.LANGSMITH_PROJECT
            )
            
            logger.info("✅ LangSmith client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing LangSmith: {e}")
            raise
    
    def get_callback_manager(self) -> CallbackManager:
        """
        Get a callback manager with LangSmith tracing.
        
        Returns:
            CallbackManager with LangSmith tracer
        """
        if self.tracer:
            return CallbackManager([self.tracer])
        return CallbackManager([])
    
    @traceable(name="rag_query", run_type="chain")
    def trace_rag_query(self, question: str, context: List[str], answer: str, 
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Trace a RAG query with LangSmith.
        
        Args:
            question: The user's question
            context: Retrieved context documents
            answer: Generated answer
            metadata: Additional metadata
        """
        try:
            # Create metadata for the trace
            trace_metadata = {
                "question": question,
                "context_count": len(context),
                "context_preview": [ctx[:100] + "..." if len(ctx) > 100 else ctx for ctx in context],
                "answer_length": len(answer),
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Log the trace
            logger.info(f"📊 Tracing RAG query: {question[:50]}...")
            
            return {
                "status": "traced",
                "metadata": trace_metadata,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error tracing RAG query: {e}")
            return {"status": "error", "error": str(e)}
    
    def create_dataset(self, name: str, description: str = "") -> str:
        """
        Create a new dataset in LangSmith.
        
        Args:
            name: Name of the dataset
            description: Description of the dataset
            
        Returns:
            Dataset ID
        """
        try:
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=description
            )
            
            logger.info(f"✅ Created dataset: {name} (ID: {dataset.id})")
            return dataset.id
            
        except Exception as e:
            logger.error(f"❌ Error creating dataset: {e}")
            raise
    
    def add_example_to_dataset(self, dataset_id: str, question: str, 
                              context: List[str], answer: str, 
                              ground_truth: Optional[str] = None):
        """
        Add an example to a LangSmith dataset.
        
        Args:
            dataset_id: ID of the dataset
            question: The question
            context: Retrieved context
            answer: Generated answer
            ground_truth: Ground truth answer (optional)
        """
        try:
            # Prepare inputs and outputs
            inputs = {
                "question": question,
                "context": context
            }
            
            outputs = {
                "answer": answer
            }
            
            if ground_truth:
                outputs["ground_truth"] = ground_truth
            
            # Add example to dataset
            self.client.create_example(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset_id
            )
            
            logger.info(f"✅ Added example to dataset {dataset_id}")
            
        except Exception as e:
            logger.error(f"❌ Error adding example to dataset: {e}")
            raise
    
    def get_project_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent runs from the LangSmith project.
        
        Args:
            limit: Maximum number of runs to retrieve
            
        Returns:
            List of run information
        """
        try:
            runs = self.client.list_runs(
                project_name=self.config.LANGSMITH_PROJECT,
                limit=limit
            )
            
            run_info = []
            for run in runs:
                run_info.append({
                    "id": run.id,
                    "name": run.name,
                    "status": run.status,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "latency": run.latency,
                    "tokens_used": getattr(run, 'tokens_used', None),
                    "cost": getattr(run, 'cost', None)
                })
            
            logger.info(f"✅ Retrieved {len(run_info)} runs from project")
            return run_info
            
        except Exception as e:
            logger.error(f"❌ Error retrieving project runs: {e}")
            return []
    
    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Detailed run information
        """
        try:
            run = self.client.read_run(run_id)
            
            run_details = {
                "id": run.id,
                "name": run.name,
                "status": run.status,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "latency": run.latency,
                "tokens_used": getattr(run, 'tokens_used', None),
                "cost": getattr(run, 'cost', None),
                "inputs": run.inputs,
                "outputs": run.outputs,
                "error": run.error,
                "tags": run.tags,
                "metadata": run.metadata
            }
            
            logger.info(f"✅ Retrieved details for run {run_id}")
            return run_details
            
        except Exception as e:
            logger.error(f"❌ Error retrieving run details: {e}")
            return None
    
    def get_project_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics for the project.
        
        Returns:
            Project metrics
        """
        try:
            # Get recent runs
            runs = self.get_project_runs(limit=1000)
            
            if not runs:
                return {}
            
            # Calculate metrics
            total_runs = len(runs)
            successful_runs = len([r for r in runs if r["status"] == "completed"])
            failed_runs = len([r for r in runs if r["status"] == "failed"])
            
            # Calculate average latency
            latencies = [r["latency"] for r in runs if r["latency"] is not None]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            # Calculate total cost
            costs = [r["cost"] for r in runs if r["cost"] is not None]
            total_cost = sum(costs) if costs else 0
            
            metrics = {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                "average_latency_seconds": avg_latency,
                "total_cost_usd": total_cost,
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info("✅ Retrieved project metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating project metrics: {e}")
            return {}
    
    def export_traces_to_csv(self, output_path: str, limit: int = 1000):
        """
        Export project traces to a CSV file.
        
        Args:
            output_path: Path to save the CSV file
            limit: Maximum number of traces to export
        """
        try:
            import pandas as pd
            
            # Get runs
            runs = self.get_project_runs(limit=limit)
            
            if not runs:
                logger.warning("No runs found to export")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(runs)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Exported {len(runs)} traces to {output_path}")
            
        except ImportError:
            logger.error("❌ pandas is required for CSV export")
        except Exception as e:
            logger.error(f"❌ Error exporting traces: {e}")
