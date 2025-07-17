"""Data loader for routellm/gpt4_dataset."""

from typing import Any, Dict, List, Optional
import pandas as pd
from datasets import load_dataset, Dataset


class GPT4DatasetLoader:
    """Loader for routellm/gpt4_dataset from HuggingFace."""
    
    def __init__(self, dataset_name: str = "routellm/gpt4_dataset") -> None:
        """Initialize the dataset loader.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
        """
        self.dataset_name = dataset_name
        self.dataset: Optional[Dataset] = None
        
    def load_dataset(self, split: str = "train") -> Dataset:
        """Load the dataset from HuggingFace using streaming mode.
        
        Args:
            split: Dataset split to load
            
        Returns:
            Loaded dataset in streaming mode (no local storage)
        """
        # Always use streaming=True to avoid downloading/storing locally
        self.dataset = load_dataset(self.dataset_name, split=split, streaming=True)
        return self.dataset
    
    def to_dataframe(self, n_samples: int = 1000) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame.
        
        Args:
            n_samples: Number of samples to convert (streaming datasets need limit)
            
        Returns:
            Dataset as DataFrame
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # For streaming datasets, collect limited samples
        sample_data = []
        for i, item in enumerate(self.dataset):
            if i >= n_samples:
                break
            sample_data.append(item)
        
        return pd.DataFrame(sample_data)
    
    def get_sample(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get a sample of the dataset.
        
        Args:
            n: Number of samples to return
            
        Returns:
            Sample dataset as list of dictionaries
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # For streaming datasets, collect samples without storing locally
        sample_data = []
        for i, item in enumerate(self.dataset):
            if i >= n:
                break
            sample_data.append(item)
        
        return sample_data
    
    def get_conversations(self, n_samples: int = 1000) -> List[Dict[str, Any]]:
        """Extract conversations from the dataset.
        
        Args:
            n_samples: Number of conversations to extract (streaming limit)
        
        Returns:
            List of conversation dictionaries
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        conversations = []
        for i, item in enumerate(self.dataset):
            if i >= n_samples:
                break
            conversations.append({
                "conversation": item.get("conversation", []),
                "metadata": {k: v for k, v in item.items() if k != "conversation"}
            })
        
        return conversations
    
    def analyze_task_types(self, n_samples: int = 1000) -> Dict[str, int]:
        """Analyze task types in the dataset.
        
        Args:
            n_samples: Number of samples to analyze
        
        Returns:
            Dictionary mapping task types to counts
        """
        conversations = self.get_conversations(n_samples)
        task_counts: Dict[str, int] = {}
        
        for conv in conversations:
            # Extract task type from conversation or metadata
            # This is a simplified analysis - adapt based on actual dataset structure
            messages = conv["conversation"]
            if messages:
                first_message = messages[0]
                content = first_message.get("content", "").lower()
                
                # Simple heuristic task classification
                if "code" in content or "programming" in content:
                    task_type = "code_generation"
                elif "summarize" in content or "summary" in content:
                    task_type = "summarization"
                elif "question" in content or "?" in content:
                    task_type = "question_answering"
                elif "classify" in content or "categorize" in content:
                    task_type = "classification"
                else:
                    task_type = "other"
                
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        return task_counts