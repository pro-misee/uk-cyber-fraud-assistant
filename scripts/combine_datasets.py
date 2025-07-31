"""
Combine all fraud Q&A datasets from different sources into a master training dataset.
This script consolidates data from GetSafeOnline, FCA, UK Finance, Action Fraud, and Which.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

class DatasetCombiner:
    def __init__(self, project_root: str = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_sources_dir = self.project_root / "data_sources"
        self.output_dir = self.project_root / "model_training"
        self.output_dir.mkdir(exist_ok=True)
        
        # Source configurations
        self.sources = {
            'getsafeonline': {'expected_pairs': 23, 'priority': 1},
            'fca': {'expected_pairs': 34, 'priority': 1},
            'ukfinance': {'expected_pairs': 7, 'priority': 1},
            'actionfraud': {'expected_pairs': 39, 'priority': 1},
            'citizensadvice': {'expected_pairs': 0, 'priority': 3}  
        }
    
    def load_source_data(self, source_name: str) -> List[Dict[str, Any]]:
        """Load Q&A data from a specific source."""
        qa_file = self.data_sources_dir / source_name / "processed" / "gemini_qa_pairs" / "final_combined_qa_pairs.json"
        
        if not qa_file.exists():
            print(f"No processed data found for {source_name} at {qa_file}")
            return []
        
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Expected list format for {source_name}, got {type(data)}")
                return []
            
            # Add source metadata to each Q&A pair
            for item in data:
                item['data_source'] = source_name
                item['source_priority'] = self.sources[source_name]['priority']
            
            print(f"Loaded {len(data)} Q&A pairs from {source_name}")
            return data
            
        except Exception as e:
            print(f"Error loading {source_name}: {e}")
            return []
    
    def validate_qa_pair(self, qa_pair: Dict[str, Any]) -> bool:
        """Validate that a Q&A pair has required fields and content."""
        required_fields = ['instruction', 'input', 'output']
        
        for field in required_fields:
            if field not in qa_pair or not qa_pair[field]:
                if field != 'input':  # input can be empty
                    return False
        
        # Check minimum content length
        if len(qa_pair['instruction'].strip()) < 10:
            return False
        if len(qa_pair['output'].strip()) < 50:
            return False
        
        return True
    
    def combine_all_sources(self) -> List[Dict[str, Any]]:
        """Combine Q&A data from all available sources."""
        all_qa_pairs = []
        source_stats = {}
        
        print("Combining datasets from all sources...")
        print("=" * 50)
        
        for source_name, config in self.sources.items():
            qa_data = self.load_source_data(source_name)
            
            # Validate data quality
            valid_pairs = [qa for qa in qa_data if self.validate_qa_pair(qa)]
            invalid_count = len(qa_data) - len(valid_pairs)
            
            if invalid_count > 0:
                print(f"Filtered out {invalid_count} invalid Q&A pairs from {source_name}")
            
            all_qa_pairs.extend(valid_pairs)
            source_stats[source_name] = {
                'total_pairs': len(qa_data),
                'valid_pairs': len(valid_pairs),
                'expected_pairs': config['expected_pairs']
            }
        
        print("=" * 50)
        print(f"📊 Data Collection Summary:")
        for source, stats in source_stats.items():
            status = "✅" if stats['valid_pairs'] > 0 else "❌"
            print(f"  {status} {source}: {stats['valid_pairs']}/{stats['expected_pairs']} pairs")
        
        print(f"\n Total combined dataset: {len(all_qa_pairs)} Q&A pairs")
        return all_qa_pairs
    
    def create_train_validation_split(self, qa_pairs: List[Dict[str, Any]], 
                                    train_ratio: float = 0.8) -> tuple:
        """Split dataset into training and validation sets."""
        import random
        
        # Shuffle while maintaining reproducibility
        random.seed(42)
        shuffled_pairs = qa_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        split_idx = int(len(shuffled_pairs) * train_ratio)
        train_data = shuffled_pairs[:split_idx]
        val_data = shuffled_pairs[split_idx:]
        
        print(f"Dataset split: {len(train_data)} training, {len(val_data)} validation")
        return train_data, val_data
    
    def format_for_llama2(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format Q&A pairs for LLaMA 2 fine-tuning with UK fraud assistant system prompt."""

        system_prompt = """You are a specialized AI assistant helping UK cyber fraud victims. You provide empathetic, accurate, and actionable guidance based on official UK cyber fraud prevention resources. Always:

- Respond with empathy and understanding
- Provide specific UK contact numbers and procedures
- Reassure victims that cyber fraud is not their fault
- Give clear, step-by-step guidance
- Direct to appropriate authorities (Action Fraud, local police, banks)
- Maintain a supportive, non-judgmental tone"""
        
        formatted_data = []
        
        for qa in qa_pairs:
            # Create conversation format for LLaMA 2
            conversation = {
                "instruction": qa['instruction'],
                "input": qa.get('input', ''),
                "output": qa['output'],
                "system": system_prompt,
                "source_document": qa.get('source_document', ''),
                "source_url": qa.get('source_url', ''),
                "data_source": qa.get('data_source', ''),
                "generated_by": qa.get('generated_by', 'gemini')
            }
            formatted_data.append(conversation)
        
        return formatted_data
    
    def save_datasets(self, train_data: List[Dict[str, Any]], 
                     val_data: List[Dict[str, Any]], 
                     all_data: List[Dict[str, Any]]):
        """Save the combined and split datasets."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save master dataset
        master_file = self.output_dir / "master_fraud_qa_dataset.json"
        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        # Save training dataset
        train_file = self.output_dir / "train_fraud_qa_dataset.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # Save validation dataset
        val_file = self.output_dir / "val_fraud_qa_dataset.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        # Save CSV for analysis
        df = pd.DataFrame(all_data)
        csv_file = self.output_dir / "master_fraud_qa_dataset.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Create dataset info file
        dataset_info = {
            "created_at": datetime.now().isoformat(),
            "total_qa_pairs": len(all_data),
            "training_pairs": len(train_data),
            "validation_pairs": len(val_data),
            "sources_included": list(set([qa.get('data_source', 'unknown') for qa in all_data])),
            "source_distribution": {
                source: len([qa for qa in all_data if qa.get('data_source') == source])
                for source in set([qa.get('data_source', 'unknown') for qa in all_data])
            },
            "files": {
                "master_dataset": str(master_file.name),
                "training_dataset": str(train_file.name),
                "validation_dataset": str(val_file.name),
                "analysis_csv": str(csv_file.name)
            }
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f" Datasets saved to {self.output_dir}/")
        print(f"   Master: {master_file.name} ({len(all_data)} pairs)")
        print(f"   Training: {train_file.name} ({len(train_data)} pairs)")
        print(f"   Validation: {val_file.name} ({len(val_data)} pairs)")
        print(f"   Analysis: {csv_file.name}")
        print(f"    Info: {info_file.name}")
        
        return dataset_info
    
    def generate_sample_conversations(self, data: List[Dict[str, Any]], num_samples: int = 3):
        """Generate sample conversations to preview the dataset quality."""
        import random
        
        print(f"\n Sample Conversations (previewing {num_samples} examples):")
        print("=" * 70)
        
        samples = random.sample(data, min(num_samples, len(data)))
        
        for i, sample in enumerate(samples, 1):
            print(f"\n Sample {i} (Source: {sample.get('data_source', 'unknown')})")
            print(f" User: {sample['instruction']}")
            print(f" Assistant: {sample['output'][:200]}...")
            if len(sample['output']) > 200:
                print("   [...truncated]")
            print("-" * 50)

def main():
    """Main execution function."""
    print(" UK Fraud Chatbot Dataset Combiner")
    print("=" * 50)
    
    # Initialize combiner
    combiner = DatasetCombiner()
    
    # Combine all sources
    all_qa_pairs = combiner.combine_all_sources()
    
    if not all_qa_pairs:
        print(" No valid Q&A pairs found. Please check your data sources.")
        return
    
    # Format for LLaMA 2
    formatted_pairs = combiner.format_for_llama2(all_qa_pairs)
    
    # Create train/validation split
    train_data, val_data = combiner.create_train_validation_split(formatted_pairs)
    
    # Save datasets
    dataset_info = combiner.save_datasets(train_data, val_data, formatted_pairs)
    
    # Generate preview
    combiner.generate_sample_conversations(formatted_pairs)
    
    print(f"\n Dataset combination completed successfully!")
    print(f"Ready for LLaMA 2 fine-tuning with {len(formatted_pairs)} total Q&A pairs")
    
    return dataset_info

if __name__ == "__main__":
    main()