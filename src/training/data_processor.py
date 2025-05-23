from datasets import Dataset
import json

class NHSDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_jsonl(self):
        """Load JSONL training data"""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        # Ensure it works with HuggingFace datasets by creating a Dataset object
        # This might require the 'datasets' library to be available.
        # If 'datasets' is not available, returning the list 'data' is also acceptable for structure.
        try:
            return Dataset.from_list(data)
        except ImportError:
            print("Warning: 'datasets' library not found. Returning raw list of data.")
            return data
    
    def format_for_training(self, example):
        """Format examples for instruction tuning"""
        instruction = example['instruction']
        input_text = example['input']
        output = example['output']
        
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        return {"text": prompt}
    
    def prepare_dataset(self):
        """Prepare final training dataset"""
        dataset = self.load_jsonl()
        # The .map() function is specific to HuggingFace datasets.
        # If using a raw list, a list comprehension would be used here.
        if hasattr(dataset, 'map'):
            dataset = dataset.map(self.format_for_training)
        else: # Assuming dataset is a list of dicts
            dataset = [self.format_for_training(ex) for ex in dataset]
        return dataset
