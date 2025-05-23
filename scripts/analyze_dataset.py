import json

def analyze_dataset():
    """Analyze dataset quality and statistics"""
    file_path = 'data/nhs_qa_dataset.jsonl'
    try:
        with open(file_path, 'r') as f:
            examples = [json.loads(line) for line in f]
        
        if not examples:
            print("Dataset is empty.")
            return

        print(f"Total examples: {len(examples)}")
        
        total_input_len = sum(len(str(ex.get('input', ''))) for ex in examples)
        total_output_len = sum(len(str(ex.get('output', ''))) for ex in examples)

        print(f"Avg input length: {total_input_len / len(examples):.1f} characters")
        print(f"Avg output length: {total_output_len / len(examples):.1f} characters")
        
        # Check for duplicates based on 'input'
        unique_inputs = set(ex.get('input', '') for ex in examples)
        print(f"Unique inputs: {len(unique_inputs)}")
        if len(unique_inputs) < len(examples):
            print(f"Warning: Found {len(examples) - len(unique_inputs)} duplicate inputs.")
        
        return examples

    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at {file_path}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during analysis: {e}")


if __name__ == "__main__":
    analyze_dataset()
