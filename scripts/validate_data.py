import json

def validate_dataset():
    """Ensure dataset quality and format"""
    file_path = 'data/nhs_qa_dataset.jsonl'
    errors_found = 0
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    assert 'instruction' in data, f"Missing 'instruction' in line {line_num}"
                    assert 'input' in data, f"Missing 'input' in line {line_num}"
                    assert 'output' in data, f"Missing 'output' in line {line_num}"
                    assert len(data['output']) > 10, f"Output too short in line {line_num}"
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error in line {line_num}: {e}")
                    errors_found +=1
                except AssertionError as e:
                    print(f"Validation error: {e}")
                    errors_found +=1
        if errors_found == 0:
            print("✅ Dataset validation complete. No errors found.")
        else:
            print(f"❌ Dataset validation completed with {errors_found} errors.")
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at {file_path}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during validation: {e}")


if __name__ == "__main__":
    validate_dataset()
