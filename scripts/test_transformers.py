from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("✅ Transformers working correctly")
except Exception as e:
    print(f"❌ Transformers error: {e}")
