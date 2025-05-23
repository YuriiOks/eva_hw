import json

def create_initial_dataset():
    nhs_qa_base = [
        {
            "instruction": "You are an NHS assistant helping patients understand hospital processes.",
            "input": "What questions will NHS staff ask me in A&E?",
            "output": "NHS A&E staff typically ask: 1) What brought you here today? 2) When did this happen? 3) Rate your pain 1-10? 4) Any allergies? 5) Current medications? 6) Your GP details? 7) Emergency contact information? Being prepared with these answers saves time."
        }
    ]
    
    training_examples = [
        {
            "instruction": "Help prepare a patient for NHS A&E visit",
            "input": "My elderly mother broke her arm, what should we expect?",
            "output": "For a broken arm in A&E, prepare these answers: How it happened, exact time of injury, pain level 1-10, any allergies, current medications, GP practice details, and emergency contact. You'll likely need X-rays, possible cast/splint, and pain relief. Average wait: 3-4 hours."
        },
        {
            "instruction": "Explain NHS process for non-English speakers",
            "input": "My parent doesn't speak English well, how can we prepare?",
            "output": "Bring a translator or family member. Write down key information beforehand: allergies, medications, GP details. NHS should provide interpreters, but having written answers to common questions speeds up the process. Consider translation apps as backup."
        }
    ]

    # Placeholder for synthetic data generation (README Step 15)
    # def generate_synthetic_data():
    #     templates = [
    #         "As an NHS patient with {condition}, what should I expect?",
    #         "Help me prepare for {department} visit",
    #         "What questions will {staff_type} ask me?",
    #         "Explain {medical_term} in simple terms",
    #         "How long does {procedure} take in NHS?"
    #     ]
    #     # In a real scenario, this function would generate more data
    #     pass
    
    full_dataset = nhs_qa_base + training_examples

    with open('data/nhs_qa_dataset.jsonl', 'w') as f:
        for item in full_dataset:
            f.write(json.dumps(item) + '\n')
    print("data/nhs_qa_dataset.jsonl created/updated successfully.")

if __name__ == "__main__":
    create_initial_dataset()
