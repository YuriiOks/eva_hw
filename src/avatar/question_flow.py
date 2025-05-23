import json
from typing import List, Dict, Optional, Any
import time
import os

class NHSQuestionFlow:
    def __init__(self, questions_file_path='data/nhs_questions.json'):
        self.questions_file_path = questions_file_path
        self.questions = self._load_nhs_questions()
        self.current_question_index = 0
        self.patient_responses: Dict[str, Dict[str, Any]] = {} # Store question text and response
        self.flow_active = False
        self.start_time: Optional[float] = None
        print("NHSQuestionFlow initialized.")
        
    def _load_nhs_questions(self) -> List[Dict[str, Any]]:
        try:
            # Check direct path
            if not os.path.exists(self.questions_file_path):
                # Construct path relative to this file's location (src/avatar/)
                # then go up two levels to project root, then to data/
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                alt_path = os.path.join(base_dir, self.questions_file_path)
                
                if os.path.exists(alt_path):
                    self.questions_file_path = alt_path
                else:
                    # Fallback for cases where CWD might be project root (e.g. local dev)
                    simple_relative_path = self.questions_file_path 
                    if os.path.exists(simple_relative_path):
                        self.questions_file_path = simple_relative_path
                    else:
                        raise FileNotFoundError(f"Questions file not found at primary '{self.questions_file_path}', alt '{alt_path}', or simple relative '{simple_relative_path}'")

            with open(self.questions_file_path, 'r') as f:
                data = json.load(f)
            
            if "questions" in data and isinstance(data["questions"], list):
                print(f"Loaded {len(data['questions'])} questions from {self.questions_file_path}")
                return data['questions']
            else:
                print(f"Warning: 'questions' key not found or not a list in {self.questions_file_path}. Using fallback.")
                return self._get_fallback_questions()
        except FileNotFoundError as e:
            print(f"Error loading questions: {e}. Using fallback.")
            return self._get_fallback_questions()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.questions_file_path}: {e}. Using fallback.")
            return self._get_fallback_questions()
        except Exception as e:
            print(f"An unexpected error occurred while loading questions: {e}. Using fallback.")
            return self._get_fallback_questions()

    def _get_fallback_questions(self) -> List[Dict[str, Any]]:
        print("Warning: Using fallback questions due to loading issues.")
        return [
            {"id": 1, "question": "What happened to bring you here today?", "type": "incident_description", "departments": ["Reception", "Triage", "Nurse", "Doctor"]},
            {"id": 2, "question": "When exactly did this injury/problem occur?", "type": "time_of_incident", "departments": ["Triage", "Nurse", "Doctor"]},
            {"id": 3, "question": "On a scale of 1-10, how bad is the pain?", "type": "pain_level", "departments": ["Triage", "Nurse", "Doctor"]},
        ]
    
    def start_flow(self, patient_scenario: str = "general") -> Dict[str, Any]:
        self.flow_active = True
        self.current_question_index = 0
        self.patient_responses = {} # Reset responses
        self.start_time = time.time()
        print(f"Question flow started for scenario: {patient_scenario}. Total questions: {len(self.questions)}")
        welcome_message = "Welcome! I'm Sarah, your virtual NHS assistant. Let's go through some questions to prepare for your visit."
        return {"status": "started", "scenario": patient_scenario, "total_questions": len(self.questions), "message": welcome_message}
    
    def get_current_question(self) -> Optional[Dict[str, Any]]:
        if not self.flow_active or self.current_question_index >= len(self.questions):
            return None
        
        q_data = self.questions[self.current_question_index]
        # Ensure all questions have a 'type' for consistent keying in responses, default if missing
        if 'type' not in q_data:
            q_data['type'] = f"question_{q_data.get('id', self.current_question_index)}"

        return {"question_number": self.current_question_index + 1, "total_questions": len(self.questions), **q_data}
    
    def process_response(self, response: str) -> Dict[str, Any]:
        if not self.flow_active:
            return {"error": "Flow not active. Please start the flow first."}
        
        current_q_data = self.get_current_question()
        if not current_q_data:
            return {"error": "No current question available. The flow might be complete or not started."}
        
        # Use the 'type' field from the question data as the key for responses
        question_type_key = current_q_data.get("type", f"question_{self.current_question_index}")
        self.patient_responses[question_type_key] = {
            "question": current_q_data["question"],
            "response": response,
            "question_id": current_q_data.get("id", "N/A")
        }
        print(f"Response to Q{current_q_data['question_number']} ('{current_q_data['question'][:30]}...') recorded: '{response[:30]}...'")
        
        self.current_question_index += 1
        
        if self.current_question_index >= len(self.questions):
            return self.complete_flow()
        else:
            return {"status": "continuing", "next_question_number": self.current_question_index + 1, "total_questions": len(self.questions)}

    def complete_flow(self) -> Dict[str, Any]:
        self.flow_active = False
        time_taken = time.time() - (self.start_time if self.start_time else time.time())
        summary = self.generate_patient_summary()
        print(f"Question flow completed. Summary generated. Time taken: {time_taken:.2f}s")
        completion_message = "Thank you for answering all questions. Your information has been compiled."
        return {"status": "completed", "summary": summary, "time_taken_seconds": time_taken, "message": completion_message}
    
    def generate_patient_summary(self) -> str:
        if not self.patient_responses:
            return "No responses recorded."
        summary_lines = [f"Patient Responses Summary:"]
        for q_type, data in self.patient_responses.items():
            summary_lines.append(f"- Q (ID {data.get('question_id', 'N/A')} | Type: {q_type}): {data['question']}")
            summary_lines.append(f"  A: {data['response']}")
        return "\n".join(summary_lines)

    def get_preparation_advice(self) -> Dict[str, Any]:
        # This could be enhanced based on patient_scenario or responses
        advice = {
            "tips": [
                "Have a list of current medications ready.",
                "Know your GP's name and practice details.",
                "Bring any relevant medical history documents.",
                "Prepare your emergency contact information."
            ],
            "estimated_questions_to_expect": len(self.questions), # Or a dynamic number
            "general_advice": "Stay calm and provide clear answers. Ask for clarification if needed."
        }
        print("Preparation advice generated.")
        return advice

class NHSSimulation: # As per README structure
    def __init__(self):
        self.question_flow = NHSQuestionFlow()
        print("NHSSimulation initialized.")

    def simulate_full_visit(self, patient_scenario: str = "general") -> List[Dict[str, Any]]:
        print(f"Simulating full visit for scenario: {patient_scenario} (Placeholder)")
        # This would involve stepping through departments and asking questions
        # For now, a placeholder:
        log = []
        self.question_flow.start_flow(patient_scenario)
        log.append({"event": "Simulation Started", "scenario": patient_scenario, "total_questions": len(self.question_flow.questions)})
        
        current_q = self.question_flow.get_current_question()
        while current_q:
            department = current_q.get("departments", ["Unknown Department"])[0] # Take first department
            log.append({"department_interaction": department, "question_asked": current_q["question"]})
            # Simulate a generic response
            self.question_flow.process_response(f"Simulated answer to Q{current_q['question_number']}")
            current_q = self.question_flow.get_current_question()
        
        completion_data = self.question_flow.complete_flow()
        log.append({"event": "Simulation Ended", "summary": completion_data["summary"]})
        return log

    def get_department_questions(self, department_name: str) -> List[Dict[str, Any]]:
        print(f"Getting questions for department: {department_name} (Placeholder)")
        if not self.question_flow.questions: # Ensure questions are loaded
            self.question_flow._load_nhs_questions()

        dept_questions = [
            q for q in self.question_flow.questions 
            if department_name in q.get("departments", [])
        ]
        if not dept_questions:
            return [{"question": f"No specific questions found for {department_name}. What happened?", "departments": [department_name]}]
        return dept_questions

if __name__ == "__main__":
    pass
