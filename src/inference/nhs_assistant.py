import re
import time
import json # For loading offline guidance
import os # For path joining
import logging # For getting audit logger instance directly
try:
    from src.inference.model_loader import ModelLoader, ModelLoadError # Assuming ModelLoadError is defined there or in error_handling
    from src.training.config import ModelConfig
    from src.utils.error_handling import setup_logging, ErrorHandler, NHSNavigatorError # NHSNavigatorError for generic assistant issues
except ImportError as e:
    print(f"Error importing for NHSAssistant (Phase 9): {e}")
    # Dummy classes
    class ModelLoader: 
        def __init__(self,c): self.is_properly_loaded = False; self.config=c; self.logger = logging.getLogger("DummyMLLogger")
        def generate_response(self,p,ml=0): return "Dummy LLM response"
    class ModelConfig: model_name="dummy"
    class ModelLoadError(Exception): pass
    def setup_logging(): import logging; return logging.getLogger("DummyAssistantLogger")
    class ErrorHandler: 
        def __init__(self): self.fallback_guidance={"ae_questions":"Dummy A&E fallback", "technical_difficulty": "Dummy tech diff"}
    class NHSNavigatorError(Exception):pass

class NHSAssistant:
    # Circuit Breaker States
    CIRCUIT_CLOSED = "CLOSED"
    CIRCUIT_OPEN = "OPEN"
    CIRCUIT_HALF_OPEN = "HALF_OPEN"

    # Config
    FAILURE_THRESHOLD = 3 # Max failures before opening circuit
    COOLDOWN_PERIOD = 60  # Seconds to wait before moving to HALF_OPEN
    MEDICAL_DISCLAIMER = "Disclaimer: I am an AI assistant and not a medical professional. This information is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns."
    EMERGENCY_MESSAGE = "If you believe this is a medical emergency, please call 999 or go to your nearest A&E department immediately."
    CRITICAL_KEYWORDS = ["chest pain", "can't breathe", "cannot breathe", "stroke symptoms", "severe bleeding", "suicidal", "suicide", "overdose"]


    def __init__(self):
        self.logger = setup_logging() # Main logger
        self.audit_logger = logging.getLogger('NHS_Navigator_Audit') # Audit logger
        self.error_handler = ErrorHandler() # For user-friendly messages

        self.config = ModelConfig()
        # Ensure ModelLoader is initialized with its own config.
        # ModelLoader's __init__ should also setup its own logger via ErrorHandler or directly.
        self.model_loader = ModelLoader(self.config) 
        self.is_initialized = False # Tracks overall assistant initialization

        # Circuit Breaker attributes
        self.circuit_state = self.CIRCUIT_CLOSED
        self.failure_count = 0
        self.last_failure_time = 0

        # Offline guidance cache
        self.offline_guidance = self._load_offline_guidance()
        
        self.logger.info("NHSAssistant initialized (Phase 9).")

    def _load_offline_guidance(self, file_path="data/nhs_questions.json"):
        try:
            # Robust path construction
            current_script_dir = os.path.dirname(os.path.abspath(__file__)) # src/inference
            project_root = os.path.dirname(os.path.dirname(current_script_dir)) # Up to project root
            actual_file_path = os.path.join(project_root, file_path)


            if not os.path.exists(actual_file_path):
                self.logger.warning(f"Offline guidance file not found at {actual_file_path} (derived from {file_path}). Offline mode will be limited.")
                return {}

            with open(actual_file_path, 'r') as f:
                data = json.load(f)
            
            cache = {}
            if "questions" in data and isinstance(data["questions"], list) and data["questions"]:
                 first_q_obj = data["questions"][0]
                 first_q_text = first_q_obj.get("question", "General A&E information is available from staff.")
                 # Using a generic key that might be hit by some inputs
                 cache["ae"] = first_q_text 
                 cache["help"] = first_q_text
                 cache["information"] = first_q_text
                 self.logger.info(f"Offline guidance loaded from {actual_file_path}. Cache has {len(cache)} items. Example: '{first_q_text}'")
            else:
                self.logger.warning(f"No questions found or invalid format in {actual_file_path} for offline cache.")
            return cache
        except Exception as e:
            self.logger.error(f"Failed to load offline guidance from '{file_path}': {e}", exc_info=True)
            return {}

    def _sanitize_input(self, text: str) -> str:
        # self.audit_logger.info(f"SANITIZATION_INPUT - Original length: {len(text)}") # PII risk
        original_len = len(text)
        text = re.sub(r'<[^>]+>', '', text) # Remove HTML-like tags
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        max_len = 1000 
        if len(text) > max_len:
            text = text[:max_len]
            self.logger.warning(f"Input truncated from {original_len} to {max_len} characters.")
        # self.audit_logger.info(f"SANITIZATION_OUTPUT - Sanitized length: {len(text)}") # PII risk
        self.audit_logger.info(f"SANITIZATION_EVENT - InputLengthBefore: {original_len}, InputLengthAfter: {len(text)}, Truncated: {original_len > max_len}")
        return text

    def _check_emergency_keywords(self, text: str) -> bool:
        text_lower = text.lower()
        for keyword in self.CRITICAL_KEYWORDS:
            if keyword in text_lower:
                self.audit_logger.warning(f"EMERGENCY_KEYWORD_DETECTED - Keyword: '{keyword}'") # Avoid logging user input directly
                return True
        return False

    def initialize(self):
        if self.is_initialized: 
            self.logger.info("NHSAssistant already initialized.")
            return
        
        self.logger.info("Initializing NHSAssistant dependencies...")
        self.audit_logger.info("SYSTEM_EVENT - Type: AssistantInitializationAttempt")
        try:
            # ModelLoader's load_base_model now has retry and raises ModelLoadError
            self.model_loader.load_base_model() 
            # self.model_loader.load_lora_adapter("./models/lora_adapters") # Optional adapter load
            
            self.is_initialized = self.model_loader.is_properly_loaded
            if self.is_initialized:
                self.logger.info("NHSAssistant initialized successfully.")
                self.audit_logger.info("SYSTEM_EVENT - Type: AssistantInitialized, Status: Success")
            else: # Should ideally be caught by ModelLoadError from load_base_model
                self.logger.error("NHSAssistant initialization failed: ModelLoader did not confirm proper loading.")
                self.circuit_state = self.CIRCUIT_OPEN # Open circuit on init fail
                self.last_failure_time = time.time()
                self.audit_logger.error("SYSTEM_EVENT - Type: AssistantInitialized, Status: Failed, Reason: ModelNotProperlyLoaded")
        except ModelLoadError as mle: 
            self.logger.error(f"NHSAssistant initialization failed due to ModelLoadError: {mle}", exc_info=True)
            self.is_initialized = False
            self.circuit_state = self.CIRCUIT_OPEN 
            self.last_failure_time = time.time()
            self.audit_logger.error(f"SYSTEM_EVENT - Type: AssistantInitialized, Status: Failed, Reason: {type(mle).__name__} - {str(mle)[:100]}")


    def get_response(self, user_input: str) -> str:
        # Using a simple hash for audit to avoid PII. In real systems, use proper PII handling.
        input_hash = abs(hash(user_input)) % (10**8) 
        self.audit_logger.info(f"QUERY_RECEIVED - InputHash: {input_hash:08d}")

        if not self.is_initialized:
            self.logger.warning("Attempting to get response but assistant not initialized. Trying to initialize now.")
            self.initialize() 
            if not self.is_initialized:
                self.audit_logger.warning(f"FALLBACK_USED - InputHash: {input_hash:08d}, Reason: AssistantNotInitialized")
                return self.error_handler.fallback_guidance["technical_difficulty"] + " (Assistant not ready)"

        sanitized_input = self._sanitize_input(user_input)

        if self._check_emergency_keywords(sanitized_input):
            self.audit_logger.warning(f"RESPONSE_TYPE - InputHash: {input_hash:08d}, Type: EmergencyMessage")
            return self.EMERGENCY_MESSAGE 

        current_time = time.time()
        if self.circuit_state == self.CIRCUIT_OPEN:
            if current_time - self.last_failure_time > self.COOLDOWN_PERIOD:
                self.circuit_state = self.CIRCUIT_HALF_OPEN
                self.logger.info("Circuit breaker cooldown expired. Moved to HALF_OPEN state.")
                self.audit_logger.info(f"CIRCUIT_BREAKER_EVENT - InputHash: {input_hash:08d}, StateChange: HALF_OPEN, Reason: CooldownExpired")
            else:
                self.logger.warning("Circuit breaker is OPEN. Using offline/fallback guidance.")
                self.audit_logger.warning(f"FALLBACK_USED - InputHash: {input_hash:08d}, Reason: CircuitOpen, FallbackType: OfflineGuidance")
                return self._get_offline_response(sanitized_input)

        try:
            response = self.model_loader.generate_response(sanitized_input) 
            
            # Check if the response from model_loader indicates a model failure (e.g., it returned a fallback)
            # This check needs to be robust against the actual fallback messages from ModelLoader
            if response == self.model_loader.error_handler.fallback_guidance.get("technical_difficulty", "unique_fallback_string"):
                self.logger.warning("ModelLoader returned its technical_difficulty fallback, treating as a failure for circuit breaker.")
                raise ModelLoadError("ModelLoader returned a fallback, indicating an internal issue.")

            if self.circuit_state == self.CIRCUIT_HALF_OPEN:
                self.circuit_state = self.CIRCUIT_CLOSED
                self.failure_count = 0
                self.logger.info("Successful response in HALF_OPEN state. Circuit breaker RESET to CLOSED.")
                self.audit_logger.info(f"CIRCUIT_BREAKER_EVENT - InputHash: {input_hash:08d}, StateChange: CLOSED, Reason: SuccessfulResponseInHalfOpen")
            
            self.audit_logger.info(f"RESPONSE_TYPE - InputHash: {input_hash:08d}, Type: ModelGenerated")
            if len(response) > 20 and "sorry" not in response.lower() and "unable" not in response.lower() and "assistant" not in response.lower() : # Avoid disclaimer on short/error/meta messages
                return f"{self.MEDICAL_DISCLAIMER}\n\n{response}"
            return response

        except Exception as e: 
            self.logger.error(f"Error getting response from model (InputHash: {input_hash:08d}): {type(e).__name__} - {str(e)[:100]}", exc_info=True)
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.audit_logger.error(f"MODEL_FAILURE_EVENT - InputHash: {input_hash:08d}, Count: {self.failure_count}, ErrorType: {type(e).__name__}")

            if self.circuit_state == self.CIRCUIT_HALF_OPEN or self.failure_count >= self.FAILURE_THRESHOLD:
                self.circuit_state = self.CIRCUIT_OPEN
                self.logger.warning(f"Circuit breaker OPENED due to failures (count: {self.failure_count}).")
                self.audit_logger.warning(f"CIRCUIT_BREAKER_EVENT - InputHash: {input_hash:08d}, StateChange: OPEN, Reason: FailureThresholdReached")
            
            self.audit_logger.warning(f"FALLBACK_USED - InputHash: {input_hash:08d}, Reason: ModelError, FallbackType: OfflineGuidance")
            return self._get_offline_response(sanitized_input)


        def _get_offline_response(self, user_input: str) -> str:
            input_hash = abs(hash(user_input)) % (10**8) 
            self.logger.info(f"Attempting to find offline guidance for InputHash: {input_hash:08d}...")
            input_lower = user_input.lower()
            for keyword, guidance in self.offline_guidance.items():
                if keyword in input_lower:
                    self.logger.info(f"Offline guidance found for keyword '{keyword}'.")
                    self.audit_logger.info(f"OFFLINE_RESPONSE_PROVIDED - InputHash: {input_hash:08d}, Keyword: {keyword}")
                    return guidance + " (This is basic guidance as the AI assistant is currently limited.)"
            
            self.logger.info("No specific offline guidance found. Returning generic A&E fallback.")
            self.audit_logger.info(f"OFFLINE_RESPONSE_PROVIDED - InputHash: {input_hash:08d}, Type: GenericFallback")
            # Use the error_handler's fallback to ensure consistency if it changes
            return self.error_handler.fallback_guidance.get("ae_questions", "Please provide symptoms, history, allergies, and medications in A&E.") + " (The AI assistant is experiencing issues.)"
            
    if __name__ == '__main__':
        logger_main = setup_logging() 
        audit_logger_main = logging.getLogger('NHS_Navigator_Audit')
        logger_main.info("NHSAssistant self-test (Phase 9) started.")
        audit_logger_main.info("SELF_TEST_EVENT - Type: SystemStart, Component: NHSAssistantMain")

        assistant = NHSAssistant()
        assistant.initialize()

        print("\n--- Testing Normal Response ---")
        print(assistant.get_response("What are common A&E questions?"))
        audit_logger_main.info("SELF_TEST_EVENT - TestNormalResponse")

        print("\n--- Testing Emergency Keyword ---")
        print(assistant.get_response("I have severe chest pain and I can't breathe!"))
        audit_logger_main.info("SELF_TEST_EVENT - TestEmergencyKeyword")

        print("\n--- Testing Sanitization ---")
        print(assistant.get_response("Test with <script>alert('bad')</script> and lots of spaces   and new\nlines."))
        audit_logger_main.info("SELF_TEST_EVENT - TestSanitization")

        print("\n--- Testing Circuit Breaker ---")
        # Temporarily mock model_loader's generate_response to simulate failures
        original_generate_response = assistant.model_loader.generate_response
        assistant.model_loader.generate_response = lambda p: (_ for _ in ()).throw(ModelLoadError("Simulated persistent model failure in test"))
        
        for i in range(NHSAssistant.FAILURE_THRESHOLD + 1):
            print(f"Simulating failure attempt {i+1}:")
            response = assistant.get_response("Trigger model failure for circuit breaker test")
            print(response)
            if assistant.circuit_state == NHSAssistant.CIRCUIT_OPEN:
                print(f"Circuit is now OPEN after {i+1} failures.")
                break
        audit_logger_main.info("SELF_TEST_EVENT - TestCircuitBreakerOpen")

        print("\n--- Testing Cooldown and HALF_OPEN ---")
        if assistant.circuit_state == NHSAssistant.CIRCUIT_OPEN:
            assistant.last_failure_time = time.time() - NHSAssistant.COOLDOWN_PERIOD - 5 # Force cooldown expiry
            print("Simulated cooldown period expired by setting last_failure_time back.")
            
            # Restore original generate_response or a successful mock for recovery test
            assistant.model_loader.generate_response = lambda p: "Simulated successful recovery response from mock!"
            print("Next call should move to HALF_OPEN, then to CLOSED on success.")
            response_after_cooldown = assistant.get_response("Test recovery after cooldown")
            print(response_after_cooldown)
            print(f"Circuit state after recovery attempt: {assistant.circuit_state}")
            if assistant.circuit_state != NHSAssistant.CIRCUIT_CLOSED:
                 logger_main.error("Circuit breaker did not close after successful half-open attempt!")
        else:
            logger_main.warning("Circuit breaker did not open as expected in test, skipping HALF_OPEN test.")
        
        assistant.model_loader.generate_response = original_generate_response # Restore original method
        audit_logger_main.info("SELF_TEST_EVENT - TestCircuitBreakerHalfOpenAndClose")
        
        print("\n--- Testing Offline Guidance (after circuit opened) ---")
        if assistant.circuit_state == NHSAssistant.CIRCUIT_OPEN:
            print(assistant.get_response("I need A&E help")) # Should trigger 'ae help' from cache
            print(assistant.get_response("Tell me something random")) # Should trigger generic fallback
        else:
            logger_main.warning("Circuit breaker not OPEN, offline guidance test might not be effective.")
            # Force open for testing offline
            assistant.circuit_state = NHSAssistant.CIRCUIT_OPEN
            assistant.last_failure_time = time.time() - 10 # Ensure it's not in cooldown for HALF_OPEN
            print("Forced circuit to OPEN for offline test.")
            print(assistant.get_response("I need A&E help for information")) 
            print(assistant.get_response("What is a paracetamol"))


        audit_logger_main.info("SELF_TEST_EVENT - TestOfflineGuidance")
        logger_main.info("NHSAssistant self-test (Phase 9) finished.")
