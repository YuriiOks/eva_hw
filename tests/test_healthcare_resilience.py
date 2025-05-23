# tests/test_healthcare_resilience.py
import unittest
from unittest.mock import patch, MagicMock
import time # For circuit breaker cooldown test
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils.error_handling import ErrorHandler, ModelLoadError, VoiceSystemError, AvatarError, ConfigurationError, NHSNavigatorError
    from src.inference.nhs_assistant import NHSAssistant
    from src.inference.model_loader import ModelLoader # ModelLoader itself for direct testing
    from src.training.config import ModelConfig
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in test_healthcare_resilience.py: {e}")
    # Define dummy classes if imports fail, so the test file can at least be parsed
    class ErrorHandler: fallback_guidance={"technical_difficulty": "Dummy fallback"}
    class ModelLoadError(Exception): pass
    class VoiceSystemError(Exception): pass
    class AvatarError(Exception): pass
    class ConfigurationError(Exception): pass
    class NHSNavigatorError(Exception): pass
    class NHSAssistant: 
        CIRCUIT_OPEN="OPEN"; CIRCUIT_CLOSED="CLOSED"; CIRCUIT_HALF_OPEN="HALF_OPEN"
        FAILURE_THRESHOLD=3; COOLDOWN_PERIOD=1 # Short for test
        MEDICAL_DISCLAIMER="Dummy Disclaimer"; EMERGENCY_MESSAGE="Dummy Emergency"
        CRITICAL_KEYWORDS=["dummy_emergency"]
        def __init__(self): self.circuit_state=self.CIRCUIT_CLOSED; self.failure_count=0; self.last_failure_time=0; self.offline_guidance={"help":"dummy offline help"}; self.is_initialized=True; self.model_loader=MagicMock(); self.logger=MagicMock(); self.audit_logger=MagicMock(); self.error_handler=ErrorHandler()
        def get_response(self, t): return "dummy_response"
        def _sanitize_input(self, t): return t
        def _check_emergency_keywords(self, t): return False
        def initialize(self): self.is_initialized = True

    class ModelLoader: 
        def __init__(self,c): self.is_properly_loaded = True; self.error_handler=ErrorHandler(); self.logger=MagicMock()
        def generate_response(self,p,ml=0): return "dummy_model_response"
    class ModelConfig: model_name="dummy"


class TestErrorHandler(unittest.TestCase):
    def setUp(self):
        self.handler = ErrorHandler()
        # Mock the logger to spy on its calls if needed, or check file output (harder in unit tests)
        self.handler.logger = MagicMock()

    def test_get_user_friendly_message_model_load_error(self):
        error = ModelLoadError("Simulated model loading failure.")
        message = self.handler.get_user_friendly_message(error, "test_model_load")
        self.assertIn("AI Model Error", message) # Check against the simplified message from ErrorHandler's handle_model_error
        self.handler.logger.warning.assert_called()

    def test_get_user_friendly_message_voice_system_error(self):
        # Assuming ErrorHandler is enhanced to handle VoiceSystemError specifically
        error = VoiceSystemError("Simulated mic issue.")
        # Temporarily patch the specific handler if get_user_friendly_message routes to it
        with patch.object(self.handler, 'handle_voice_error', return_value="Voice system error specific.") as mock_handle:
            message = self.handler.get_user_friendly_message(error, "test_voice_err")
            self.assertEqual(message, "Voice system error specific.")
            mock_handle.assert_called_once_with(error, "UserFriendly-test_voice_err")

    def test_get_user_friendly_message_generic_error(self):
        error = Exception("A very generic error.")
        message = self.handler.get_user_friendly_message(error, "test_generic")
        self.assertEqual(message, self.handler.fallback_guidance["technical_difficulty"])
        self.handler.logger.warning.assert_called()
        
    def test_handle_model_error_cuda(self):
        # This test assumes the internal logic of handle_model_error (which was simplified)
        # For full test, ErrorHandler would need more complex internal routing for this.
        # For now, we test the output of get_user_friendly_message for an error containing "CUDA"
        error = ModelLoadError("CUDA out of memory")
        # In the provided ErrorHandler, handle_model_error returns a generic message.
        # If it were more specific for CUDA:
        # with patch.object(self.handler, 'handle_model_error', return_value="CUDA specific error message") as mock_handle_cuda:
        #     message = self.handler.get_user_friendly_message(error, "test_cuda_error")
        #     self.assertEqual(message, "CUDA specific error message")
        # else, test the generic output:
        message = self.handler.get_user_friendly_message(error, "test_cuda_error")
        self.assertIn(self.handler.fallback_guidance["technical_difficulty"], message)


@patch('src.inference.model_loader.ErrorHandler') # Mock ErrorHandler within ModelLoader
class TestModelResilience(unittest.TestCase):
    def test_model_loader_generate_response_failure(self, MockErrorHandler):
        mock_error_handler_instance = MockErrorHandler.return_value
        mock_error_handler_instance.get_user_friendly_message.return_value = "Test Fallback: AI Unavailable"
        
        config = ModelConfig(model_name="test_model")
        loader = ModelLoader(config) # ModelLoader will use the mocked ErrorHandler
        loader.is_properly_loaded = True # Assume loaded for this test
        loader.model = "simulated_model" # Ensure it passes the check
        loader.tokenizer = "simulated_tokenizer"
        
        # Configure the (placeholder) generate_response to simulate an internal error
        # The actual ModelLoader's generate_response has its own try-except.
        # We need to ensure that when an error *does* occur inside it, it calls the error_handler.
        
        # To test this, we mock the internal call that might fail, e.g. model.generate
        # Since loader.model is a string in placeholder, we use the "fail_generation" trigger
        response = loader.generate_response("fail_generation prompt")
        self.assertEqual(response, "Test Fallback: AI Unavailable")
        # Check that get_user_friendly_message was called by the ModelLoader's generate_response
        mock_error_handler_instance.get_user_friendly_message.assert_called()


class TestNHSAssistantResilience(unittest.TestCase):
    def setUp(self):
        # Patch ModelLoader to control its behavior during NHSAssistant tests
        self.mock_model_loader_patch = patch('src.inference.nhs_assistant.ModelLoader')
        self.MockModelLoaderClass = self.mock_model_loader_patch.start()
        
        self.mock_model_loader_instance = MagicMock()
        self.mock_model_loader_instance.is_properly_loaded = True # Simulate loader itself is fine initially
        self.MockModelLoaderClass.return_value = self.mock_model_loader_instance
        
        # Patch the loggers for NHSAssistant
        self.mock_logger_patch = patch('src.inference.nhs_assistant.setup_logging')
        self.mock_audit_logger_patch = patch('src.inference.nhs_assistant.logging.getLogger')

        self.MockSetupLogging = self.mock_logger_patch.start()
        self.MockGetLogger = self.mock_audit_logger_patch.start()

        self.mock_main_logger = MagicMock()
        self.mock_audit_logger = MagicMock()
        self.MockSetupLogging.return_value = self.mock_main_logger
        # Ensure that any call to logging.getLogger with 'NHS_Navigator_Audit' returns our mock
        self.MockGetLogger.side_effect = lambda name: self.mock_audit_logger if name == 'NHS_Navigator_Audit' else MagicMock()

        self.assistant = NHSAssistant()
        self.assistant.is_initialized = True # Assume assistant init itself is fine
        self.assistant.model_loader = self.mock_model_loader_instance # Ensure it uses the mock

        NHSAssistant.COOLDOWN_PERIOD = 0.1 # Shorten for faster tests
        NHSAssistant.FAILURE_THRESHOLD = 2


    def tearDown(self):
        self.mock_model_loader_patch.stop()
        self.mock_logger_patch.stop()
        self.mock_audit_logger_patch.stop()

    def test_circuit_breaker_opens_and_uses_fallback(self):
        self.mock_model_loader_instance.generate_response.side_effect = ModelLoadError("Simulated persistent model failure")
        
        for i in range(NHSAssistant.FAILURE_THRESHOLD):
            self.assistant.get_response("test input to fail")
        
        self.assertEqual(self.assistant.circuit_state, NHSAssistant.CIRCUIT_OPEN)
        
        # Check audit log for circuit open event
        # Consolidate calls from different log levels if necessary
        all_audit_calls_args = []
        for call_args_list_item in [self.mock_audit_logger.info.call_args_list, self.mock_audit_logger.warning.call_args_list, self.mock_audit_logger.error.call_args_list]:
            all_audit_calls_args.extend([call_args[0][0] for call_args in call_args_list_item])
        
        self.assertTrue(any("CIRCUIT_BREAKER_EVENT - StateChange: OPEN" in call_arg for call_arg in all_audit_calls_args))

        response = self.assistant.get_response("another input")
        # Check if it's the offline A&E fallback from _get_offline_response
        self.assertIn(self.assistant.error_handler.fallback_guidance.get("ae_questions", "fallback ae_questions"), response)
        
        warning_audit_calls_args = [call_args[0][0] for call_args in self.mock_audit_logger.warning.call_args_list]
        self.assertTrue(any("FALLBACK_USED - Reason: CircuitOpen" in call_arg for call_arg in warning_audit_calls_args))


    def test_circuit_breaker_half_open_and_reset(self):
        self.mock_model_loader_instance.generate_response.side_effect = ModelLoadError("Fail")
        for _ in range(NHSAssistant.FAILURE_THRESHOLD):
            self.assistant.get_response("fail")
        self.assertEqual(self.assistant.circuit_state, NHSAssistant.CIRCUIT_OPEN)

        self.assistant.last_failure_time = time.time() - (NHSAssistant.COOLDOWN_PERIOD + 0.05) # Ensure cooldown passed
        
        self.mock_model_loader_instance.generate_response.side_effect = None
        self.mock_model_loader_instance.generate_response.return_value = "Successful recovery"
        
        response = self.assistant.get_response("test half-open success")
        self.assertEqual(self.assistant.circuit_state, NHSAssistant.CIRCUIT_CLOSED)
        self.assertIn("Successful recovery", response)
        self.assertEqual(self.assistant.failure_count, 0)
        
        info_audit_calls_args = [call_args[0][0] for call_args in self.mock_audit_logger.info.call_args_list]
        self.assertTrue(any("CIRCUIT_BREAKER_EVENT - StateChange: CLOSED" in call_arg for call_arg in info_audit_calls_args))


    def test_offline_mode_when_circuit_open(self):
        self.assistant.circuit_state = NHSAssistant.CIRCUIT_OPEN
        self.assistant.last_failure_time = time.time() # Ensure it's not in cooldown for half-open
        self.assistant.offline_guidance = {"test keyword": "Offline response for test keyword"} # Ensure a known keyword
        response = self.assistant.get_response("some input with test keyword here")
        self.assertIn("Offline response for test keyword", response)

    def test_input_sanitization(self):
        raw_input = "<script>alert('xss')</script> some text" + "a"*1050 # Make it exceed 1000 after initial part
        expected_sanitized_content = "alert('xss') some text" # Tags removed
        
        sanitized = self.assistant._sanitize_input(raw_input)
        self.assertNotIn("<script>", sanitized)
        self.assertTrue(sanitized.startswith(expected_sanitized_content)) # Check if the beginning is as expected
        self.assertTrue(len(sanitized) <= 1000) # Check overall truncation

    def test_medical_disclaimer_added(self):
        # Ensure the mock response is long enough to trigger disclaimer logic
        self.mock_model_loader_instance.generate_response.return_value = "This is a valid medical-like response that is certainly long enough for a standard medical disclaimer to be appended by the system."
        response = self.assistant.get_response("Valid question about symptoms")
        self.assertIn(NHSAssistant.MEDICAL_DISCLAIMER, response)

    def test_emergency_keyword_detection(self):
        # Use one of the actual keywords from NHSAssistant for robustness
        actual_emergency_keyword = NHSAssistant.CRITICAL_KEYWORDS[0] if NHSAssistant.CRITICAL_KEYWORDS else "severe bleeding" # Provide a default if list is empty
        response = self.assistant.get_response(f"I have {actual_emergency_keyword} and need help immediately.")
        self.assertEqual(response, NHSAssistant.EMERGENCY_MESSAGE)
        
        # Check that the audit logger was called with a message indicating emergency keyword detection
        warning_audit_calls_args = [call_args[0][0] for call_args in self.mock_audit_logger.warning.call_args_list]
        self.assertTrue(any(f"EMERGENCY_KEYWORD_DETECTED - Keyword: '{actual_emergency_keyword}'" in call_arg for call_arg in warning_audit_calls_args))


if __name__ == '__main__':
    # Create dummy log dir if it doesn't exist, for cases where ErrorHandler might try to log to file
    # This is more for local execution convenience than for the CI/sandbox.
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    unittest.main()
