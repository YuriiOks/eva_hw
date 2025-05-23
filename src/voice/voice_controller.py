import logging # Added for audit logger
from typing import Optional # Added for type hinting NHSAssistant
try:
    from src.voice.speech_to_text import SpeechToText
    from src.voice.text_to_speech import NHSTextToSpeech
    from src.inference.nhs_assistant import NHSAssistant 
    from src.utils.error_handling import setup_logging # For getting main logger
except ImportError as e:
    print(f"Error importing for VoiceController (Phase 9): {e}")
    # Dummy classes
    class SpeechToText: 
        def listen_for_speech(self, timeout=0): return None, "Dummy STT error"
        def recognize_speech(self, _): return "Dummy recognized speech"
        def recognize_from_streamlit_audio(self, _): return "Dummy recog from streamlit"
    class NHSTextToSpeech:
        def speak_nhs_phrase(self, _): pass
        def speak(self, _): pass
        def get_streamlit_audio(self, _): return b"dummy_audio"
    class NHSAssistant:
        def __init__(self): self.is_initialized = False # Ensure dummy has this
        def initialize(self): self.is_initialized = True # Simulate successful init for dummy
        def get_response(self, _): return "Dummy NHSAssistant response"
    def setup_logging(): import logging; return logging.getLogger("DummyVCLogger")

class VoiceController:
    def __init__(self):
        self.logger = setup_logging() # Main logger
        self.audit_logger = logging.getLogger('NHS_Navigator_Audit') # Audit logger

        self.stt = SpeechToText()
        self.tts = NHSTextToSpeech()
        self.nhs_assistant: Optional[NHSAssistant] = None 
        self.is_listening = False
        self.conversation_active = False
        self.logger.info("VoiceController initialized (Phase 9).")
    
    def initialize_assistant(self):
        if not self.nhs_assistant: # Check if it's already initialized (could be None or an instance)
            self.logger.info("Initializing NHS Assistant for voice controller...")
            self.audit_logger.info("SYSTEM_EVENT - Type: AssistantInitializationAttempt, Component: VoiceController")
            self.nhs_assistant = NHSAssistant()
            self.nhs_assistant.initialize() # NHSAssistant's own init handles its audit
            if self.nhs_assistant.is_initialized:
                 self.logger.info("NHS Assistant ready for voice interaction.")
                 # No separate audit log here, NHSAssistant.initialize() handles its own success/failure audit
            else:
                 self.logger.error("NHS Assistant failed to initialize properly for VoiceController.")
                 # NHSAssistant.initialize() should have logged the failure reason to audit
        elif not self.nhs_assistant.is_initialized: # If instance exists but not initialized
            self.logger.warning("NHS Assistant instance exists but was not initialized. Attempting initialization again.")
            self.audit_logger.info("SYSTEM_EVENT - Type: AssistantReinitializationAttempt, Component: VoiceController")
            self.nhs_assistant.initialize()


    def start_conversation(self) -> str:
        self.initialize_assistant() 
        self.conversation_active = True
        # Use a simple hash for audit to avoid PII. In real systems, use proper PII handling.
        # For generic events like Start/End, a SessionID would be more appropriate if available.
        # session_id_hash = abs(hash(time.time())) % (10**8) # Example session ID
        self.audit_logger.info(f"VOICE_CONVERSATION_EVENT - Type: Start") # Removed SessionID for simplicity
        self.tts.speak_nhs_phrase("welcome")
        return "🎤 Voice conversation started. Speak now... (simulated)"
    
    def listen_and_respond(self) -> str:
        if not self.conversation_active: 
            self.audit_logger.warning("VOICE_INPUT_ERROR - Reason: ConversationNotActive")
            return "Voice conversation not active."
        
        self.initialize_assistant() # Ensure assistant is ready, re-check
        if not self.nhs_assistant or not self.nhs_assistant.is_initialized:
            self.audit_logger.error("VOICE_INPUT_ERROR - Reason: AssistantNotReady")
            return "AI Assistant not ready. Please try again shortly."

        self.is_listening = True
        self.audit_logger.info("VOICE_INPUT_EVENT - Type: ListenAttempt")
        audio, error = self.stt.listen_for_speech(timeout=10) 
        self.is_listening = False
        
        if error:
            self.tts.speak_nhs_phrase("error")
            self.audit_logger.warning(f"VOICE_INPUT_ERROR - Type: ListenFailed, Details: {str(error)[:100]}") # Limit error string length
            return f"❌ STT Error: {error}"
        
        if not audio: # Handle case where listen_for_speech might return (None, None) on timeout without error
            self.tts.speak_nhs_phrase("error")
            self.audit_logger.warning(f"VOICE_INPUT_ERROR - Type: NoAudioDetected")
            return "❌ STT Error: No audio detected."

        user_text = self.stt.recognize_speech(audio)
        if not user_text:
            self.tts.speak_nhs_phrase("error")
            self.audit_logger.warning("VOICE_INPUT_ERROR - Type: RecognitionFailed")
            return "❌ Could not understand speech (simulated)"
        
        # Using a simple hash for audit to avoid PII.
        input_hash = abs(hash(user_text)) % (10**8)
        self.audit_logger.info(f"VOICE_INPUT_RECEIVED - InputHash: {input_hash:08d}, TextLength: {len(user_text)}")
        
        if any(word in user_text.lower() for word in ['goodbye', 'bye', 'stop', 'exit']):
            return self.end_conversation()
        
        self.tts.speak("Let me process that for you.") 
        ai_response = self.nhs_assistant.get_response(user_text) # NHSAssistant handles its own audit for query processing
        
        self.tts.speak(ai_response) 
        output_hash = abs(hash(ai_response)) % (10**8)
        self.audit_logger.info(f"VOICE_RESPONSE_SENT - OutputHash: {output_hash:08d}, TextLength: {len(ai_response)}")
        return f"You said: {user_text}\nJARVIS: {ai_response}" # In sim, actual text fine for main log / UI
    
    def end_conversation(self) -> str:
        self.conversation_active = False
        self.tts.speak_nhs_phrase("goodbye")
        self.audit_logger.info("VOICE_CONVERSATION_EVENT - Type: End")
        return "🔇 Voice conversation ended."
    
    def process_audio_file(self, audio_bytes: bytes) -> tuple[str, Optional[bytes]]:
        self.initialize_assistant()
        if not self.nhs_assistant or not self.nhs_assistant.is_initialized:
             self.audit_logger.error("AUDIO_FILE_ERROR - Reason: AssistantNotReady")
             return "AI Assistant not ready.", None
        
        self.audit_logger.info(f"AUDIO_FILE_RECEIVED - SizeBytes: {len(audio_bytes)}")
        user_text = self.stt.recognize_from_streamlit_audio(audio_bytes)
        
        if not user_text:
            self.audit_logger.warning("AUDIO_FILE_ERROR - Type: RecognitionFailed")
            return "❌ Could not understand audio from file.", None
        
        input_hash = abs(hash(user_text)) % (10**8)
        self.audit_logger.info(f"AUDIO_FILE_TEXT_EXTRACTED - InputHash: {input_hash:08d}, TextLength: {len(user_text)}")
        ai_response = self.nhs_assistant.get_response(user_text) # NHSAssistant handles its own audit
        response_audio = self.tts.get_streamlit_audio(ai_response)
        
        output_hash = abs(hash(ai_response)) % (10**8)
        self.audit_logger.info(f"AUDIO_FILE_RESPONSE_GENERATED - OutputHash: {output_hash:08d}, ResponseAudioBytes: {len(response_audio) if response_audio else 0}")
        return f"From audio file: {user_text}\nJARVIS: {ai_response}", response_audio

if __name__ == '__main__':
    main_logger_vc = setup_logging() # Sets up both loggers
    audit_logger_vc = logging.getLogger('NHS_Navigator_Audit') # Get instance for direct use if needed
    main_logger_vc.info("VoiceController self-test (Phase 9) started.")
    audit_logger_vc.info("SELF_TEST_EVENT - Type: SystemStart, Component: VoiceControllerMain")
    
    vc = VoiceController()
    
    # Test conversation flow
    main_logger_vc.info("Testing voice conversation flow...")
    print(vc.start_conversation())
    
    # Simulate a listen and respond cycle 
    # Mocking STT parts for testing purposes, as real audio input isn't available
    vc.stt.listen_for_speech = lambda timeout=0: ("simulated_audio_data_vc_main", None) # type: ignore
    vc.stt.recognize_speech = lambda audio_data: "Tell me about A&E" # type: ignore
    
    print(vc.listen_and_respond())
    
    print(vc.end_conversation())
    audit_logger_vc.info("SELF_TEST_EVENT - Type: VoiceControllerTestComplete, Details: Full conversation flow simulated.")
    
    # Test audio file processing
    main_logger_vc.info("Testing audio file processing...")
    vc.stt.recognize_from_streamlit_audio = lambda audio_bytes: "What is a GP?" # type: ignore
    result_text, result_audio = vc.process_audio_file(b"simulated_audio_bytes_for_file_processing")
    print(f"Audio file processing result: {result_text}")
    if result_audio:
        main_logger_vc.info(f"Generated {len(result_audio)} bytes of audio for file processing response.")
    audit_logger_vc.info("SELF_TEST_EVENT - Type: VoiceControllerTestComplete, Details: Audio file processing simulated.")

    main_logger_vc.info("VoiceController self-test (Phase 9) finished.")
