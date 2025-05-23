import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.voice.voice_controller import VoiceController
    # NHSAssistant might not be directly used here if VoiceController handles it
    # from src.inference.nhs_assistant import NHSAssistant 
    from app.config import setup_page_config, load_custom_css
except ImportError as e:
    st.error(f"Failed to import modules for Milestone 2: {e}")
    # Define dummy classes
    class VoiceController:
        def __init__(self): 
            print("Dummy VoiceController for UI M2.")
            self.tts = self.TTS() # Assign dummy instances
            self.stt = self.STT()
        def start_conversation(self): return "Dummy convo started."
        def end_conversation(self): return "Dummy convo ended."
        def listen_and_respond(self): return "Dummy listen and respond."
        def process_audio_file(self, _): return "Dummy audio process result.", None
        # def tts_speak(self, _): pass # Dummy method for tts - covered by self.tts.speak
        # def stt_listen_for_speech(self, timeout=0): return None, "Dummy STT error M2" # Covered by self.stt
        # def stt_recognize_speech(self, _): return "Dummy recognized speech M2" # Covered by self.stt
        class TTS: # Nested dummy for tts attribute
            def speak(self, _): print("Dummy TTS speak called")
        class STT: # Nested dummy for stt attribute
            def listen_for_speech(self, timeout=0): return ("simulated_audio_data_m2", None) # Return tuple
            def recognize_speech(self, _): return "Dummy recognized speech M2"
        
    def setup_page_config(): pass
    def load_custom_css(): pass

setup_page_config()
load_custom_css()

@st.cache_resource
def get_voice_controller():
    vc = VoiceController()
    # vc.initialize_assistant() # VoiceController's init or a dedicated method should handle this
    return vc

def main():
    st.markdown('<h1 class="main-header">🎤 JARVIS AI - Voice Enabled (Milestone 2)</h1>', unsafe_allow_html=True)
    
    voice_controller = get_voice_controller()

    st.sidebar.markdown("## 🗣️ Voice Controls")
    
    if 'voice_active_m2' not in st.session_state:
        st.session_state.voice_active_m2 = False

    if st.session_state.voice_active_m2:
        st.sidebar.success("🎤 Voice Active")
        if st.sidebar.button("🔇 Stop Voice Chat", key="stop_voice_m2"):
            status = voice_controller.end_conversation()
            st.session_state.voice_active_m2 = False
            st.sidebar.info(status)
            st.rerun()
    else:
        st.sidebar.info("🔇 Voice Inactive")
        if st.sidebar.button("🎤 Start Voice Chat", key="start_voice_m2"):
            status = voice_controller.start_conversation()
            st.session_state.voice_active_m2 = True
            st.sidebar.success(status)
            st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["💬 Voice Chat", "📁 Audio Upload", "⚙️ Settings"])
    
    with tab1:
        st.markdown('<div class="nhs-container" style="background-color: var(--nhs-light-blue);"><h3>🎤 Voice Conversation</h3></div>', unsafe_allow_html=True)
        if st.session_state.voice_active_m2:
            st.info("🎤 Voice conversation is active. Click 'Listen & Respond' to interact.")
            if st.button("👂 Listen & Respond", type="primary", key="listen_respond_m2"):
                with st.spinner("🎤 Listening... (simulated)"):
                    result = voice_controller.listen_and_respond()
                st.text_area("Conversation Log:", result, height=200, key="conv_log_m2")
        else:
            st.warning("🔇 Voice conversation not active. Click 'Start Voice Chat' in sidebar.")
    
    with tab2:
        st.markdown('<div class="nhs-container" style="background-color: var(--nhs-green);"><h3>📁 Audio File Processing</h3></div>', unsafe_allow_html=True)
        audio_file = st.file_uploader("Choose audio file", type=['wav', 'mp3', 'm4a'], key="audio_upload_m2")
        if audio_file:
            st.audio(audio_file, format='audio/wav') # Display uploaded audio
            if st.button("🔄 Process Uploaded Audio", key="process_upload_m2"):
                with st.spinner("🔄 Processing audio... (simulated)"):
                    audio_bytes = audio_file.getvalue()
                    result, response_audio = voice_controller.process_audio_file(audio_bytes)
                st.text_area("Processing Result:", result, height=150, key="upload_result_m2")
                if response_audio:
                    st.markdown("### 🔊 JARVIS Response Audio (Simulated)")
                    st.audio(response_audio)

        st.markdown("--- ### 🎙️ Record Audio (Streamlit Component)")
        recorded_audio_bytes = st.audio_input("Record your question directly:", key="audio_input_m2")
        if recorded_audio_bytes:
            # st.audio(recorded_audio_bytes) # Displaying the recorded audio; audio_input already does this
            if st.button("🔄 Process Recording", key="process_record_m2"):
                with st.spinner("🔄 Processing recording... (simulated)"):
                    result, response_audio = voice_controller.process_audio_file(recorded_audio_bytes)
                st.text_area("Recording Result:", result, height=150, key="record_result_m2")
                if response_audio:
                    st.markdown("### 🔊 JARVIS Response Audio (Simulated)")
                    st.audio(response_audio)

    with tab3:
        st.markdown('<div class="nhs-container" style="background-color: var(--nhs-grey);"><h3>⚙️ Voice Settings</h3></div>', unsafe_allow_html=True)
        st.subheader("🗣️ Speech Settings (Placeholders)")
        st.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1, key="rate_m2")
        st.selectbox("Language", ["en", "en-uk", "en-us"], key="lang_m2")
        st.slider("Listening Timeout (seconds)", 5, 30, 10, key="timeout_m2")
        
        st.subheader("🧪 Test Voice System (Simulated)")
        test_text_tts = st.text_input("Test TTS:", "Hello, JARVIS AI here.", key="tts_test_m2")
        if st.button("🗣️ Test Speech", key="test_tts_m2"):
            with st.spinner("🗣️ Speaking... (simulated)"):
                voice_controller.tts.speak(test_text_tts) 
            st.success("✅ TTS test complete (simulated)")

        if st.button("🎤 Test Microphone & STT", key="test_stt_m2"):
            with st.spinner("🎤 Listening... (simulated)"):
                audio_sim, error_sim = voice_controller.stt.listen_for_speech(timeout=5)
            if error_sim:
                st.error(f"❌ {error_sim}")
            else:
                text_sim = voice_controller.stt.recognize_speech(audio_sim)
                if text_sim:
                    st.success(f"✅ Heard (simulated): {text_sim}")
                else:
                    st.warning("⚠️ No speech detected or understood (simulated)")
                        
if __name__ == "__main__":
    main()
