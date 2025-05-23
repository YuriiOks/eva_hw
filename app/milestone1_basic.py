import streamlit as st
import sys
import os

# Add src to path FOR DEVELOPMENT/ TESTING in IDE. 
# For streamlit run, ensure PYTHONPATH is set or structure allows imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.inference.nhs_assistant import NHSAssistant
except ImportError:
    # Fallback for scenarios where path isn't set up as expected during subtask execution
    st.error("Failed to import NHSAssistant. Ensure src is in PYTHONPATH.")
    # Define a dummy class to allow the rest of the UI to be defined
    class NHSAssistant:
        def __init__(self): self.is_loaded = False; print("Dummy NHSAssistant for UI.")
        def initialize(self): self.is_loaded = True; print("Dummy initialize.")
        def get_response(self, _): return "Dummy NHS Assistant response."
        def get_nhs_questions_prediction(self, _): return "Dummy prediction for NHS questions."
        def explain_medical_term(self, _): return "Dummy explanation for medical term."
        
# Page config
st.set_page_config(
    page_title="JARVIS AI - NHS Navigator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (from README)
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #003087; /* NHS Blue */
    text-align: center;
    margin-bottom: 2rem;
}
.nhs-blue {
    background-color: #003087; /* NHS Blue */
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.user-message {
    background-color: #e3f2fd; /* Light Blue */
    border-left: 4px solid #2196f3; /* Blue */
}
.assistant-message {
    background-color: #f1f8e9; /* Light Green */
    border-left: 4px solid #4caf50; /* Green */
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource # Cache the assistant object
def get_assistant():
    assistant = NHSAssistant()
    assistant.initialize() # Initialize it once
    return assistant

def main():
    st.markdown('<h1 class="main-header">🤖 JARVIS AI - NHS Navigator (Milestone 1)</h1>', unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    ## 🏥 About JARVIS AI
    **Mission**: Revolutionize NHS patient experience through AI.
    **Inspired by**: Real experience with NHS inefficiencies.
    **Goal**: Reduce the infamous "17 repeated questions" problem.
    """)
    st.sidebar.markdown("---")
    
    assistant = get_assistant() # Get the cached assistant

    st.sidebar.subheader("🤖 Model Status")
    if assistant and assistant.is_loaded: # Check if assistant and its model are loaded
        st.sidebar.success("✅ JARVIS AI Online")
    else:
        st.sidebar.warning("⏳ Initializing...") # Should ideally be handled by spinner during get_assistant
        # If get_assistant fails or assistant.is_loaded is false after init
        # This part might show if initialization within get_assistant had issues.

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="nhs-blue"><h3>💬 Chat with JARVIS AI</h3></div>', unsafe_allow_html=True)
        
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm JARVIS AI, your NHS Navigator (Basic Chat). How can I help you today?"}
            ]
        
        for message in st.session_state.messages:
            css_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(f'<div class="chat-message {css_class}"><strong>{message["role"].capitalize()}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        user_input = st.text_input("Ask JARVIS AI anything about the NHS:", key="user_input_m1")
        
        if st.button("Send", type="primary", key="send_m1") and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("🤖 JARVIS AI is thinking..."):
                response = assistant.get_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🚑 Broken Arm A&E Prep", key="prep_m1"):
            scenario = "My 61-year-old mother broke her arm and we're going to A&E"
            st.session_state.messages.append({"role": "user", "content": scenario})
            with st.spinner("🤖 Preparing A&E guidance..."):
                response = assistant.get_nhs_questions_prediction(scenario)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Simplified "Explain Medical Term" for Milestone 1
        term_to_explain = st.text_input("Enter medical term to explain:", key="medical_term_m1")
        if st.button("🩺 Explain Term", key="explain_m1") and term_to_explain:
            st.session_state.messages.append({"role": "user", "content": f"What is {term_to_explain}?"})
            with st.spinner("🤖 Explaining medical term..."):
                response = assistant.explain_medical_term(term_to_explain)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        if st.button("❓ Common NHS Questions", key="common_q_m1"):
            question = "What are the most common questions NHS staff ask patients?"
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("🤖 Listing common questions..."):
                response = assistant.get_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Impact Statistics")
        st.metric("NHS Annual Budget", "£220B")
        st.metric("Questions Repeated", "17x")
        st.metric("Potential Savings", "£50B+")
        
        if st.button("🗑️ Clear Chat", key="clear_m1"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm JARVIS AI, your NHS Navigator. How can I help you today?"}
            ]
            st.rerun()

if __name__ == "__main__":
    main()
