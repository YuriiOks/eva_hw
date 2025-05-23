# app/main.py
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.milestone1_basic import main as milestone1_main
    from app.milestone2_voice import main as milestone2_main
    from app.milestone3_avatar import main as milestone3_main
    from app.config import setup_page_config, load_custom_css
except ImportError as e:
    st.error(f"Critical Error: Failed to import modules: {e}")
    def milestone1_main(): st.write("Milestone 1 unavailable.")
    def milestone2_main(): st.write("Milestone 2 unavailable.")
    def milestone3_main(): st.write("Milestone 3 unavailable.")
    def setup_page_config(): pass
    def load_custom_css(): pass

setup_page_config()
load_custom_css()

if 'main_system_initialized' not in st.session_state:
    st.session_state.main_system_initialized = False

@st.cache_resource
def initialize_jarvis_systems():
    print("Main JARVIS System Initialized (simulated)")
    return True

def main_app():
    st.markdown("<div style='text-align: center; background: linear-gradient(135deg, #003087, #0072ce); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'><h1 style='font-size: 3rem; margin: 0;'>JARVIS AI</h1><p style='font-size: 1rem;'>NHS Navigator</p></div>", unsafe_allow_html=True)

    if not st.session_state.main_system_initialized:
        with st.spinner("Initializing JARVIS..."):
            st.session_state.main_system_initialized = initialize_jarvis_systems()
        if st.session_state.main_system_initialized: st.sidebar.success("Systems Online!")
        else: st.sidebar.error("Systems Init Failed."); return

    tabs_titles = ["Mission", "Chat (M1)", "Voice (M2)", "Avatar (M3)", "Impact"]
    selected_tab = st.sidebar.radio("Navigate", tabs_titles)

    if selected_tab == "Mission":
        st.markdown('<h2>🎯 Mission Control</h2>', unsafe_allow_html=True)
        st.markdown("### The Crisis & The Solution..."); st.metric("NHS Budget", "£220B")
    elif selected_tab == "Chat (M1)":
        milestone1_main()
    elif selected_tab == "Voice (M2)":
        milestone2_main()
    elif selected_tab == "Avatar (M3)":
        milestone3_main()
    elif selected_tab == "Impact":
        st.markdown('<h2>📊 NHS Impact Analysis</h2>', unsafe_allow_html=True); st.markdown("### Potential Savings: £50+ Billion...")
if __name__ == "__main__": main_app()
