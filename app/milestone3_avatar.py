# app/milestone3_avatar.py
import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.avatar.avatar_voice_integration import AvatarVoiceIntegration
    from app.config import setup_page_config, load_custom_css
except ImportError: # Dummy classes
    class AvatarVoiceIntegration:
        current_state = "idle"; avatar_controller = type("DummyAC",(),{"get_nhs_avatar":lambda s,c="",so=None:"dummy_b64"})()
        question_flow = type("DummyQF",(),{"process_response":lambda s,r="":{"status":"dummy"}, "get_preparation_advice":lambda s:{}})()
        initialize_system=lambda s:None; start_nhs_simulation=lambda s,sc="":{"status":"started","avatar_image":"","message":""}
        ask_next_question=lambda s:{"status":"asked","avatar_image":"","question_text":"","question_data":{"type":""}}
        process_voice_response=lambda s:{"status":"processed","avatar_image":"","acknowledgment":""}
        complete_simulation=lambda s:{"status":"completed","avatar_image":"","completion_message":"","preparation_advice":{}}
        demonstrate_repetition_problem=lambda s:{"repeated_questions":[],"time_wasted_simulated":0}
    setup_page_config=lambda:None; load_custom_css=lambda:None

setup_page_config(); load_custom_css()
@st.cache_resource def get_avatar_system(): system = AvatarVoiceIntegration(); system.initialize_system(); return system

def main():
    st.markdown("<h1 style='text-align:center;'>Virtual NHS Receptionist (M3)</h1>", unsafe_allow_html=True)
    avatar_system = get_avatar_system()
    if 'sim_state_m3' not in st.session_state: 
        st.session_state.sim_state_m3 = "idle"
        st.session_state.avatar_m3 = avatar_system.avatar_controller.get_nhs_avatar()
        st.session_state.msg_m3 = "Sarah is ready."
    
    col1, col2 = st.columns(2)
    with col1: 
        st.image(st.session_state.avatar_m3, width=300)
        st.info(st.session_state.msg_m3)
    with col2:
        if st.session_state.sim_state_m3 == "idle":
            if st.button("Begin Simulation", key="bgn_m3"):
                res = avatar_system.start_nhs_simulation()
                st.session_state.avatar_m3, st.session_state.msg_m3, st.session_state.sim_state_m3 = res["avatar_image"], res["message"], "active"
                st.rerun()
        elif st.session_state.sim_state_m3 == "active":
            if st.button("Next Question", key="nxt_m3"):
                res = avatar_system.ask_next_question()
                if res.get("status")=="sim_completed": st.session_state.sim_state_m3 = "completed" # Check if ask_next_question completed it
                st.session_state.avatar_m3, st.session_state.msg_m3 = res["avatar_image"], res.get("question_text", "End of questions.") # Added .get for safety
                st.rerun()
            ans = st.text_input("Your answer:", key="ans_m3")
            if st.button("Submit Answer", key="sub_ans_m3") and ans:
                res = avatar_system.question_flow.process_response(ans) # Direct to QF for text
                if res.get("status")=="completed": st.session_state.sim_state_m3 = "completed"
                st.session_state.msg_m3 = f"Recorded: {ans}. Next?"
                st.rerun()
            if st.button("End Simulation", key="end_m3"): st.session_state.sim_state_m3 = "completed"; st.rerun()
        elif st.session_state.sim_state_m3 == "completed":
            res = avatar_system.complete_simulation()
            st.session_state.msg_m3 = res["completion_message"]
            st.balloons()
            if st.button("New Simulation", key="new_m3"): st.session_state.sim_state_m3 = "idle"; st.rerun()
    if st.button("Demo Repetition Problem", key="rep_m3"): st.write(avatar_system.demonstrate_repetition_problem())
if __name__ == "__main__": main()
