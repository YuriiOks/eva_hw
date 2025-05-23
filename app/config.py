import streamlit as st

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="JARVIS AI - NHS Navigator",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo', # Placeholder
            'Report a bug': 'https://github.com/your-repo/issues', # Placeholder
            'About': """
            # JARVIS AI - NHS Navigator
            Revolutionizing healthcare communication through AI.
            Born from real NHS experience where a patient was asked 
            the same 7 questions 17 times in 12 hours.
            Built with ❤️ and determination in 5 hours.
            """
        }
    )

def load_custom_css():
    """Load custom CSS styling (as provided in README Step 33)"""
    st.markdown("""
    <style>
    /* NHS Color Scheme */
    :root {
        --nhs-blue: #003087;
        --nhs-light-blue: #0072ce;
        --nhs-green: #009639;
        --nhs-grey: #425563;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: var(--nhs-blue);
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* NHS branded containers */
    .nhs-container { /* This class was used in milestone3_avatar.py, ensure consistency or add if needed */
        background: linear-gradient(135deg, var(--nhs-blue), var(--nhs-light-blue));
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Additional class from milestone1_basic.py for consistency */
    .nhs-blue { 
        background-color: #003087; 
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 15px; /* README had 15px, milestone1 had 10px. Standardizing to 15px */
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb); /* From README */
        border-left: 4px solid var(--nhs-light-blue);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f1f8e9, #c8e6c9); /* From README */
        border-left: 4px solid var(--nhs-green);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--nhs-blue), var(--nhs-light-blue));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric styling */
    .metric-container { /* This class was used in milestone3_avatar.py */
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--nhs-green);
        margin: 0.5rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)

def show_loading_animation(): # As per README Step 33
    """Show loading animation"""
    return st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
        <div style="border: 4px solid #f3f3f3; border-top: 4px solid #003087; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite;"></div>
    </div>
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
