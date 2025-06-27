import streamlit as st
import base64
from pathlib import Path

def set_quiz_background():
    bg_path = Path("assets/questions/bg1.png")
    if not bg_path.exists():
        st.error("Background image not found at assets/question/bg1.png")
        return

    bg_encoded = base64.b64encode(bg_path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            animation: fadeIn 1s ease-in-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: rgba(255, 255, 255, 0.75); 
            border-radius: 20px;
            margin: 2rem auto;
            max-width: 800px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def show():
    set_quiz_background()
