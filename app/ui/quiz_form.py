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
            background-position: center top 3px;
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

    def load_button_base64(filename):
        path = Path(f"assets/buttons/{filename}")
        if not path.exists():
            st.error(f"Image not found: {filename}")
            return ""
        return base64.b64encode(path.read_bytes()).decode()

    button_a = load_button_base64("button_a.png")
    button_b = load_button_base64("button_b.png")
    button_c = load_button_base64("button_c.png")
    button_d = load_button_base64("button_d.png")

    buttons_html = f"""
    <div style="
        position: fixed;
        top: 63%;
        left: 55%;
        transform: translate(-50%, -50%);
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 999;
    ">
        <img src="data:image/png;base64,{button_a}" style="width: 650px; margin-bottom: 3px;" />
        <img src="data:image/png;base64,{button_b}" style="width: 650px; margin-bottom: 3px;" />
        <img src="data:image/png;base64,{button_c}" style="width: 650px; margin-bottom: 3px;" />
        <img src="data:image/png;base64,{button_d}" style="width: 650px; margin-bottom: 30px;" />
        <!-- Navigation Buttons -->
        <div class="quiz-nav-buttons">
            <button class="quiz-button">PREV</button>
            <button class="quiz-button">NEXT</button>
        </div>
    </div>

    <style>
    @import url('https://fonts.googleapis.com/css2?family=Samaritan+Antique&display=swap');

    .quiz-nav-buttons {{
        display: flex;
        justify-content: flex-end;
        gap: 20px;
        width: 650px;
        margin-top: -10px;
        padding-right: 20px;
    }}

    .quiz-button {{
        padding: 8px 24px;
        font-size: 16px;
        font-family: 'Samaritan Antique', cursive;
        background-color: #FF9500;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s, background-color 0.3s;
    }}

    .quiz-button:hover {{
        background-color: #e57f00;
        transform: scale(1.05);
    }}
    </style>
    """

    st.markdown(buttons_html, unsafe_allow_html=True)