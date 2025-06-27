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
    
    # Your quiz options text
    option_texts = [
        "A) Your first option text here",
        "B) Your second option text here", 
        "C) Your third option text here",
        "D) Your fourth option text here"
    ]
    
    buttons_html = f"""
    <div style="
        position: fixed;
        top: 60%;
        left: 55%;
        transform: translate(-50%, -50%);
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 999;
    ">
        <button class="image-button" onclick="alert('Option A Clicked')" data-option="A">
            <img src="data:image/png;base64,{button_a}" />
            <span class="button-text">{option_texts[0]}</span>
        </button>
        <button class="image-button" onclick="alert('Option B Clicked')" data-option="B">
            <img src="data:image/png;base64,{button_b}" />
            <span class="button-text">{option_texts[1]}</span>
        </button>
        <button class="image-button" onclick="alert('Option C Clicked')" data-option="C">
            <img src="data:image/png;base64,{button_c}" />
            <span class="button-text">{option_texts[2]}</span>
        </button>
        <button class="image-button" onclick="alert('Option D Clicked')" data-option="D">
            <img src="data:image/png;base64,{button_d}" />
            <span class="button-text">{option_texts[3]}</span>
        </button>
    </div>
    
    <!-- Navigation Buttons outside floating container -->
    <div class="quiz-nav-buttons">
        <button class="quiz-button">PREV</button>
        <button class="quiz-button">NEXT</button>
    </div>
    
    <style>
    .image-button {{
        background: none;
        border: none;
        padding: 0;
        margin-bottom: 3px;
        cursor: pointer;
        transition: transform 0.2s ease;
        position: relative; /* Important for absolute positioning of text */
        display: inline-block;
    }}
    
    .image-button:hover {{
        transform: scale(1.02);
    }}
    
    .image-button:active {{
        transform: scale(0.98);
    }}
    
    .image-button img {{
        width: 650px;
        display: block;
        pointer-events: none;
        user-select: none;
    }}
    
    .button-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #333;
        font-size: 18px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
        text-align: center;
        pointer-events: none;
        user-select: none;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
        max-width: 80%;
        line-height: 1.2;
        z-index: 1;
    }}
    
    .quiz-nav-buttons {{
        position: fixed;
        bottom: 20px;
        right: 300px;
        display: flex;
        gap: 20px;
        z-index: 999;
    }}
    
    .quiz-button {{
        padding: 8px 24px;
        font-size: 16px;
        font-family: 'Samaritan Antique', cursive;
        background-color: #FF9500;
        color: white;
        border: none;
        cursor: pointer;
        box-shadow: 0px 4px 8px rgba(0,0,0,1);
        transition: transform 0.2s, background-color 0.3s;
    }}
    
    .quiz-button:hover {{
        background-color: #e57f00;
        transform: scale(1.05);
    }}
    </style>
    """
    
    st.markdown(buttons_html, unsafe_allow_html=True)