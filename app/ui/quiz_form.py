import streamlit as st
import base64
import json
from pathlib import Path

def load_questions():
    """Load questions from JSON file"""
    try:
        with open('questions.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['questions']
    except FileNotFoundError:
        st.error("questions.json file not found!")
        return []
    except json.JSONDecodeError:
        st.error("Error reading questions.json file!")
        return []

def set_quiz_background(index):
    bg_files = [f"assets/questions/bg{i+1}.png" for i in range(15)]

    if index < 0 or index >= len(bg_files):
        st.error("Invalid question index for background.")
        return

    bg_path = Path(bg_files[index])
    if not bg_path.exists():
        st.error(f"Background image not found at {bg_path}")
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
        </style>
        """,
        unsafe_allow_html=True
    )

def get_question_options(question_index):
    """Returns the options for each question based on the question index"""
    
    # Load questions from JSON file
    all_questions = load_questions()
    
    # Return the options for the current question, or default if index is out of range
    if 0 <= question_index < len(all_questions):
        return all_questions[question_index]
    else:
        return {
            "A": "Default Option A",
            "B": "Default Option B", 
            "C": "Default Option C",
            "D": "Default Option D"
        }

def show():
    if "question_index" not in st.session_state:
        st.session_state.question_index = 0

    # Add vertical space to push buttons downward
    st.markdown("<div style='height: 560px;'></div>", unsafe_allow_html=True)

    # Navigation buttons at bottom
    nav1, nav2 = st.columns([1, 1])
    with nav1:
        pass  
    with nav2:
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("⬅ PREV") and st.session_state.question_index > 0:
                st.session_state.question_index -= 1
        with col_next:
            if st.button("NEXT ➡") and st.session_state.question_index < 14:
                st.session_state.question_index += 1

    set_quiz_background(st.session_state.question_index)

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

    # Get the current question's options
    current_options = get_question_options(st.session_state.question_index)
    
    option_texts = [
        f"A) {current_options['A']}",
        f"B) {current_options['B']}", 
        f"C) {current_options['C']}",
        f"D) {current_options['D']}"
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
        <button class="image-button" onclick="alert('Option A Clicked: {current_options['A']}')" data-option="A">
            <img src="data:image/png;base64,{button_a}" />
            <span class="button-text">{option_texts[0]}</span>
        </button>
        <button class="image-button" onclick="alert('Option B Clicked: {current_options['B']}')" data-option="B">
            <img src="data:image/png;base64,{button_b}" />
            <span class="button-text">{option_texts[1]}</span>
        </button>
        <button class="image-button" onclick="alert('Option C Clicked: {current_options['C']}')" data-option="C">
            <img src="data:image/png;base64,{button_c}" />
            <span class="button-text">{option_texts[2]}</span>
        </button>
        <button class="image-button" onclick="alert('Option D Clicked: {current_options['D']}')" data-option="D">
            <img src="data:image/png;base64,{button_d}" />
            <span class="button-text">{option_texts[3]}</span>
        </button>
    </div>

    <style>
        div[data-testid="column"]:has(button) {{
        position: fixed !important;
        bottom: 20px;
        right: 30px;
        display: flex !important;
        gap: 5px !important;
        z-index: 10000;
    }}
    .image-button {{
        background: none;
        border: none;
        padding: 0;
        margin-bottom: 3px;
        cursor: pointer;
        transition: transform 0.2s ease;
        position: relative;
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
    </style>
    """

    st.markdown(buttons_html, unsafe_allow_html=True)