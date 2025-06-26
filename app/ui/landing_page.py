import streamlit as st
import base64
import os

def set_bg_with_floating_bubble():
    try:
        with open("assets/background-photo.png", "rb") as image_file:
            bg_encoded = base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error("Background image not found. Please ensure 'assets/background-photo.png' exists.")
        return

    try:
        with open("assets/bubble.png", "rb") as bubble_file:
            bubble_encoded = base64.b64encode(bubble_file.read()).decode()
    except FileNotFoundError:
        st.error("Bubble image not found. Please ensure 'assets/bubble.png' exists.")
        return

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Luckiest+Guy&display=swap');

        .stApp {{
            background-image: url(data:image/png;base64,{bg_encoded});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .main .block-container {{
            padding-top: 0;
            padding-bottom: 0;
            max-width: 100%;
        }}

        .floating-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            overflow: hidden;
        }}

        .floating-bubble {{
            position: relative;
            width: 700px;
            height: 700px;
            background-image: url(data:image/png;base64,{bubble_encoded});
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            animation: float 3s ease-in-out infinite;
        }}

        @keyframes float {{
            0%, 100% {{
                transform: translateY(0px);
            }}
            50% {{
                transform: translateY(-20px);
            }}
        }}

        .bubble-text {{
            font-family: 'Luckiest Guy', cursive;
            color: #E72B29;
            text-align: center;
            font-size: 4rem;
            line-height: 1;
            margin-bottom: 30px;
            text-shadow: 4px 4px 8px rgba(0,0,0,0.5);
        }}

        .quiz-button {{
            font-family: 'Luckiest Guy', cursive;
            background-color: #E72B29;
            color: white;
            border: none;
            padding: 20px 40px;
            font-size: 1.8rem;
            border-radius: 30px;
            cursor: pointer;
            box-shadow: 4px 4px 12px rgba(0,0,0,0.4);
            transition: all 0.3s ease;
            text-transform: uppercase;
        }}

        .quiz-button:hover {{
            background-color: #C6241F;
            transform: translateY(-2px);
            box-shadow: 6px 6px 16px rgba(0,0,0,0.5);
        }}

        @media (max-width: 768px) {{
            .floating-bubble {{
                width: 500px;
                height: 500px;
            }}
            .bubble-text {{
                font-size: 2.5rem;
            }}
            .quiz-button {{
                font-size: 1.3rem;
                padding: 15px 30px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def show():
    set_bg_with_floating_bubble()

    st.markdown(
        """
        <div class="floating-container">
            <div class="floating-bubble">
                <div class="bubble-text">
                    EPITHET<br>
                    PERSONALITY<br>
                    GAME
                </div>
                <form action="?page=quiz" method="get">
                    <button class="quiz-button" type="submit">
                        Go to Quiz
                    </button>
                </form>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
