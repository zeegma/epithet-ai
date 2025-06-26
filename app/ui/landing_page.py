import streamlit as st
import base64

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

        /* Updated button styling */
        div.stButton > button {{
            font-family: 'Luckiest Guy', cursive !important;
            background-color: #E72B29 !important;
            color: white !important;
            border: 3px solid #E72B29 !important;
            padding: 15px 35px !important;
            font-size: 1.5rem !important;
            border-radius: 25px !important;
            cursor: pointer !important;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.3) !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            font-weight: normal !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        }}

        div.stButton > button:hover {{
            background-color: white !important;
            color: #E72B29 !important;
            transform: translateY(-2px) !important;
            box-shadow: 4px 4px 12px rgba(0,0,0,0.4) !important;
        }}

        .stButton {{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin-top: 310px !important;
        }}

        @media (max-width: 768px) {{
            .floating-bubble {{
                width: 500px;
                height: 500px;
            }}
            .bubble-text {{
                font-size: 2.5rem;
            }}
            div.stButton > button {{
                font-size: 1.2rem !important;
                padding: 12px 25px !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def show():
    set_bg_with_floating_bubble()

    # Layout the container
    st.markdown(
        """
        <div class="floating-container">
            <div class="floating-bubble">
                <div class="bubble-text">
                    EPITHET<br>
                    PERSONALITY<br>
                    GAME
                </div>
        """,
        unsafe_allow_html=True
    )

    # Button with full styling - positioned within the bubble
    if st.button("GO TO QUIZ", key="go_to_quiz_button"):
        st.session_state.page = "quiz"
        st.rerun()  
    st.markdown("</div></div>", unsafe_allow_html=True)
