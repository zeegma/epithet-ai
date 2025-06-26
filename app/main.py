import streamlit as st
from ui import landing_page, quiz_form, result_display

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Epithet Personality Game",
        page_icon="🎮",
        layout="wide"
    )

    # Initialize navigation state if it doesn't exist yet
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'

    # Check query parameters from the URL to determine current page
    query_params = st.query_params
    if 'page' in query_params:
        st.session_state.page = query_params['page'][0]

    # Render the appropriate UI based on current page state
    if st.session_state.page == 'landing':
        landing_page.show()
    elif st.session_state.page == 'quiz':
        quiz_form.show()
    elif st.session_state.page == 'results':
        result_display.show()

if __name__ == "__main__":
    main()
