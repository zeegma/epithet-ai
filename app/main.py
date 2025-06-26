import streamlit as st
from ui import landing_page, quiz_form, result_display

def main():
    st.set_page_config(
        page_title="Epithet Personality Game",
        page_icon="🎮",
        layout="wide"
    )

    if 'page' not in st.session_state:
        st.session_state.page = 'landing'

    query_params = st.query_params
    if 'page' in query_params:
        st.session_state.page = query_params['page'][0]

    if st.session_state.page == 'landing':
        landing_page.show()
    elif st.session_state.page == 'quiz':
        quiz_form.show()
    elif st.session_state.page == 'results':
        result_display.show()

if __name__ == "__main__":
    main()
