import streamlit as st
from ui import landing_page, quiz_form, result_display

def main():
    st.set_page_config(
        page_title="Epithet Personality Game",
        page_icon="🎮",
        layout="wide"
    )

    # Use new st.query_params API
    page = st.query_params.get("page", "landing")

    # Save to session_state if needed
    st.session_state.page = page

    # Render based on the page
    if page == 'landing':
        landing_page.show()
    elif page == 'quiz':
        
        quiz_form.show()
    elif page == 'results':
        result_display.show()
    else:
        st.error("Page not found.")

if __name__ == "__main__":
    main()
