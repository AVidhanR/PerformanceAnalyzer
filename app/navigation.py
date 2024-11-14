import streamlit as st # type: ignore

# This the required page for the navigation

st.set_page_config(page_icon=":bar_chart:")

st.markdown(
    """
    <style>
     body, p, h1, h2, h3, h4, h5, li, ul, ol{
        font-family: 'Cascadia Code';
     }
    </style>
    """,
    unsafe_allow_html=True,
)

st.navigation({
    "Home": [
        st.Page("pages/main.py", title="Performance Analyzer"),
    ],
    "Resources": [
        st.Page("pages/about.py", title="Learn About us"),
    ],
}).run()