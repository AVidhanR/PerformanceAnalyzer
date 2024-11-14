import streamlit as st # type: ignore

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
        st.Page("main.py", title="Performance Analyzer"),
    ],
    "About": [
        st.Page("about.py", title="About us"),
    ],
}).run()