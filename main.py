import streamlit as st

# --- PAGE SETUP ---
about_page = st.Page(
    page="views/chatbot.py",
    title="Chatbot",
    icon="💬",
    default=True,
)

# --- NAVIGATION ---
pg = st.navigation(
    {
        "Info" : [about_page],
    }
)

# --- RUN NAVIGATION ---
pg.run()

# --- SIDEBAR ---
st.sidebar.text("Made by @estebanscabrera")
