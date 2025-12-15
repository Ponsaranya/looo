
import streamlit as st
import pandas as pd
import pickle
import sqlite3
import os
from datetime import datetime
# IMPORT MODULES
from utils.recommendation_engine import get_recommendation
from utils.faq_engine import get_faq_answer
# DATABASE SETUP
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(sender, message):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    local_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO chat (sender, message,timestamp) VALUES (?, ?,?)", (sender, message,local_time))
    conn.commit()
    conn.close()

def load_chat_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sender, message FROM chat ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()

# LOAD FAQ MODEL FILES 
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

df_faq = pd.read_excel("data/FAQ_dataset.xlsx")

# FAQ CHATBOT RESPONSE
def chatbot_faq(user_query):
    try:
        query_vec = vectorizer.transform([user_query])
        probs = model.predict_proba(query_vec)[0]
        max_prob = max(probs)

        if max_prob < 0.40:
            return None

        pred = model.predict(query_vec)[0]
        intent = label_encoder.inverse_transform([pred])[0]
        answers = df_faq[df_faq["intent"] == intent]["answer"]

        if answers.empty:
            return None

        return answers.sample(1).values[0]

    except Exception as e:
        print("FAQ Engine Error:", e)
        return None

# Load CSS helper 
def load_css_try(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            return True
    return False

load_css_try(["style.css", os.path.join("static", "style.css")])

# Page config
st.set_page_config(page_title="Banking Chatbot", layout="wide")
# Session: load history once
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
# submit handler (used for Enter and Send)
def submit_message():
    msg = st.session_state.get("chat_input", "")
    if not msg or not msg.strip():
        return

    # Keep exact logic: recommendation keywords first
    if ("suggest" in msg.lower()) or ("recommend" in msg.lower()):
        bot_reply = get_recommendation(msg)
    else:
        bot_reply = chatbot_faq(msg)
        if bot_reply is None:
            bot_reply = "Please ask something related to banking and Product suggestions."

    # Save to session and DB
    st.session_state.chat_history.append(("You", msg))
    st.session_state.chat_history.append(("Bot", bot_reply))

    save_to_db("You", msg)
    save_to_db("Bot", bot_reply)

    # Clear the input (safe inside callback)
    st.session_state["chat_input"] = ""

# Top header (sticky) + conversation + input row
# Header block (sticky)
st.markdown("<div class='page-header'>", unsafe_allow_html=True)
st.markdown("<h1 class='app-title'>Banking Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='app-desc'>"
    "Ask about banking and product suggestions"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Conversation card (single scrollable panel)
st.markdown('<div class="conversation-card">', unsafe_allow_html=True)
st.markdown("<div class='conv-title-row'><h2 class='conv-title'>Conversation</h2></div>", unsafe_allow_html=True)

# chat-window: single panel where all messages are appended (scroll here)
st.markdown('<div class="chat-window" id="chat_window">', unsafe_allow_html=True)

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(
            f"""
            <div class="msg-row user-row">
                <div class="bubble user-bubble">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="msg-row bot-row">
                <div class="bubble bot-bubble">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)  # close chat-window

# Input row under chat-window with Send (left) and Reset (right)
st.markdown('<div class="input-row">', unsafe_allow_html=True)
left_col, right_col = st.columns([4, 1], gap="small")

with left_col:
    st.text_input("", key="chat_input", placeholder="Type your message...", on_change=submit_message, label_visibility="collapsed")
    if st.button("Send"):
        submit_message()

with right_col:
    if st.button("Reset"):
        st.session_state.chat_history = []
        # conn = sqlite3.connect(DB_PATH)
        # cursor = conn.cursor()
        # cursor.execute("DELETE FROM chat")
        # conn.commit()
        # conn.close()
        st.rerun()
    st.markdown("<div class='reset-note'>If you want to reset the conversation, click this button.</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close input-row
st.markdown('</div>', unsafe_allow_html=True)  # close conversation-card
