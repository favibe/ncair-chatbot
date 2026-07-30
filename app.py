import streamlit as st
import html

from chatbot import get_chatbot_response
import memory

USER_ID = "user1"  # keep in sync with chatbot.py until a login system exists

st.set_page_config(page_title="NCAIR Chatbot", layout="centered")

st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: 0 auto;
    padding-top: 2rem;
}
.user-message, .bot-message {
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 75%;
    font-size: 16px;
    line-height: 1.5;
    word-wrap: break-word;
}
.user-message {
    background-color: #dcf8c6;
    text-align: right;
    float: right;
}
.bot-message {
    background-color: #f1f0f0;
    text-align: left;
    float: left;
}
.timestamp {
    font-size: 12px;
    color: #777;
    margin-top: 4px;
}
.clear {
    clear: both;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 NCAIR Chatbot")

# memory.json (via memory.py's per-user functions) is the single source of
# truth for chat state. Do NOT write to it directly from this file --
# chatbot.get_chatbot_response() handles all reads/writes so the schema
# stays consistent.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = memory.get_user_history(USER_ID)

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    memory.clear_user_history(USER_ID)

user_input = st.chat_input("Type your message...")
if user_input:
    result = get_chatbot_response(user_input, None)
    # get_chatbot_response already persisted this turn to memory.json --
    # just sync local session state to the returned, up-to-date history.
    st.session_state.chat_history = result["context"]

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for chat in st.session_state.chat_history:
    msg_class = "user-message" if chat["sender"] == "user" else "bot-message"
    # Escape user-controlled content before rendering as HTML to prevent
    # stored XSS (a message like "<img src=x onerror=alert(1)>" would
    # otherwise execute).
    safe_message = html.escape(chat["message"])
    timestamp = chat.get("timestamp", chat.get("time", ""))
    st.markdown(f"""
        <div class="{msg_class}">{safe_message}</div>
        <div class="timestamp">{timestamp}</div>
        <div class="clear"></div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)