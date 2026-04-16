import streamlit as st
from agent import ask

# Page config
st.set_page_config(page_title="HR Policy Assistant", layout="centered")

st.title("HR Policy Assistant Bot")
st.write("Ask any question related to company HR policies.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session"

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input box
user_input = st.chat_input("Type your question here...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Call agent
    try:
        result = ask(user_input, thread_id=st.session_state.thread_id)
        answer = result.get("answer", "No response generated.")
    except Exception as e:
        answer = f"Error: {str(e)}"

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })