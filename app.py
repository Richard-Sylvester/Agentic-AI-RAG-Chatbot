import streamlit as st
from graph import app  # Import the brain we just built

# Page Config
st.set_page_config(page_title="Agentic AI Chatbot", page_icon="🤖")
st.title("📖 Agentic AI Expert")
st.markdown("Ask me anything about the **Agentic AI eBook**! (I use RAG to find answers)")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ex: What are the benefits of Agentic AI?"):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching the book..."):
            
            # Run the LangGraph
            inputs = {"question": prompt}
            result = app.invoke(inputs)
            
            answer = result["answer"]
            context = result["context"]
            
            # Show Answer
            message_placeholder.markdown(answer)
            
            # Show Evidence (Optional but impressive for interviews)
            with st.expander("View Source Context (Debug)"):
                for i, doc in enumerate(context):
                    st.caption(f"**Source {i+1}:** {doc[:300]}...")

    # 3. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": answer})