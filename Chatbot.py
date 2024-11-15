import streamlit as st
import google.generativeai as genai
from google.generativeai import protos
import json
import pandas as pd

# Initialize messages in session state
if "messages" not in st.session_state:
    # Create the initial assistant message with Part instances
    initial_message = protos.Content(
        parts=[protos.Part(text="How can I help you?")],
        role="model"
    )
    st.session_state["messages"] = [initial_message]

# Sidebar options with download buttons
with st.sidebar:
    st.markdown("## Gemini API")
    st.markdown("""
        **Welcome to the Gemini API Chat Interface!**

        This app allows you to various Gemini language models.
        Use the options below to configure model settings, manage chat history, and download your chat.

        *Happy chatting!*
    """)
    
    user_api_key = st.text_input("Enter your Gemini API key",type="password")
    if user_api_key:
        genai.configure(api_key=user_api_key)
    else:
        st.error("Please enter your API key")
        # st.info("✅ API key is provided by developer")
        st.markdown("You can create your Gemini API Key [here](https://aistudio.google.com/app/apikey)")
    
    tool = st.toggle("Code Execution Tool",False)
    if tool:
        st.write("Code Execution Tool is now active. it enables the model to generate and run Python code")
        tools = "code_execution"
    else:
        tools = None
        
    model_name = st.selectbox("Choose a gemini Model",("gemini-1.5-flash","gemini-1.5-pro-002","gemini-1.5-pro","gemini-1.5-flash-002","gemini-1.5-flash-8b"))
    with st.expander("Output Control Parameter"):
        temp = st.slider("🌡Temperature",0.0,2.0,1.0,0.05)
        top_p = st.slider("Top P",0.0,1.0,.95,.05)
        top_k = st.slider("Top K",0,100,30)
        max_length = st.number_input("Max Output Length",1,8192,4096)
        stop_sequence_input = st.text_input("Stop Sequence", placeholder="e.g., Thank you!")
        stop_sequence = stop_sequence_input.split(",") if stop_sequence_input else []

    # Download Chat
    with st.expander("Download Chat"):
        if st.session_state["messages"]:
            # JSON download button
            
            def content_to_dict(content_obj):
                if isinstance(content_obj, dict):
                    return content_obj  
                return {
                    "role": content_obj.role,
                    "parts": [part.text for part in content_obj.parts],
                }

            json_data = json.dumps([content_to_dict(msg) for msg in st.session_state["messages"]], indent=4)

            # json_data = json.dumps(st.session_state["messages"], indent=4)
            st.download_button("As JSON Format", json_data, file_name="chat_history.json", mime="application/json")
            
            # CSV download button
            df = pd.DataFrame(st.session_state["messages"])
            csv_data = df.to_csv(index=False)
            st.download_button("As CSV Format", csv_data, file_name="chat_history.csv", mime="text/csv")

    if st.button("Clear Chat"):
        st.session_state["messages"] = [protos.Content(parts=[protos.Part(text="How can I help you?")], role="model")]
        st.rerun() 
        
# Set up generation configuration
generation_config = {
  "temperature": temp,
  "top_p": top_p,
  "top_k": top_k,
  "max_output_tokens": max_length,
  "stop_sequences": stop_sequence,
  "response_mime_type": "text/plain",
}

# Initialize chat in session state 
try:
    model = genai.GenerativeModel(model_name, generation_config=generation_config, tools=tools)
    st.session_state.chat = model.start_chat(history=st.session_state["messages"])
except Exception as e:
    st.error(f"Failed to initialize chat: {e}")

st.title("AI Chatbot")
st.caption("A Chatbot powered by Gemini")

# Display chat messages
for msg in st.session_state.messages:
    # st.chat_message(msg["role"]).write(msg["parts"])
    role = "user" if msg.role == "user" else "assistant"
    st.chat_message(role).write("".join([part.text for part in msg.parts]))
    
if prompt := st.chat_input():
    # Append user input as a Content object with Part instance
    if prompt and user_api_key:
        user_message = protos.Content(
            parts=[protos.Part(text=prompt)],
            role="user"
        )
        st.session_state["messages"].append(user_message)
        st.chat_message("user").write(prompt)
        
        if "chat" in st.session_state:
            # Concatenate messages into a single string
            full_prompt = "\n".join([
                "".join([part.text for part in msg.parts])
                for msg in st.session_state.messages
            ])
            full_prompt += "\n" + prompt
            
            # Send the user message and get a response
            response = st.session_state.chat.send_message(full_prompt)
            
            # Create the assistant's response as a Content object with Part instance
            model_message = protos.Content(
                parts=[protos.Part(text=response.text)],
                role="model"
            )
            st.session_state["messages"].append(model_message)
            st.chat_message("assistant").write(response.text)
        else:
            st.error("Failed to start chat message")
    else:
        st.error("Please provide a API key in left sidebar")