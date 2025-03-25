import streamlit as st
import requests
import json
import pandas as pd

st.title("CSV Upload and Analysis")

# ---------------------------------------------------------------------
# Session State Setup
# ---------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "data" not in st.session_state:
    st.session_state.data = None

# ---------------------------------------------------------------------
# CSV Upload Section
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Display the CSV in a scrollable table
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded CSV Data")
        st.dataframe(df, height=300)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

    # Reset file pointer before sending it to the backend
    uploaded_file.seek(0)
    with st.spinner("Processing your file..."):
        files = {"file": uploaded_file}
        try:
            response = requests.post("http://localhost:5000/upload", files=files)
            if response.status_code == 200:
                # Store the response data in session state for later use.
                st.session_state.data = response.json()
                st.success("File uploaded successfully!")
                st.subheader("File Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of Rows", st.session_state.data.get("rows", "N/A"))
                with col2:
                    st.metric("Number of Columns", len(st.session_state.data.get("columns", [])))
                with st.expander("View Column Names"):
                    st.write(st.session_state.data.get("columns", []))
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend server. Please make sure it is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # ---------------------------------------------------------------------
    # Chat Interface Section (Only if CSV is uploaded)
    # ---------------------------------------------------------------------
    st.subheader("Ask about the features")

    # Display the chat conversation (if any) using st.chat_message
    for msg in st.session_state.chat_history:
        st.chat_message(msg["sender"]).write(msg["message"])

    # Use st.chat_input for the new query
    prompt = st.chat_input("Enter your question about the data features:")

    if prompt:
        # Add the user's prompt to the chat history and display it
        st.session_state.chat_history.append({"sender": "user", "message": prompt})
        st.chat_message("user").write(prompt)

        # Show a status message while contacting Groq
        status_container = st.empty()
        status_container.info("Sending request to Groq...")

        full_text = ""
        try:
            # Call the streaming endpoint with CSV columns and the user query
            with requests.post(
                "http://localhost:5000/stream_analysis",
                json={
                    "columns": st.session_state.data.get("columns", []),
                    "query": prompt,
                },
                stream=True,
            ) as stream_response:
                if stream_response.status_code != 200:
                    st.error(f"Error: {stream_response.text}")
                else:
                    for line in stream_response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if chunk["type"] in ["info", "progress"]:
                                    status_container.info(chunk["message"])
                                elif chunk["type"] == "content":
                                    # Accumulate the full response text
                                    full_text += chunk["text"]
                                elif chunk["type"] == "complete":
                                    status_container.success(chunk["message"])
                                elif chunk["type"] == "error":
                                    status_container.error(chunk["message"])
                            except json.JSONDecodeError:
                                st.warning(f"Received invalid JSON: {line}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend server.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

        # After the full response is received, add it to chat history and display it
        st.session_state.chat_history.append({"sender": "assistant", "message": full_text})
        st.chat_message("assistant").write(full_text)

        # Optionally, limit the history to last 5 user-assistant pairs (10 messages)
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]
else:
    st.info("Please upload a CSV file to begin analysis")
