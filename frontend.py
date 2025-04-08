import streamlit as st
import requests
import json
import pandas as pd
import time
import threading

st.title("CSV Upload and Analysis with LlamaIndex")

# ---------------------------------------------------------------------
# Session State Setup
# ---------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "data" not in st.session_state:
    st.session_state.data = None
if "generation_started" not in st.session_state:
    st.session_state.generation_started = False
if "polling_thread" not in st.session_state:
    st.session_state.polling_thread = None
if "stop_polling" not in st.session_state:
    st.session_state.stop_polling = False

# ---------------------------------------------------------------------
# Sidebar for Data Generation Progress
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Data Generation Progress")
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    file_placeholder = st.empty()
    
    def poll_status():
        while not st.session_state.stop_polling:
            try:
                response = requests.get("http://localhost:5000/generation_status")
                if response.status_code == 200:
                    status = response.json()
                    
                    # Update session state with the status
                    st.session_state.generation_status = status
                    
                    # Don't need to update UI here as that happens in the main thread
                    if status["progress"] == 100 or status["error"]:
                        break
                        
                time.sleep(1)  # Poll every second
            except:
                time.sleep(2)  # If error, wait a bit longer before retrying
    
    # Show current status if we're polling
    if st.session_state.generation_started:
        if "generation_status" in st.session_state:
            status = st.session_state.generation_status
            
            if status["error"]:
                progress_placeholder.empty()
                status_placeholder.error(f"Error: {status['error']}")
            elif status["is_generating"]:
                progress_placeholder.progress(status["progress"] / 100)
                status_placeholder.info(f"Generating data: {status['progress']:.1f}% complete")
                if status["current_file"]:
                    file_placeholder.info(f"File: {status['current_file']}")
            elif status["progress"] == 100:
                progress_placeholder.progress(1.0)
                status_placeholder.success("Data generation complete!")
                if status["current_file"]:
                    file_placeholder.info(f"File: {status['current_file']}")
            else:
                status_placeholder.info("Waiting to start...")
                
        # Start polling thread if needed
        if st.session_state.polling_thread is None or not st.session_state.polling_thread.is_alive():
            st.session_state.stop_polling = False
            st.session_state.polling_thread = threading.Thread(target=poll_status)
            st.session_state.polling_thread.daemon = True
            st.session_state.polling_thread.start()

# ---------------------------------------------------------------------
# CSV Upload Section
# ---------------------------------------------------------------------
st.header("Step 1: Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Display the CSV in a scrollable table
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head(10), height=300)
        
        # Show statistics
        st.subheader("Data Statistics")
        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Show column datatypes
        st.expander("Column Details").write(pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isna().sum()
        }))
        
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

    # Process button
    if st.button("Process File and Generate Data"):
        # Reset file pointer before sending it to the backend
        uploaded_file.seek(0)
        with st.spinner("Processing your file..."):
            files = {"file": uploaded_file}
            try:
                response = requests.post("http://localhost:5000/upload", files=files)
                if response.status_code == 200:
                    # Store the response data in session state for later use
                    st.session_state.data = response.json()
                    st.success(st.session_state.data.get("message", "File uploaded successfully!"))
                    st.session_state.generation_started = True
                    
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
    st.header("Step 2: Ask Questions About Your Data")
    st.info("Ask about feature relationships, statistics, or potential insights")

    # Display the chat conversation (if any) using st.chat_message
    for msg in st.session_state.chat_history:
        st.chat_message(msg["sender"]).write(msg["message"])

    # Use st.chat_input for the new query
    prompt = st.chat_input("Ask a question about the dataset:")

    if prompt:
        # Add the user's prompt to the chat history and display it
        st.session_state.chat_history.append({"sender": "user", "message": prompt})
        st.chat_message("user").write(prompt)

        # Show a status message while analyzing
        status_container = st.empty()
        status_container.info("Analyzing your question using LlamaIndex...")

        full_text = ""
        try:
            # Call the streaming endpoint with the user query
            with requests.post(
                "http://127.0.0.1:5000/stream_analysis",
                json={
                    "query": prompt,
                },
                stream=True,
            ) as stream_response:
                if stream_response.status_code != 200:
                    status_container.error(f"Error: {stream_response.text}")
                else:
                    response_container = st.empty()
                    
                    for line in stream_response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if chunk["type"] in ["info", "progress"]:
                                    status_container.info(chunk["message"])
                                elif chunk["type"] == "content":
                                    # Accumulate the full response text
                                    full_text += chunk["text"]
                                    # Update the visible response in real-time
                                    response_container.markdown(full_text)
                                elif chunk["type"] == "complete":
                                    status_container.success(chunk["message"])
                                elif chunk["type"] == "error":
                                    status_container.error(chunk["message"])
                            except json.JSONDecodeError:
                                st.warning(f"Received invalid JSON: {line}")
        except requests.exceptions.ConnectionError:
            status_container.error("Failed to connect to the backend server.")
        except Exception as e:
            status_container.error(f"An unexpected error occurred: {str(e)}")

        # After the full response is received, add it to chat history
        if full_text:
            st.session_state.chat_history.append({"sender": "assistant", "message": full_text})

        # Limit the history to last 10 messages
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]
else:
    st.info("Please upload a CSV file to begin analysis")
    
    # Examples section
    st.subheader("How it works")
    st.write("""
    1. Upload a CSV file to analyze its features
    2. The system will index your data using LlamaIndex 
    3. Ask questions about your data features and relationships
    4. In parallel, synthetic data will be generated based on your file
    """)
    
# Clean up when the app is closed
def on_close():
    st.session_state.stop_polling = True
    
# Register the cleanup
st.on_script_run.append(on_close)
