import streamlit as st
import requests
import json

st.title("CSV Upload and Analysis")


# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Display a spinner while processing
    with st.spinner('Processing your file...'):
        # Create the files dictionary for the POST request
        files = {'file': uploaded_file}
        
        try:
            # Send the file to the backend
            response = requests.post('http://localhost:5000/upload', files=files)
            
            if response.status_code == 200:
                data = response.json()
                
                # Show success message with file details
                st.success('File uploaded successfully!')
                
                # Display file information in an organized way
                st.subheader('File Details')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Number of Rows', data['rows'])
                with col2:
                    st.metric('Number of Columns', len(data['columns']))
                
                # Display column names in an expandable section
                with st.expander('View Column Names'):
                    st.write(data['columns'])
                    
                # Add streaming chat interface for querying about columns
                st.subheader("Ask about the features")
                user_question = st.text_input("Enter your question about the data features:")
                
                if st.button("Ask Groq (Streaming)"):
                    if user_question:
                        # Create containers for response
                        status_container = st.empty()
                        response_container = st.empty()
                        
                        status_container.info("Sending request to Groq...")
                        
                        try:
                            # Send request to streaming endpoint
                            with requests.post(
                                'http://localhost:5000/stream_analysis',
                                json={
                                    'columns': data['columns'],
                                    'query': user_question
                                },
                                stream=True
                            ) as stream_response:
                                
                                if stream_response.status_code != 200:
                                    st.error(f"Error: {stream_response.text}")
                                else:
                                    # Initialize response display
                                    full_text = ""
                                    
                                    # Process stream
                                    for line in stream_response.iter_lines():
                                        if line:
                                            try:
                                                chunk = json.loads(line)
                                                
                                                # Handle different message types
                                                if chunk["type"] == "info" or chunk["type"] == "progress":
                                                    status_container.info(chunk["message"])
                                                
                                                elif chunk["type"] == "content":
                                                    # Append to the growing response
                                                    full_text += chunk["text"]
                                                    response_container.markdown(full_text)
                                                
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
                    else:
                        st.warning("Please enter a question first")
            else:
                # Display error message if upload failed
                st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")
                
        except requests.exceptions.ConnectionError:
            st.error('Failed to connect to the backend server. Please make sure it is running.')
        except Exception as e:
            st.error(f'An unexpected error occurred: {str(e)}')
else:
    # Display instructions when no file is uploaded
    st.info('Please upload a CSV file to begin analysis')
