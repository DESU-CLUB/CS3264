import streamlit as st
import requests

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
