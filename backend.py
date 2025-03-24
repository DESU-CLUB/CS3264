import flask
from flask import request, jsonify, Response, stream_with_context
import pandas as pd #Use this if you have no GPU
from dotenv import load_dotenv
import time
import json

# Load environment variables from .env file
load_dotenv()

# Add Groq integration
import os
from groq import Groq  # You'll need to pip install groq

app = flask.Flask(__name__)

# Initialize Groq client - you'll need an API key
# Set this with: export GROQ_API_KEY=your_api_key
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        # Read the CSV using cudf
        try:
            df = pd.read_csv(file)
            # Return basic info about the dataframe
            return jsonify({
                'success': True,
                'rows': len(df),
                'columns': df.columns.tolist()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File must be a CSV'}), 400

@app.route('/chat', methods=['POST'])
def chat_about_features():
    data = request.json
    prompt = f"Analyze the following CSV file and provide a summary of the features and their relationships: {data['file']}"
    
    # Use Groq to generate a response
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[{"role": "user", "content": prompt}]
    )
    
    return jsonify({
        'success': True,
        'response': response.choices[0].message.content
    })

@app.route('/stream_analysis', methods=['POST'])
def stream_analysis():
    try:
        # Get request data
        data = request.json
        column_names = data.get('columns', [])
        user_query = data.get('query', '')
        
        if not column_names or not user_query:
            return jsonify({'error': 'Missing columns or query'}), 400
        
        def generate():
            # Initial message
            yield json.dumps({"type": "info", "message": "Starting analysis..."}) + "\n"
            
            # Create prompt with just the feature names
            prompt = f"""
            The CSV contains the following features/columns:
            {', '.join(column_names)}
            
            User question: {user_query}
            
            Please answer the question based only on the column names provided.
            """
            
            # Stream that we're sending to Groq
            yield json.dumps({"type": "progress", "message": "Sending to Groq..."}) + "\n"
            
            try:
                # Call Groq API with streaming
                stream = groq_client.chat.completions.create(
                    model="llama3-70b-8192",  # or another available model
                    messages=[
                        {"role": "system", "content": "You analyze CSV features and answer questions about them."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    stream=True  # Enable streaming
                )
                
                # Stream the response chunks
                full_response = ""
                for chunk in stream:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield json.dumps({"type": "content", "text": content}) + "\n"
                
                # Final completion message
                yield json.dumps({"type": "complete", "message": "Analysis complete", "full_response": full_response}) + "\n"
                
            except Exception as e:
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
