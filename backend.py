import flask
from flask import request, jsonify, Response, stream_with_context
import pandas as pd #Use this if you have no GPU
from dotenv import load_dotenv
import time
import json
import os
import tempfile

# Load environment variables from .env file
load_dotenv()

# Add Groq integration
from groq import Groq  # You'll need to pip install groq

app = flask.Flask(__name__)

# Initialize Groq client - you'll need an API key
# Set this with: export GROQ_API_KEY=your_api_key
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def describe_csv(file_path=None, dataframe=None):
    """
    Generate statistical descriptions of a CSV file.
    Returns median, min, max, and interquartile ranges for each numeric column.
    
    Args:
        file_path: Path to CSV file
        dataframe: Pandas DataFrame (alternative to file_path)
    
    Returns:
        Dictionary with statistical descriptions
    """
    try:
        if dataframe is not None:
            df = dataframe
        elif file_path:
            df = pd.read_csv(file_path)
        else:
            return {"error": "No data provided"}
        
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"warning": "No numeric columns found in the dataset"}
        
        # Calculate statistics
        stats = {}
        for column in numeric_df.columns:
            col_stats = {
                "median": numeric_df[column].median(),
                "min": numeric_df[column].min(),
                "max": numeric_df[column].max(),
                "q1": numeric_df[column].quantile(0.25),
                "q3": numeric_df[column].quantile(0.75),
                "iqr": numeric_df[column].quantile(0.75) - numeric_df[column].quantile(0.25)
            }
            stats[column] = col_stats
            
        # Add general info
        result = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_df.columns),
            "column_names": df.columns.tolist(),
            "statistics": stats
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

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

@app.route('/describe_csv', methods=['POST'])
def api_describe_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save the file to a temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            file.save(temp_file.name)
            temp_file.close()
            
            # Get statistics
            stats = describe_csv(file_path=temp_file.name)
            
            # Clean up
            os.unlink(temp_file.name)
            
            return jsonify({
                'success': True,
                'statistics': stats
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File must be a CSV'}), 400


@app.route('/stream_analysis', methods=['POST'])
def stream_analysis():
    try:
        # Get request data
        data = request.json
        column_names = data.get('columns', [])
        user_query = data.get('query', '')
        file_path = data.get('file_path', '')
        
        if not column_names or not user_query:
            return jsonify({'error': 'Missing columns or query'}), 400
        
        def generate():
            # Initial message
            yield json.dumps({"type": "info", "message": "Starting analysis..."}) + "\n"
            
            # Get statistics if file path is provided
            stats_info = ""
            if file_path and os.path.exists(file_path):
                try:
                    stats = describe_csv(file_path=file_path)
                    stats_info = f"Here are some statistics about the dataset: {json.dumps(stats)}"
                    yield json.dumps({"type": "info", "message": "Retrieved dataset statistics"}) + "\n"
                except Exception as e:
                    stats_info = f"Failed to get statistics: {str(e)}"
                    yield json.dumps({"type": "warning", "message": stats_info}) + "\n"
            
            # Create prompt with feature names and statistics if available
            prompt = f"""
            The CSV contains the following features/columns:
            {', '.join(column_names)}
            
            {stats_info}
            
            User question: {user_query}
            
            Please answer the question based on the column names and statistics provided.
            """
            
            # Stream that we're sending to Groq
            yield json.dumps({"type": "progress", "message": "Sending to Groq..."}) + "\n"
            
            try:
                # Define the function for Groq to call
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_column_statistics",
                            "description": "Get detailed statistics for specific columns in the dataset",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of column names to get statistics for"
                                    }
                                },
                                "required": ["columns"]
                            }
                        }
                    }
                ]
                
                # Call Groq API with streaming
                stream = groq_client.chat.completions.create(
                    model="llama3-70b-8192",  # or another available model
                    messages=[
                        {"role": "system", "content": f"You analyze CSV features and answer questions about them. You have access to the current dataset:{data['file']}"},
                        {"role": "user", "content": prompt}
                    ],
                    tools=tools,
                    tool_choice="auto",
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
                    
                    # Handle tool calls
                    if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            if tool_call.function.name == "get_column_statistics":
                                try:
                                    # Parse the function arguments
                                    args = json.loads(tool_call.function.arguments)
                                    columns = args.get('columns', [])
                                    
                                    # Get statistics for the requested columns if file_path exists
                                    if file_path and os.path.exists(file_path):
                                        df = pd.read_csv(file_path)
                                        column_stats = {}
                                        for col in columns:
                                            if col in df.columns:
                                                if pd.api.types.is_numeric_dtype(df[col]):
                                                    column_stats[col] = {
                                                        "median": float(df[col].median()),
                                                        "min": float(df[col].min()),
                                                        "max": float(df[col].max()),
                                                        "mean": float(df[col].mean()),
                                                        "std": float(df[col].std())
                                                    }
                                                else:
                                                    # For non-numeric columns, provide value counts
                                                    column_stats[col] = {
                                                        "type": "categorical",
                                                        "unique_values": df[col].nunique(),
                                                        "most_common": df[col].value_counts().index[0] if not df[col].empty else None
                                                    }
                                        
                                        # Send statistics back as a message
                                        stat_message = f"Retrieved statistics for columns: {', '.join(columns)}"
                                        yield json.dumps({"type": "tool_result", "name": "get_column_statistics", "data": column_stats}) + "\n"
                                except Exception as e:
                                    yield json.dumps({"type": "error", "message": f"Error in tool call: {str(e)}"}) + "\n"
                
                # Final completion message
                yield json.dumps({"type": "complete", "message": "Analysis complete", "full_response": full_response}) + "\n"
                
            except Exception as e:
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
