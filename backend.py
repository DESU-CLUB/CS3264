import flask
from flask import request, jsonify, Response, stream_with_context
import pandas as pd #Use this if you have no GPU
from dotenv import load_dotenv
import time
import json
import os
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChatHistory():
    def __init__(self):
        self.history = [{"role": "system", "content": "You analyze CSV features and answer questions about them. You have access to the current dataset:"},
]
        self.system = {"role": "system", "content": "You analyze CSV features and answer questions about them. You have access to the current dataset:"},


    def add_user_message(self, message):
        self.history.append(message)
        #Have groq summarize the conversation in 1 message
        # If history is getting long, have Groq summarize
        if len(self.history) > 8:
            try:
                # Create summarization prompt
                summary_prompt = {
                    "role": "user", 
                    "content": "Please summarize our conversation so far in one concise message that captures the key points and context. Focus on the data analysis aspects."
                }
                
                # Get summary from Groq
                summary_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[self.system, summary_prompt],
                    max_tokens=200
                )
                
                # Replace history with system message + summary + last 2 messages
                summary_msg = {"role": "assistant", "content": summary_response.choices[0].message.content}
                self.history = [self.system, summary_msg] + self.history[-2:]
                
            except Exception as e:
                logger.error(f"Error summarizing chat history: {str(e)}")
                # On error, just truncate to last few messages
                self.history = [self.system] + self.history[-4:]
        if len(self.history) > 10:
            self.history = self.history[0] + self.history[2:]

    def get_history(self):
        return self.history
    


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
        stats_lines = []
        for column in numeric_df.columns:
            median = numeric_df[column].median()
            min_val = numeric_df[column].min()
            max_val = numeric_df[column].max()
            q1 = numeric_df[column].quantile(0.25)
            q3 = numeric_df[column].quantile(0.75)
            iqr = q3 - q1
            
            col_stats = f"Column: {column}\n"
            col_stats += f"  Median: {median}\n"
            col_stats += f"  Min: {min_val}\n" 
            col_stats += f"  Max: {max_val}\n"
            col_stats += f"  Q1: {q1}\n"
            col_stats += f"  Q3: {q3}\n"
            col_stats += f"  IQR: {iqr}\n"
            stats_lines.append(col_stats)
            
        # Build full result string
        result = f"Dataset Summary:\n"
        result += f"Total Rows: {len(df)}\n"
        result += f"Total Columns: {len(df.columns)}\n" 
        result += f"Numeric Columns: {len(numeric_df.columns)}\n"
        result += f"Column Names: {', '.join(df.columns.tolist())}\n\n"
        result += "Statistics:\n"
        result += "\n".join(stats_lines)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

def get_csv_sample(file_path, sample_size=100):
    """
    Read a sample of rows from a CSV file.
    
    Args:
        file_path: Path to CSV file
        sample_size: Number of rows to sample
        
    Returns:
        Dictionary with sample data and column info
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get a sample of rows
        if len(df) > sample_size:
            sample_df = df.sample(sample_size)
        else:
            sample_df = df
            
        # Convert to formatted string representation
        # Convert sample to string table format
        table_rows = []
        
        # Add header row
        header = "|" + "|".join(f" {col} " for col in sample_df.columns) + "|"
        separator = "|" + "|".join("-" * (len(col) + 2) for col in sample_df.columns) + "|"
        
        table_rows.append(header)
        table_rows.append(separator)
        
        # Add data rows
        for _, row in sample_df.iterrows():
            row_str = "|" + "|".join(f" {str(val)} " for val in row) + "|"
            table_rows.append(row_str)
            
        table_str = "\n".join(table_rows)

        return table_str 
       
    except Exception as e:
        logger.error(f"Error getting CSV sample: {str(e)}")
        return "Error"

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


@app.route('/stream_analysis', methods=['POST'])
def stream_analysis():
    try:
        logger.info("Starting stream_analysis function")
        # Get request data
        data = request.json
        column_names = data.get('columns', [])
        user_query = data.get('query', '')
        file_path = data.get('file_path', '')
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'datasets', file_path)
        
        logger.info(f"Request data: columns={len(column_names)}, query={user_query[:50]}..., file_path={file_path}")
        
        if not column_names or not user_query:
            logger.error("Missing columns or query in request")
            return jsonify({'error': 'Missing columns or query'}), 400
        
        def generate():
            try:
                logger.info("Starting generate function in stream_analysis")
                # Initial message
                initial_message = json.dumps({"type": "info", "message": "Starting analysis..."}) + "\n"
                logger.debug(f"Yielding initial message: {initial_message}")
                yield initial_message
                
                # Get statistics if file path is provided
                stats_info = ""
                sample_data_info = ""
                
                if file_path and os.path.exists(file_path):
                    logger.info(f"File exists, getting data from: {file_path}")
                    try:
                        # Get statistics
                        logger.debug("Getting statistics from file")
                        stats = describe_csv(file_path=file_path)
                        stats_info = f"Here are some statistics about the dataset: {json.dumps(stats)}"
                        stats_message = json.dumps({"type": "info", "message": "Retrieved dataset statistics"}) + "\n"
                        logger.debug("Yielding statistics message")
                        yield stats_message
                        
                        # Get sample data
                        logger.debug("Getting sample data from file")
                        sample_data = get_csv_sample(file_path, sample_size=100)
                        sample_data_info = f"Here is a sample of the data (up to 100 rows): f{sample_data}"
                        sample_message = json.dumps({"type": "info", "message": f"Retrieved sample rows"}) + "\n"
                        logger.debug("Yielding sample data message")
                        yield sample_message
                        
                    except Exception as e:
                        error_msg = f"Failed to get data: {str(e)}"
                        logger.error(f"Error getting data: {error_msg}", exc_info=True)
                        stats_info = error_msg
                        warning_message = json.dumps({"type": "warning", "message": error_msg}) + "\n"
                        yield warning_message
                else:
                    logger.warning(f"File path not provided or file doesn't exist: {file_path}")
                
                # Create prompt with feature names, statistics, and sample data if available
                prompt = f"""
                The CSV contains the following features/columns:
                {', '.join(column_names)}
                
                {stats_info}
                
                {sample_data_info}
                
                User question: {user_query}
                
                Please answer the question based on the column names, statistics, and sample data provided.
                """
                
                # Stream that we're sending to Groq
                progress_message = json.dumps({"type": "progress", "message": "Sending to Groq..."}) + "\n"
                logger.info("Sending request to Groq")
                logger.debug(f"Yielding progress message: {progress_message}")
                yield progress_message
                
                try:
                    # Define the function for Groq to call
          
                    
                    logger.debug("Creating Groq streaming request")
                    # Call Groq API with streaming

                    chat_history.add_user_message({"role": "user", "content": prompt})

                    stream = groq_client.chat.completions.create(
                        model="llama3-70b-8192",  # or another available model
                        messages=chat_history.get_history(),
                        max_tokens=500,
                        stream=True  # Enable streaming
                    )
                    
                    logger.info("Groq stream created, processing chunks")
                    # Stream the response chunks
                    full_response = ""
                    chunk_count = 0
                    for chunk in stream:
                        chunk_count += 1
                        logger.debug(f"Processing chunk {chunk_count}")
                        
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            content_message = json.dumps({"type": "content", "text": content}) + "\n"
                            logger.debug(f"Yielding content chunk: {content[:50]}...")
                            yield content_message
                        
                        # Handle tool calls
                        if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            logger.info("Tool call detected in response")
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                if tool_call.function.name == "get_column_statistics":
                                    try:
                                        # Parse the function arguments
                                        logger.debug("Parsing tool call arguments")
                                        args = json.loads(tool_call.function.arguments)
                                        columns = args.get('columns', [])
                                        logger.info(f"Tool call for columns: {columns}")
                                        
                                        # Get statistics for the requested columns if file_path exists
                                        if file_path and os.path.exists(file_path):
                                            logger.debug("Getting statistics for requested columns")
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
                                            tool_result = json.dumps({"type": "tool_result", "name": "get_column_statistics", "data": column_stats}) + "\n"
                                            logger.info(f"Yielding tool result: {stat_message}")
                                            yield tool_result
                                    except Exception as e:
                                        error_message = f"Error in tool call: {str(e)}"
                                        logger.error(error_message, exc_info=True)
                                        yield json.dumps({"type": "error", "message": error_message}) + "\n"
                    
                    logger.info(f"Processed {chunk_count} chunks from Groq")
                    # Final completion message
                    complete_message = json.dumps({"type": "complete", "message": "Analysis complete", "full_response": full_response}) + "\n"
                    logger.info("Analysis complete, yielding final message")
                    yield complete_message
                    
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Error in Groq request: {error_message}", exc_info=True)
                    yield json.dumps({"type": "error", "message": error_message}) + "\n"
            
            except Exception as e:
                error_message = f"Error in generate function: {str(e)}"
                logger.error(error_message, exc_info=True)
                yield json.dumps({"type": "error", "message": error_message}) + "\n"
        
        logger.info("Setting up streaming response")
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in stream_analysis: {error_message}", exc_info=True)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    chat_history = ChatHistory()
    app.run()
