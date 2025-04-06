import flask
from flask import request, jsonify, Response, stream_with_context
import pandas as pd
from dotenv import load_dotenv
import time
import json
import os
import tempfile
import logging
import threading
import shutil
from rag import build_persisted_index, load_persisted_index, query_index
from generate_data import generate_data

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

class DataGenerationStatus:
    def __init__(self):
        self.is_generating = False
        self.progress = 0
        self.total_rows = 0
        self.generated_data = None
        self.error = None
        self.current_file = None

    def reset(self):
        self.is_generating = False
        self.progress = 0
        self.total_rows = 0
        self.generated_data = None
        self.error = None
        # Keep the current file reference

    def start_generation(self, total_rows, file_path):
        self.reset()
        self.is_generating = True
        self.total_rows = total_rows
        self.current_file = file_path

    def update_progress(self, current_row):
        self.progress = (current_row / self.total_rows) * 100

    def complete_generation(self, data):
        self.is_generating = False
        self.progress = 100
        self.generated_data = data

    def set_error(self, error):
        self.is_generating = False
        self.error = str(error)

# Load environment variables from .env file
load_dotenv()

app = flask.Flask(__name__)

# Initialize global objects
data_status = DataGenerationStatus()
query_engine = None

def create_features_dir():
    """Create features directory for feature documents"""
    features_dir = "./data/features/"
    os.makedirs(features_dir, exist_ok=True)
    return features_dir

def prepare_feature_documents(df, features_dir):
    """
    Prepare feature documents for LlamaIndex ingestion
    Each feature gets its own text file with descriptions
    """
    logger.info(f"Preparing feature documents in {features_dir}")
    
    # Get column information
    columns = df.columns.tolist()
    
    # Clear existing feature documents
    for file in os.listdir(features_dir):
        if file.endswith(".txt"):
            os.remove(os.path.join(features_dir, file))
    
    # Generate a document for each feature
    for column in columns:
        file_path = os.path.join(features_dir, f"{column}.txt")
        
        # Get column statistics
        if pd.api.types.is_numeric_dtype(df[column]):
            stats = {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "mean": float(df[column].mean()),
                "median": float(df[column].median()),
                "std": float(df[column].std())
            }
            
            # Write stats to file
            with open(file_path, "w") as f:
                f.write(f"# Feature: {column}\n\n")
                f.write("## Type: Numeric\n\n")
                f.write("## Description\n")
                f.write(f"This is a numeric feature in the dataset.\n\n")
                f.write("## Statistics\n")
                f.write(f"- Minimum value: {stats['min']}\n")
                f.write(f"- Maximum value: {stats['max']}\n")
                f.write(f"- Mean: {stats['mean']}\n")
                f.write(f"- Median: {stats['median']}\n")
                f.write(f"- Standard deviation: {stats['std']}\n\n")
                
                # Add correlation information
                f.write("## Correlations with other features\n")
                for other_col in columns:
                    if other_col != column and pd.api.types.is_numeric_dtype(df[other_col]):
                        corr = df[column].corr(df[other_col])
                        f.write(f"- Correlation with {other_col}: {corr:.4f}\n")
        
        else:
            # For categorical columns
            value_counts = df[column].value_counts()
            unique_count = len(value_counts)
            top_values = value_counts.head(10)
            
            with open(file_path, "w") as f:
                f.write(f"# Feature: {column}\n\n")
                f.write("## Type: Categorical\n\n")
                f.write("## Description\n")
                f.write(f"This is a categorical feature in the dataset.\n\n")
                f.write("## Statistics\n")
                f.write(f"- Unique values: {unique_count}\n")
                f.write(f"- Missing values: {df[column].isna().sum()}\n\n")
                f.write("## Most common values\n")
                
                for val, count in top_values.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"- {val}: {count} occurrences ({percentage:.2f}%)\n")
    
    logger.info(f"Created {len(columns)} feature documents")

def cleanup_previous_data():
    """Clean up previous data and RAG indices"""
    try:
        # Clear the persist directory
        persist_dir = "./data/chroma_db"
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        
        # Clear any generated data
        generated_dir = "./data/generated"
        if os.path.exists(generated_dir):
            shutil.rmtree(generated_dir)
            
        # Recreate directories
        os.makedirs(persist_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        
        logger.info("Cleaned up previous data")
    except Exception as e:
        logger.error(f"Error cleaning up data: {str(e)}")

def generate_data_background(file_path, n_samples=1000):
    """Background task for data generation"""
    try:
        global data_status
        data_status.start_generation(n_samples, file_path)
        
        # Add progress callback
        def progress_callback(current_row):
            data_status.update_progress(current_row)
        
        # Generate the data
        generated_df = generate_data(
            csv_path=file_path,
            n_samples=n_samples,
            persist_dir="./data/chroma_db",
            features_dir="./data/features/",
            collection_name="dquery",
            output_path="./data/generated/generated_data.csv",
            max_workers=5,
            batch_size=100
        )
        
        data_status.complete_generation(generated_df)
        logger.info(f"Generated {n_samples} rows of data successfully")
        
    except Exception as e:
        logger.error(f"Error in data generation: {str(e)}")
        data_status.set_error(e)

def describe_csv(file_path=None, dataframe=None):
    """Generate statistical descriptions of a CSV file"""
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

@app.route('/upload', methods=['POST'])
def upload_file():
    global query_engine
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save the file temporarily
            temp_dir = "./datasets"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            
            # Clean up previous data
            cleanup_previous_data()
            
            # Read basic info about the file
            df = pd.read_csv(file_path)
            
            # Create features directory and prepare feature documents
            features_dir = create_features_dir()
            prepare_feature_documents(df, features_dir)
            
            # Build RAG index
            logger.info("Building RAG index...")
            query_engine = build_persisted_index(
                features_dir="./data/features/",
                persist_dir="./data/chroma_db",
                collection_name="dquery"
            )
            
            # Start data generation in background
            thread = threading.Thread(
                target=generate_data_background,
                args=(file_path,),
                kwargs={'n_samples': 1000}
            )
            thread.start()
            
            return jsonify({
                'success': True,
                'rows': len(df),
                'columns': df.columns.tolist(),
                'message': 'File uploaded and processing started'
            })
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File must be a CSV'}), 400

@app.route('/generation_status', methods=['GET'])
def get_generation_status():
    """Endpoint to check data generation status"""
    return jsonify({
        'is_generating': data_status.is_generating,
        'progress': data_status.progress,
        'total_rows': data_status.total_rows,
        'current_file': os.path.basename(data_status.current_file) if data_status.current_file else None,
        'error': data_status.error
    })

@app.route('/stream_analysis', methods=['POST'])
def stream_analysis():
    global query_engine
    
    try:
        logger.info("Starting stream_analysis function")
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'Missing query'}), 400
        
        if not query_engine:
            return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
        
        def generate():
            try:
                # Initial message
                yield json.dumps({"type": "info", "message": "Starting analysis..."}) + "\n"
                
                # Use LlamaIndex QueryEngine directly
                try:
                    # Get the response from the query engine
                    response = query_index(user_query, query_engine)
                    
                    # Stream the response in chunks
                    full_response = str(response)
                    # Simulate streaming by chunking the response
                    chunk_size = 20  # Adjust based on your preference
                    for i in range(0, len(full_response), chunk_size):
                        chunk = full_response[i:i + chunk_size]
                        yield json.dumps({"type": "content", "text": chunk}) + "\n"
                        time.sleep(0.01)  # Small delay for streaming effect
                    
                except Exception as e:
                    error_msg = f"Error querying index: {str(e)}"
                    logger.error(error_msg)
                    yield json.dumps({"type": "error", "message": error_msg}) + "\n"
                
                # Complete message
                yield json.dumps({"type": "complete", "message": "Analysis complete"}) + "\n"
                
            except Exception as e:
                error_message = f"Error in generate function: {str(e)}"
                logger.error(error_message, exc_info=True)
                yield json.dumps({"type": "error", "message": error_message}) + "\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in stream_analysis: {error_message}", exc_info=True)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)
