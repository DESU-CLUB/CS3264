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
import numpy as np
from flask_cors import CORS
from rag import build_persisted_index, load_persisted_index, query_index
from generate_data import generate_data, LLMConfig

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

# Helper function to convert numpy types to Python native types for JSON serialization
def convert_np_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(item) for item in obj]
    return obj

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DataGenerationStatus:
    def __init__(self):
        self.is_generating = False
        self.progress = 0
        self.total_rows = 0
        self.generated_data = None
        self.error = None
        self.current_file = None
        self.result_path = None

    def reset(self):
        self.is_generating = False
        self.progress = 0
        self.total_rows = 0
        self.generated_data = None
        self.error = None
        self.result_path = None
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

    def set_generating(self, is_generating):
        self.is_generating = is_generating

    def set_progress(self, progress):
        self.progress = progress

    def set_result_path(self, result_path):
        self.result_path = result_path

# Load environment variables from .env file
load_dotenv()

app = flask.Flask(__name__)
# Enable CORS for all routes with proper configuration
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize global objects
data_status = DataGenerationStatus()
query_engine = None
current_csv_file = None  # Global variable to track current CSV file

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

def generate_data_background(file_path, n_samples=1000, **llm_params):
    """Background task for data generation"""
    try:
        global data_status
        data_status.start_generation(n_samples, file_path)
        
        # Add progress callback
        def progress_callback(current_row, total_rows):
            # Calculate percentage complete
            progress_pct = (current_row / total_rows) * 100 if total_rows > 0 else 0
            
            # Update the data status with the current progress
            data_status.set_progress(progress_pct)
            
            # Log progress update for tracking
            if current_row % max(1, int(total_rows/10)) == 0 or current_row == total_rows:
                logger.info(f"Generation progress: {current_row}/{total_rows} rows ({progress_pct:.1f}%)")
        
        # Generate the data, passing the progress callback
        generated_df = generate_data(
            csv_path=file_path,
            n_samples=n_samples,
            persist_dir="./data/chroma_db",
            features_dir="./data/features/",
            collection_name="dquery",
            output_path="./data/generated/generated_data.csv",
            progress_callback=progress_callback,  # Pass the callback
            **llm_params  # Pass through all LLM parameters
        )
        
        # Store the generated data
        data_status.complete_generation(generated_df)
        
        # Load the original data
        original_df = pd.read_csv(file_path)
        
        # Concatenate original data and generated data
        combined_df = pd.concat([original_df, generated_df], ignore_index=True)
        
        # Save to separate files
        output_dir = "./data/generated"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save just the synthetic data
        synthetic_path = os.path.join(output_dir, "synthetic_data.csv")
        generated_df.to_csv(synthetic_path, index=False)
        
        # Save the combined data
        combined_path = os.path.join(output_dir, "combined_data.csv")
        combined_df.to_csv(combined_path, index=False)
        
        # Also save a copy with timestamp to preserve history
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(output_dir, f"combined_data_{timestamp}.csv")
        combined_df.to_csv(archive_path, index=False)
        
        # Update result path in data_status
        data_status.set_result_path(combined_path)
        
        logger.info(f"Generated {len(generated_df)} rows of synthetic data")
        logger.info(f"Combined data saved to {combined_path} with {len(combined_df)} total rows")
        
        return generated_df
        
    except Exception as e:
        data_status.set_error(str(e))
        logger.error(f"Error in data generation: {str(e)}")
        return None

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
    global current_csv_file
    global data_status
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Create datasets directory if it doesn't exist
            temp_dir = "./datasets"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Delete any existing CSV files in the directory
            for existing_file in os.listdir(temp_dir):
                if existing_file.endswith('.csv'):
                    os.remove(os.path.join(temp_dir, existing_file))
                    logger.info(f"Deleted existing CSV file: {existing_file}")
            
            # Save the new file
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            
            # Update both global current_csv_file and data_status.current_file
            current_csv_file = file_path
            data_status.current_file = file_path
            logger.info(f"Set current CSV file to: {current_csv_file}")
            
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
            
            # Note: Data generation is no longer automatic on upload
            # It will be handled by the generation.js component instead
            
            return jsonify({
                'success': True,
                'rows': len(df),
                'columns': df.columns.tolist(),
                'message': 'File uploaded successfully'
            })
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File must be a CSV'}), 400

@app.route('/generation_status', methods=['GET'])
def get_generation_status():
    """Endpoint to check data generation status"""
    try:
        global data_status
        
        response = {
            'isGenerating': data_status.is_generating,
            'progress': data_status.progress,
            'total_rows': data_status.total_rows,
            'current_file': os.path.basename(data_status.current_file) if data_status.current_file else None,
            'error': data_status.error,
            'has_generated_data': data_status.generated_data is not None
        }
        
        # If there was an error, include it in the response
        if data_status.error:
            response['error_details'] = data_status.error
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in generation_status endpoint: {str(e)}")
        return jsonify({
            'isGenerating': False,
            'progress': 0,
            'error': f"Error retrieving generation status: {str(e)}"
        }), 500

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

@app.route('/stats_query', methods=['GET'])
def stats_query():
    """Endpoint to get statistical data for visualization"""
    try:
        chart_type = request.args.get('chart_type', 'overview')
        
        # Check if we have uploaded data
        uploaded_file = get_uploaded_csv()
        
        if not uploaded_file:
            return jsonify({'error': 'No data available. Please upload a CSV file first.'}), 400
        
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        if chart_type == 'overview':
            # Basic statistics for all numeric columns
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
            
            overview_stats = {
                'rowCount': int(len(df)),
                'columnCount': int(len(df.columns)),
                'numericColumnCount': int(len(numeric_columns)),
                'categoricalColumnCount': int(len(categorical_columns)),
                'missingValues': int(df.isna().sum().sum()),
                'numericColumns': numeric_columns,
                'categoricalColumns': categorical_columns,
            }
            
            # Convert any numpy types to native Python types
            overview_stats = convert_np_types(overview_stats)
            return jsonify(overview_stats)
            
        elif chart_type == 'histogram':
            column = request.args.get('column')
            if not column:
                return jsonify({'error': 'Column name required for histogram'}), 400
                
            if column not in df.columns:
                return jsonify({'error': f'Column {column} not found in dataset'}), 400
                
            if pd.api.types.is_numeric_dtype(df[column]):
                # Create histogram data for numeric column
                hist_data = df[column].dropna()
                bins = min(20, len(hist_data.unique()))
                
                if bins <= 1:  # Handle edge case with only one unique value
                    hist_data = [{'bin': str(hist_data.iloc[0]), 'count': len(hist_data)}]
                    return jsonify({
                        'type': 'numeric',
                        'column': column,
                        'data': hist_data
                    })
                
                counts, bin_edges = np.histogram(hist_data, bins=bins)
                
                # Format bin labels
                bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
                
                histogram_data = {
                    'type': 'numeric',
                    'column': column,
                    'data': [{'bin': bin_labels[i], 'count': int(counts[i])} for i in range(len(counts))]
                }
                return jsonify(histogram_data)
            else:
                # Create histogram for categorical column
                value_counts = df[column].value_counts().reset_index()
                value_counts.columns = ['value', 'count']
                
                # Limit to top 20 categories if there are too many
                if len(value_counts) > 20:
                    top_cats = value_counts.head(19)
                    other_count = value_counts.iloc[19:]['count'].sum()
                    other_row = pd.DataFrame({'value': ['Other'], 'count': [other_count]})
                    value_counts = pd.concat([top_cats, other_row])
                
                # Convert to dict and ensure numeric types are converted
                cat_data = value_counts.to_dict('records')
                for item in cat_data:
                    if isinstance(item['count'], (np.integer, np.int64)):
                        item['count'] = int(item['count'])
                
                histogram_data = {
                    'type': 'categorical',
                    'column': column,
                    'data': cat_data
                }
                return jsonify(histogram_data)
                
        elif chart_type == 'correlation':
            # Calculate correlation matrix for numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            
            if numeric_df.empty:
                return jsonify({'error': 'No numeric columns available for correlation analysis'}), 400
            
            corr_matrix = numeric_df.corr()
            
            # Convert to list of records for easier processing in frontend
            corr_data = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Only include each pair once
                        corr_data.append({
                            'source': col1,
                            'target': col2,
                            'correlation': float(corr_matrix.iloc[i, j])
                        })
            
            return jsonify({
                'correlationMatrix': corr_data  # Fix the key name to match what frontend expects
            })
            
        elif chart_type == 'scatter':
            x_col = request.args.get('x_column')
            y_col = request.args.get('y_column')
            
            if not x_col or not y_col:
                return jsonify({'error': 'Both x_column and y_column are required for scatter plot'}), 400
                
            if x_col not in df.columns or y_col not in df.columns:
                return jsonify({'error': 'One or both columns not found in dataset'}), 400
                
            if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
                return jsonify({'error': 'Both columns must be numeric for scatter plot'}), 400
            
            # Sample data if there are too many points (for performance)
            if len(df) > 1000:
                sample_df = df.sample(1000)
            else:
                sample_df = df
                
            scatter_data = []
            for _, row in sample_df.iterrows():
                if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                    scatter_data.append({
                        'x': float(row[x_col]),
                        'y': float(row[y_col])
                    })
            
            return jsonify({
                'x_column': x_col,
                'y_column': y_col,
                'scatterData': scatter_data  # Fix the key name to match what frontend expects
            })
            
        else:
            return jsonify({'error': f'Chart type {chart_type} not supported'}), 400
            
    except Exception as e:
        logger.error(f"Error in stats_query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/generate_data', methods=['POST'])
def generate_data_endpoint():
    global current_csv_file
    global data_status
    
    try:
        # Extract parameters from the request
        params = request.json
        
        # Ensure data_status.current_file is set from global if needed
        if data_status.current_file is None and current_csv_file is not None:
            data_status.current_file = current_csv_file
            
        # Get the CSV file path from the current file reference
        if data_status.current_file is None:
            return jsonify({"success": False, "error": "No CSV file has been uploaded"}), 400
            
        # Verify the file still exists
        if not os.path.exists(data_status.current_file):
            current_csv_file = None
            data_status.current_file = None
            return jsonify({"success": False, "error": "CSV file no longer exists"}), 400
        
        # Extract and validate parameters
        n_samples = int(params.get('numSamples', 10))
        if n_samples <= 0:
            return jsonify({"success": False, "error": "Number of samples must be positive"}), 400
            
        # Extract LLM parameters
        temperature = float(params.get('temperature', 0.7))
        top_p = float(params.get('topP', 0.9))
        repetition_penalty = float(params.get('repetitionPenalty', 1.1))
        max_tokens = int(params.get('maxTokens', 2048))
        
        # Start the data generation in a background thread
        thread = threading.Thread(
            target=generate_data_background,
            args=(data_status.current_file, n_samples),
            kwargs={
                'temperature': temperature,
                'top_p': top_p,
                'repetition_penalty': repetition_penalty,
                'max_tokens': max_tokens
            }
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "message": "Data generation started"}), 200
        
    except Exception as e:
        logger.error(f"Error in generate_data endpoint: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get_generated_data', methods=['GET'])
def get_generated_data():
    try:
        show_combined = request.args.get('combined', 'true').lower() == 'true'
        
        if data_status.generated_data is None:
            return jsonify({"success": False, "error": "No data has been generated yet"}), 404
            
        # Check if we should show combined data and if it exists
        if show_combined and data_status.result_path and os.path.exists(data_status.result_path):
            # Load combined data from file
            combined_df = pd.read_csv(data_status.result_path)
            
            # Get the original data count to mark which rows are original vs. synthetic
            original_count = len(combined_df) - len(data_status.generated_data)
            
            # Add a column to indicate if the row is original or synthetic
            combined_df['is_synthetic'] = [False] * original_count + [True] * len(data_status.generated_data)
            
            # Convert the DataFrame to a dict for JSON serialization
            data_dict = combined_df.to_dict(orient='records')
            
            return jsonify({
                "success": True,
                "data": data_dict,
                "columns": combined_df.columns.tolist(),
                "rowCount": len(data_dict),
                "originalCount": original_count,
                "syntheticCount": len(data_status.generated_data),
                "isCombined": True
            }), 200
        else:
            # Just return the synthetic data
            data_dict = data_status.generated_data.to_dict(orient='records')
            
            return jsonify({
                "success": True,
                "data": data_dict,
                "columns": data_status.generated_data.columns.tolist(),
                "rowCount": len(data_dict),
                "originalCount": 0,
                "syntheticCount": len(data_dict),
                "isCombined": False
            }), 200
            
    except Exception as e:
        logger.error(f"Error in get_generated_data endpoint: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Helper function to get the uploaded CSV
def get_uploaded_csv():
    global current_csv_file
    
    # Simply return the current CSV file if it exists
    if current_csv_file and os.path.exists(current_csv_file):
        return current_csv_file
    else:
        # Reset the variable if the file doesn't exist
        current_csv_file = None
        return None

@app.route('/delete_current_csv', methods=['POST'])
def delete_current_csv():
    global current_csv_file
    global data_status
    global query_engine
    
    try:
        # Check if there's a current file reference
        if current_csv_file:
            logger.info(f"Resetting CSV file reference: {current_csv_file}")
            
            # Reset the global variables without deleting the file
            current_csv_file = None
            data_status.current_file = None
            data_status.reset()
            
            # Clean up the RAG data
            cleanup_previous_data()
            
            # Reset the query engine
            query_engine = None
            
            return jsonify({
                'success': True,
                'message': 'CSV file reference and RAG data cleared successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No CSV file reference found to clear'
            }), 404
    except Exception as e:
        logger.error(f"Error clearing CSV file reference: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/visualize_embeddings', methods=['POST'])
def visualize_embeddings():
    try:
        global query_engine
        global current_csv_file
        
        # Check if data is uploaded and RAG index is generated
        if not current_csv_file:
            return jsonify({
                'success': False,
                'error': 'No data uploaded. Please upload a CSV file first.'
            }), 400
            
        if not query_engine:
            return jsonify({
                'success': False,
                'error': 'RAG index not generated. Please upload a CSV file first.'
            }), 400
        
        # Get user query from request
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Missing query parameter'
            }), 400
        
        try:
            # Try to import required packages
            import numpy as np
            from sklearn.manifold import TSNE
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Required packages not installed. Please install scikit-learn and numpy.'
            }), 500
        
        # Extract embeddings from the vector store and document store
        try:
            # Get the retriever and its components
            retriever = query_engine._retriever
            
            # Get the vector store (ChromaDB) and document store
            vector_store = None
            doc_store = None
            
            # Try to get vector store
            if hasattr(retriever, '_vector_store'):
                vector_store = retriever._vector_store
            elif hasattr(retriever, 'vector_store'):
                vector_store = retriever.vector_store
                
            # Try to get document store
            if hasattr(retriever, '_docstore'):
                doc_store = retriever._docstore
            elif hasattr(retriever, 'docstore'):
                doc_store = retriever.docstore
                
            if not vector_store:
                logger.error("Could not find vector store in retriever")
                return jsonify({
                    'success': False,
                    'error': 'Could not access vector store'
                }), 500
                
            if not doc_store:
                logger.error("Could not find document store in retriever")
                return jsonify({
                    'success': False,
                    'error': 'Could not access document store'
                }), 500
                
            # Get document IDs and embeddings from the vector store
            document_ids = []
            document_embeddings = []
            texts = []
            ids = []
            
            # Try different ways to get embeddings depending on the vector store type
            # For ChromaDB
            if hasattr(vector_store, '_collection'):
                # Get the ChromaDB collection
                chroma_collection = vector_store._collection
                
                # Get all embeddings from ChromaDB
                try:
                    # Get all data from the collection
                    collection_data = chroma_collection.get(include=["embeddings", "documents", "metadatas"])
                    
                    if collection_data and 'ids' in collection_data and len(collection_data['ids']) > 0:
                        # Get embeddings and document IDs
                        chroma_ids = collection_data['ids']
                        chroma_embeddings = collection_data.get('embeddings', [])
                        chroma_documents = collection_data.get('documents', [])
                        chroma_metadatas = collection_data.get('metadatas', [])
                        
                        if chroma_embeddings is None or len(chroma_embeddings) == 0:
                            logger.warning("No embeddings found in ChromaDB collection")
                        else:
                            document_ids = chroma_ids
                            document_embeddings = chroma_embeddings
                            
                            # Get the actual document text for each ID
                            for i, doc_id in enumerate(document_ids):
                                # Try to extract title from document content or metadata
                                doc_title = None
                                
                                # First try to get title from metadata if available
                                if chroma_metadatas and i < len(chroma_metadatas) and chroma_metadatas[i]:
                                    # Check if metadata has title, file_name, or doc_title
                                    metadata = chroma_metadatas[i]
                                    doc_title = metadata.get('title') or metadata.get('file_name') or metadata.get('doc_title')
                                
                                # If no title in metadata, try to extract from document content
                                if not doc_title and chroma_documents and i < len(chroma_documents):
                                    doc_content = chroma_documents[i]
                                    if doc_content:
                                        # Try to extract a title from the content (first line or first heading)
                                        lines = doc_content.strip().split('\n')
                                        if lines:
                                            # Look for a markdown heading or just use first line
                                            for line in lines[:3]:  # Check first few lines
                                                if line.startswith('# '):
                                                    doc_title = line.replace('# ', '')
                                                    break
                                            if not doc_title:
                                                doc_title = lines[0][:50]  # Use first line as title, limited to 50 chars
                                
                                # If we still don't have a title, try document store
                                if not doc_title:
                                    try:
                                        if hasattr(doc_store, 'get_document'):
                                            doc = doc_store.get_document(doc_id)
                                        elif hasattr(doc_store, 'get_document_by_id'):
                                            doc = doc_store.get_document_by_id(doc_id)
                                        elif hasattr(doc_store, 'get'):
                                            doc = doc_store.get(doc_id)
                                        else:
                                            # If we can't get the document, create a more readable ID
                                            short_id = str(doc_id)[-8:] if len(str(doc_id)) > 8 else str(doc_id)
                                            ids.append(str(doc_id))
                                            texts.append(f"Feature Document {short_id}")
                                            continue
                                            
                                        # Get text from document
                                        if hasattr(doc, 'text'):
                                            text = doc.text
                                            # Extract title from text if possible
                                            lines = text.strip().split('\n')
                                            if lines:
                                                for line in lines[:3]:
                                                    if line.startswith('# '):
                                                        doc_title = line.replace('# ', '')
                                                        break
                                                if not doc_title:
                                                    doc_title = lines[0][:50]
                                            
                                            text_preview = text[:100] + "..." if len(text) > 100 else text
                                            texts.append(text_preview)
                                            ids.append(str(doc_id))
                                        else:
                                            # Create readable ID
                                            short_id = str(doc_id)[-8:] if len(str(doc_id)) > 8 else str(doc_id)
                                            texts.append(f"Feature Document {short_id}")
                                            ids.append(str(doc_id))
                                            
                                    except Exception as doc_e:
                                        logger.warning(f"Could not get document text for ID {doc_id}: {str(doc_e)}")
                                        # Create readable ID
                                        short_id = str(doc_id)[-8:] if len(str(doc_id)) > 8 else str(doc_id)
                                        texts.append(f"Feature Document {short_id}")
                                        ids.append(str(doc_id))
                                else:
                                    # We have a title, use it along with a preview if available
                                    if chroma_documents and i < len(chroma_documents) and chroma_documents[i]:
                                        text_preview = f"{doc_title}: {chroma_documents[i][:100]}..." if len(chroma_documents[i]) > 100 else f"{doc_title}: {chroma_documents[i]}"
                                    else:
                                        text_preview = doc_title
                                    
                                    texts.append(text_preview)
                                    ids.append(str(doc_id))
                    else:
                        logger.warning("No document IDs found in ChromaDB collection")
                        
                except Exception as chroma_e:
                    logger.error(f"Error accessing ChromaDB collection: {str(chroma_e)}")
                    raise
            
            # If no embeddings found through vector store, try direct methods
            if document_embeddings is None or len(document_embeddings) == 0:
                logger.warning("No embeddings found in vector store, trying direct methods")
                # Additional fallback methods...
                
            if document_embeddings is None or len(document_embeddings) == 0:
                return jsonify({
                    'success': False,
                    'error': 'No embeddings found in the vector store.'
                }), 400
                
            # Generate embedding for the query
            from rag import get_embedding
            query_embedding = get_embedding(user_query)
            
            # Normalize all embeddings to ensure consistent scaling
            # This helps prevent the query from always appearing as an outlier
            from sklearn.preprocessing import normalize
            
            # Convert all embeddings to numpy arrays if they aren't already
            document_embeddings_np = np.array(document_embeddings)
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            # Combine and normalize all embeddings together
            all_embeddings_np = np.vstack([document_embeddings_np, query_embedding_np])
            all_embeddings_normalized = normalize(all_embeddings_np, norm='l2', axis=1)
            
            # Split back into document and query embeddings
            document_embeddings_normalized = all_embeddings_normalized[:-1]
            query_embedding_normalized = all_embeddings_normalized[-1:]
            
            # Calculate cosine similarity between query and each document
            similarities = []
            for i, doc_embedding in enumerate(document_embeddings_normalized):
                if i < len(ids):  # Make sure we have the corresponding ID
                    # Calculate similarity directly - embeddings are already normalized
                    sim_score = float(np.dot(query_embedding_normalized[0], doc_embedding))
                    
                    # Get a clean title from the text for display
                    display_text = texts[i] if i < len(texts) else f"Document {i}"
                    # If the text is long, extract just the title part
                    if ': ' in display_text and len(display_text) > 50:
                        display_title = display_text.split(': ')[0]
                    elif len(display_text) > 50:
                        display_title = display_text[:47] + "..."
                    else:
                        display_title = display_text
                        
                    similarities.append({
                        "id": ids[i] if i < len(ids) else f"doc_{i}",
                        "text": display_title,
                        "full_text": display_text,
                        "similarity": sim_score
                    })
            
            # Sort similarities in descending order for the bar chart
            sorted_similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
            
            # Apply dimensionality reduction using t-SNE
            # Fixed random state for more consistent visualizations
            perplexity = min(30, len(all_embeddings_normalized) - 1)
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                n_iter=1000,
                learning_rate='auto',
                init='pca'
            )
            embeddings_2d = tsne.fit_transform(all_embeddings_normalized)
            
            # Prepare labels and texts for visualization
            all_labels = ids + ["QUERY"]
            all_texts = texts + [user_query]
            
            # Prepare result data
            result_data = []
            for i, (coord, label, text) in enumerate(zip(embeddings_2d, all_labels, all_texts)):
                point_type = "query" if i == len(embeddings_2d) - 1 else "document"
                
                # Get display title for the point
                if point_type == "query":
                    display_title = "QUERY"
                    full_text = text
                else:
                    # Get a clean title from the text for display
                    if ': ' in text and len(text) > 50:
                        display_title = text.split(': ')[0]
                    elif len(text) > 50:
                        display_title = text[:47] + "..."
                    else:
                        display_title = text
                    full_text = text
                
                result_data.append({
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "id": label,
                    "text": display_title,
                    "full_text": full_text,
                    "type": point_type
                })
            
            return jsonify({
                "success": True,
                "data": result_data,
                "similarities": sorted_similarities
            })
            
        except AttributeError as e:
            logger.error(f"Error accessing embeddings: {str(e)}", exc_info=True)
            # Add debug information about query engine structure
            debug_info = {
                'query_engine_type': str(type(query_engine)),
                'has_retriever': hasattr(query_engine, '_retriever'),
            }
            if hasattr(query_engine, '_retriever'):
                debug_info['retriever_type'] = str(type(query_engine._retriever))
                debug_info['has_docstore'] = hasattr(query_engine._retriever, '_docstore')
                debug_info['has_vectorstore'] = hasattr(query_engine._retriever, '_vector_store')
                
                if hasattr(query_engine._retriever, '_docstore'):
                    debug_info['docstore_type'] = str(type(query_engine._retriever._docstore))
                    
                if hasattr(query_engine._retriever, '_vector_store'):
                    debug_info['vectorstore_type'] = str(type(query_engine._retriever._vector_store))
                    if hasattr(query_engine._retriever._vector_store, '_collection'):
                        debug_info['has_collection'] = True
                        debug_info['collection_type'] = str(type(query_engine._retriever._vector_store._collection))
            
            return jsonify({
                'success': False,
                'error': f'Could not access embeddings: {str(e)}. Check the structure of the query engine.',
                'debug_info': debug_info
            }), 500
        
    except Exception as e:
        logger.error(f"Error in visualize_embeddings: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Error visualizing embeddings: {str(e)}'
        }), 500

@app.route('/check_csv_status', methods=['GET'])
def check_csv_status():
    """Endpoint to check if a CSV file exists and return basic info"""
    global current_csv_file
    
    try:
        # Check if there's a current file reference
        if not current_csv_file or not os.path.exists(current_csv_file):
            if current_csv_file and not os.path.exists(current_csv_file):
                # Reset the reference if the file doesn't exist
                current_csv_file = None
            
            return jsonify({
                'success': False,
                'hasCSV': False,
                'message': 'No CSV file available'
            })
        
        # File exists, return basic info
        filename = os.path.basename(current_csv_file)
        file_size = os.path.getsize(current_csv_file)
        
        # Try to read basic info about the file
        try:
            df = pd.read_csv(current_csv_file)
            rows = len(df)
            columns = len(df.columns)
            column_names = df.columns.tolist()
            
            # Return success with file info
            return jsonify({
                'success': True,
                'hasCSV': True,
                'filename': filename,
                'fileSize': file_size,
                'rows': rows,
                'columns': columns,
                'columnNames': column_names
            })
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            # Still return success but with error message
            return jsonify({
                'success': False,
                'hasCSV': True,
                'filename': filename,
                'fileSize': file_size,
                'error': f"Error reading CSV file: {str(e)}"
            })
            
    except Exception as e:
        logger.error(f"Error in check_csv_status endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'hasCSV': False,
            'error': str(e)
        }), 500

@app.route('/query_csv', methods=['POST'])
def query_csv():
    """Endpoint to run pandas queries on the current CSV file"""
    global current_csv_file
    
    try:
        # Check if there's a current file reference
        if not current_csv_file or not os.path.exists(current_csv_file):
            if current_csv_file and not os.path.exists(current_csv_file):
                # Reset the reference if the file doesn't exist
                current_csv_file = None
            
            return jsonify({
                'success': False,
                'error': 'No CSV file available'
            }), 400
        
        # Get query from request
        data = request.json
        query_str = data.get('query', '')
        
        if not query_str:
            return jsonify({
                'success': False,
                'error': 'No query provided'
            }), 400
        
        # Read the CSV file
        df = pd.read_csv(current_csv_file)
        
        try:
            # Execute the query
            # We'll use eval to execute the pandas query string
            # This is risky but necessary for the functionality
            # We should add more validation in a production environment
            result_df = df.query(query_str)
            
            # Convert to dict for JSON response
            if len(result_df) > 100:
                # Limit to 100 rows
                result_df = result_df.head(100)
                truncated = True
            else:
                truncated = False
                
            result_dict = result_df.to_dict(orient='records')
            
            return jsonify({
                'success': True,
                'data': result_dict,
                'columns': result_df.columns.tolist(),
                'rowCount': len(result_df),
                'truncated': truncated,
                'totalRowCount': len(df),
                'matchedRowCount': len(result_df)
            })
            
        except Exception as e:
            logger.error(f"Error executing pandas query: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Error executing query: {str(e)}"
            }), 400
            
    except Exception as e:
        logger.error(f"Error in query_csv endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
<<<<<<< HEAD
    chat_history = ChatHistory()
    app.run()
=======
    app.run(debug=True)
>>>>>>> upstream/main
