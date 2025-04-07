import os
import pandas as pd
import numpy as np
import logging
import csv
import subprocess
import json
from sklearn.model_selection import train_test_split
from generate_data import generate_data
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_custom_csv(file_path, output_path=None):
    """
    Reads a CSV file where the header is stored in the second column of the first row,
    and each subsequent row's second column contains a semicolon-separated string
    representing the data. Optionally writes the parsed DataFrame to output_path
    as a semicolon-separated CSV file.

    Args:
        file_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save the processed CSV file.
                                     If provided, the DataFrame will be written with semicolons as separators.
    
    Returns:
        pd.DataFrame: Parsed DataFrame with appropriate column names.
    """
    # Read the CSV without a header so that all rows are treated as data.
    df_raw = pd.read_csv(file_path, sep=';', header=None)
    
    # The first row, second column contains the header string.
    header_str = df_raw.iloc[0, 1]
    columns = header_str.split(';')
    
    # Remove the header row and reset the index.
    df_data = df_raw.iloc[1:].reset_index(drop=True)
    
    # The data is stored in the second column as a semicolon-separated string.
    data = df_data[1].str.split(';', expand=True)
    data.columns = columns
    
    # Optionally, if you want to include the first column as an index:
    data['index'] = df_data[0].astype(int)
    data = data.set_index('index')
    
    # Write the parsed DataFrame to output_path if provided, using semicolon as separator.
    if output_path:
        data.to_csv(output_path, sep=';')
    
    return data


def run_evaluation_pipeline(dataset_path, output_dir="./results", test_size=0.2, n_samples=None, 
                           models=None, random_state=42, features_dir="./data/features/"):
    """
    Run a complete evaluation pipeline:
    1. Split original dataset into train (80%) and test (20%) 
    2. Evaluate models on the test data
    3. Generate synthetic data using the train data
    4. Evaluate models on the synthetic data
    5. Compare results between test and synthetic data
    
    Args:
        dataset_path: Path to the original dataset CSV
        output_dir: Directory to save results
        test_size: Proportion of data to use for testing (default: 0.2)
        n_samples: Number of synthetic samples to generate (default: size of test set)
        models: List of models to evaluate (default: all models in evals/models)
        random_state: Random seed for reproducibility
        features_dir: Directory for storing feature analysis files
        
    Returns:
        dict: Dictionary containing evaluation metrics for all models
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs("./data/features/", exist_ok=True)
    os.makedirs("./data/chroma_db", exist_ok=True)
    os.makedirs("./data/generated", exist_ok=True)
    
    # Step 1: Load and split original dataset
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path, sep=';')  # Read with semicolon separator
    
    # Add these debug lines
    print(f"First row of loaded data: {df.iloc[0].to_dict()}")
    print(f"Data types: {df.dtypes}")
    
    # Determine label column (assumed to be the last column)
    label_column = df.columns[-1]
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Reassemble into dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save train and test splits
    train_path = os.path.join(output_dir, "data", "train_data.csv")
    test_path = os.path.join(output_dir, "data", "test_data.csv")
    train_df.to_csv(train_path, index=False, sep=';')  # Use semicolon separator
    test_df.to_csv(test_path, index=False, sep=';')
    logger.info(f"Saved train data ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved test data ({len(test_df)} rows) to {test_path}")
    
    # Set default number of samples if not provided
    if n_samples is None:
        n_samples = len(test_df)
    
    # Step 2: Evaluate models on the test data
    logger.info("Evaluating models on test data")
    test_results = run_model_evaluations(test_df, output_prefix="test", output_dir=output_dir, models=models)
    
    # Step 3: Generate synthetic data using the train data
    logger.info(f"Generating {n_samples} synthetic data samples using train data")
    synthetic_path = os.path.join(output_dir, "data", "synthetic_data.csv")
    
    # Use train data for analysis and generation
    analyze_csv_features(train_path, features_dir)
    
    # Generate synthetic data
    synthetic_df = generate_data(
        csv_path=train_path,
        n_samples=n_samples,
        persist_dir="./data/chroma_db",
        features_dir=features_dir,
        collection_name="dquery",
        output_path=synthetic_path,
        max_workers=5,
        batch_size=100
    )
    
    if synthetic_df is None or len(synthetic_df) == 0:
        logger.error("Failed to generate synthetic data")
        return None
    
    logger.info(f"Generated {len(synthetic_df)} rows of synthetic data")
    synthetic_df.to_csv('synthetic_output.csv', index=True,sep=';')
    synthetic_df = parse_custom_csv('synthetic_output.csv')
    # Step 4: Evaluate models on the synthetic data
    logger.info("Evaluating models on synthetic data")
    synthetic_results = run_model_evaluations(synthetic_df, output_prefix="synthetic", output_dir=output_dir, models=models)
    
    # Step 5: Compare results and generate comparison report
    logger.info("Comparing results between test and synthetic data")
    comparison = compare_results(test_results, synthetic_results, output_dir)
    
    return {
        "test_results": test_results,
        "synthetic_results": synthetic_results,
        "comparison": comparison
    }

def analyze_csv_features(csv_path, output_dir="./data/features/"):
    """
    Analyze the CSV file to generate feature information for data generation
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Output directory for feature analysis files
    """
    try:
        # First try to import the asynchronous analyzer
        from dquery import analyze_csv_features
        logger.info(f"Analyzing features in {csv_path}")
        analyze_csv_features(csv_path, output_dir)
    except (ImportError, AttributeError):
        # Fallback to a basic approach if dquery is not available
        logger.warning("Could not import dquery.analyze_csv_features, using basic analysis")
        df = pd.read_csv(csv_path)
        
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a basic dataset overview
        dataset_name = os.path.basename(csv_path).split('.')[0]
        rows, cols = df.shape
        headers = df.columns.tolist()
        
        # Write overview file
        overview_path = os.path.join(output_dir, f"{dataset_name}_overview.txt")
        with open(overview_path, 'w') as f:
            f.write(f"# Dataset Overview: {dataset_name}\n\n")
            f.write(f"Total rows: {rows}\n")
            f.write(f"Total columns: {cols}\n")
            f.write(f"Features: {', '.join(headers)}\n\n")
            f.write("## Column Types\n\n")
            for feature in headers:
                dtype = df[feature].dtype
                f.write(f"- {feature}: {dtype}\n")
        
        # Create feature files
        for feature in headers:
            feature_path = os.path.join(output_dir, f"{dataset_name}_{feature}.md")
            with open(feature_path, 'w') as f:
                f.write(f"# Feature: {feature}\n\n")
                dtype = df[feature].dtype
                f.write(f"Data type: {dtype}\n\n")
                
                if pd.api.types.is_numeric_dtype(dtype):
                    f.write("## Statistics\n\n")
                    f.write(f"- Min: {df[feature].min()}\n")
                    f.write(f"- Max: {df[feature].max()}\n")
                    f.write(f"- Mean: {df[feature].mean()}\n")
                    f.write(f"- Median: {df[feature].median()}\n")
                    f.write(f"- Standard deviation: {df[feature].std()}\n")
                else:
                    f.write("## Value counts\n\n")
                    value_counts = df[feature].value_counts().head(10)
                    for value, count in value_counts.items():
                        f.write(f"- {value}: {count}\n")

def run_model_evaluations(data, output_prefix, output_dir="./results", models=None):
    """
    Run all evaluation models on the given data and save results
    
    Args:
        data: DataFrame containing the data to evaluate
        output_prefix: Prefix for output files (e.g., "test" or "synthetic")
        output_dir: Directory to save results
        models: List of model names to run (default: all models in evals/models)
        
    Returns:
        dict: Dictionary containing evaluation metrics for all models
    """
    # Ensure output directories exist
    model_results_dir = os.path.join(output_dir, "model_results")
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Save data to a temporary file for models that read from CSV
    temp_data_path = os.path.join(model_results_dir, f"{output_prefix}_data.csv")
    
    # Debug the data before saving
    print(f"Data shape before saving: {data.shape}")
    print(f"Data columns before saving: {data.columns.tolist()}")
    
    # Make sure column headers are properly included
    data.to_csv(temp_data_path, index=False, sep=';', header=True)
    
    # Determine models to evaluate
    if models is None:
        # Get all python files in the evals/models directory
        models_dir = os.path.join("evals", "models")
        if not os.path.exists(models_dir):
            logger.error(f"Models directory {models_dir} not found")
            return {}
            
        models = [f[:-3] for f in os.listdir(models_dir) 
                 if f.endswith('.py') and not f.startswith('__')]
    
    # Import the data_util module from evals/models (if needed)
    try:
        from evals.models.data_util import process_data
        logger.info("Successfully imported data processing utility")
    except ImportError:
        logger.warning("Could not import data_util.process_data, data preprocessing may be skipped")
        process_data = lambda df: df  # No-op function
    
    # Process the data
    processed_data = process_data(data)
    processed_data_path = os.path.join(model_results_dir, f"{output_prefix}_processed_data.csv")
    processed_data.to_csv(processed_data_path, index=False, sep=';')
    
    # Results will be collected here
    results = {}
    
    # Prepare metrics CSV
    metrics_path = os.path.join(output_dir, f"{output_prefix}_metrics.csv")
    with open(metrics_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])
    
    # Run each model
    for model_name in models:
        logger.info(f"Running model: {model_name}")
        
        try:
            # Extract features and target from processed_data
            X = processed_data.drop(data.columns[-1], axis=1)
            y = processed_data[data.columns[-1]]
            
            # Instead of splitting again, use the test split from processed_data
            # (Assuming processed_data represents our test set here)
            # Create a temporary module to run the model
            temp_module_name = f"temp_model_{model_name}"
            temp_module_path = os.path.join(model_results_dir, f"{temp_module_name}.py")
            
            # Precompute fixed paths to avoid backslashes in f-string expressions
            project_root = os.path.abspath(os.path.dirname(__file__))
            fixed_project_root = project_root.replace("\\", "/")
            fixed_processed_data_path = processed_data_path.replace("\\", "/")
            metrics_file_path = os.path.join(model_results_dir, f'{output_prefix}_{model_name}_metrics.json').replace("\\", "/")
            confusion_file_path = os.path.join(model_results_dir, f'{output_prefix}_{model_name}_confusion.png').replace("\\", "/")
            fixed_metrics_csv_path = metrics_path.replace("\\", "/")
            
            # Add this at the beginning of your temp script to print paths
            with open(temp_module_path, 'w') as f:
                f.write(f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import csv
import os
import json

# Debug info
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)
print("Directory contents:", os.listdir("."))
if os.path.exists("evals"):
    print("evals directory exists, contents:", os.listdir("evals"))
else:
    print("evals directory not found!")

# Add the project root to the path
project_root = "{fixed_project_root}"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "evals", "models"))

print("Python path after additions:", sys.path)

# Try to import and report results
try:
    from evals.models.{model_name} import *
    print("Successfully imported {model_name} model")
except ImportError as e:
    print(f"Import error: {{e}}")
""")
            
            with open(temp_module_path, 'a') as f:
                f.write(f"""
# Add the root directory to the path
sys.path.append("{fixed_project_root}")
sys.path.append(os.path.join("{fixed_project_root}", "evals", "models"))

# Import the required modules
try:
    from evals.models.data_util import process_data
    from evals.models.{model_name} import run_model
    print("Successfully imported modules")
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

# Debug raw CSV contents
with open("{fixed_processed_data_path}", 'r') as csvfile:
    print("First few lines of raw CSV file:")
    for i, line in enumerate(csvfile):
        if i < 5:  # Print first 5 lines
            print(f"Line {{i}}: {{line.strip()}}")
        else:
            break

# Load our prepared data with explicit header setting
df = pd.read_csv("{fixed_processed_data_path}", sep=';')
print(f"Original data shape: {{df.shape}}")
print(f"CSV column names: {{df.columns.tolist()}}")

# Make sure we're using the right target column
label_column = '{data.columns[-1]}'
X = df.drop(label_column, axis=1)
y = df[label_column]

print(f"X columns: {{X.columns.tolist()}}")
print(f"X shape: {{X.shape}}, y shape: {{len(y)}}")

print(f"First 5 rows of X:\\n{{X.head().to_string()}}")
print(f"First 5 values of y: {{y.head().values}}")

# Try to get predictions
try:
    print("Calling run_model...")
    y_pred = run_model(X, y)
    print("run_model completed successfully!")
except Exception as e:
    import traceback
    print(f"Error in run_model: {{type(e).__name__}}: {{str(e)}}")
    traceback.print_exc()
    sys.exit(1)
""")
            
            with open(temp_module_path, 'a') as f:
                f.write(f"""
# Calculate metrics
metrics = {{
    'accuracy': accuracy_score(y, y_pred),
    'classification_report': classification_report(y, y_pred, output_dict=True)
}}

# Save metrics to file
metrics_file = "{metrics_file_path}"
with open(metrics_file, 'w') as f_out:
    json.dump(metrics, f_out, indent=2)

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('{model_name} Confusion Matrix ({output_prefix} data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("{confusion_file_path}")
plt.close()

# Print key metrics
print(f"Accuracy: {{metrics['accuracy']:.4f}}")
cr = metrics['classification_report']
weighted_avg = cr['weighted avg']
print(f"Precision: {{weighted_avg['precision']:.4f}}")
print(f"Recall: {{weighted_avg['recall']:.4f}}")
print(f"F1 Score: {{weighted_avg['f1-score']:.4f}}")

# Append metrics to the CSV file
with open("{fixed_metrics_csv_path}", 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        '{model_name}', 
        f"{{metrics['accuracy']:.4f}}", 
        f"{{weighted_avg['precision']:.4f}}", 
        f"{{weighted_avg['recall']:.4f}}", 
        f"{{weighted_avg['f1-score']:.4f}}"
    ])
""")
            
            # Execute the temporary module with the correct working directory
            result = subprocess.run(
                ["python", temp_module_path], 
                capture_output=True, 
                text=True,
                cwd=os.path.abspath(os.path.dirname(__file__))  # Set working directory to project root
            )
            
            logger.info(f"Model output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Model errors: {result.stderr}")
            
            # Try to load metrics from the file
            metrics_file = os.path.join(model_results_dir, f"{output_prefix}_{model_name}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f_in:
                    metrics = json.load(f_in)
                results[model_name] = metrics
            else:
                logger.warning(f"No metrics file found for {model_name}")
                
            # Clean up temporary module
            if os.path.exists(temp_module_path):
                os.remove(temp_module_path)
                
        except Exception as e:
            logger.error(f"Error running model {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results

def compare_results(test_results, synthetic_results, output_dir):
    """
    Compare results between test and synthetic data evaluations
    
    Args:
        test_results: Results from test data evaluation
        synthetic_results: Results from synthetic data evaluation
        output_dir: Directory to save comparison results
        
    Returns:
        dict: Comparison metrics
    """
    comparison = {}
    
    # Create comparison CSV
    comparison_path = os.path.join(output_dir, "comparison.csv")
    with open(comparison_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Model', 'Test_Accuracy', 'Synthetic_Accuracy', 'Difference', 'Better'])
    
    # Create lists for plotting
    models = []
    test_acc = []
    synthetic_acc = []
    
    # Compare each model
    for model_name in set(test_results.keys()).union(synthetic_results.keys()):
        if model_name in test_results and model_name in synthetic_results:
            try:
                test_accuracy = test_results[model_name].get('accuracy', 0)
                synth_accuracy = synthetic_results[model_name].get('accuracy', 0)
                difference = synth_accuracy - test_accuracy
                better = "Synthetic" if difference > 0 else "Test" if difference < 0 else "Equal"
                
                comparison[model_name] = {
                    "test_accuracy": test_accuracy,
                    "synthetic_accuracy": synth_accuracy,
                    "difference": difference,
                    "better": better
                }
                
                # Add to CSV
                with open(comparison_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([
                        model_name, 
                        f"{test_accuracy:.4f}", 
                        f"{synth_accuracy:.4f}", 
                        f"{difference:.4f}", 
                        better
                    ])
                
                # Add to plot data
                models.append(model_name)
                test_acc.append(test_accuracy)
                synthetic_acc.append(synth_accuracy)  # fixed variable name here
                
            except Exception as e:
                logger.error(f"Error comparing results for {model_name}: {str(e)}")
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 8))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, test_acc, width, label='Test Data')
    plt.bar(x + width/2, synthetic_acc, width, label='Synthetic Data')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison: Test vs. Synthetic Data')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    comparison_plot_path = os.path.join(output_dir, "plots", "accuracy_comparison.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    
    logger.info(f"Saved comparison results to {comparison_path}")
    logger.info(f"Saved comparison plot to {comparison_plot_path}")
    
    return comparison

if __name__ == "__main__":
    # Example usage
    dataset_path = "./evals/dataset/andrew_diabetes.csv"  # Replace with your dataset
    
    # Define models to evaluate (make sure these match your filenames in evals/models)
    models = ["knn", "mlp", "naivebayes", "randomforest", "sgd", "svm"]
    
    # Run the full pipeline
    results = run_evaluation_pipeline(
        dataset_path=dataset_path,
        output_dir="./results",
        test_size=0.2,
        n_samples=None,  # Will default to test size
        models=models,
        random_state=42,
        features_dir="./data/features/"
    )
    
    if results:
        logger.info("Pipeline completed successfully")
        
        # Print summary of which approach was better for each model
        for model_name, comp in results["comparison"].items():
            better = comp["better"]
            diff = abs(comp["difference"]) * 100
            logger.info(f"{model_name}: {better} data performed better by {diff:.2f}%")
    else:
        logger.error("Pipeline failed to complete")
