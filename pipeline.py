import os
import pandas as pd
import numpy as np
import logging
import csv
import importlib
from sklearn.model_selection import train_test_split
from generate_data import generate_data
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
    df = pd.read_csv(dataset_path)
    
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
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
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
    data.to_csv(temp_data_path, index=False)
    
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
    processed_data.to_csv(processed_data_path, index=False)
    
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
            
            # Create a temporary script that uses our data and runs the model
            with open(temp_module_path, 'w') as f:
                f.write(f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
import json
import csv

# Add the parent directory to the path to import data_util
sys.path.append(os.path.abspath("./evals/models"))
from data_util import process_data

# Import the model functionality from the model script
from evals.models.{model_name} import *

# Load our prepared processed data
df = pd.read_csv("{processed_data_path.replace('\\', '/')}")
X = df.drop('{data.columns[-1]}', axis=1)
y = df['{data.columns[-1]}']

# Try to get predictions by calling a function run_model if it exists;
# otherwise, fall back to assuming a global 'model' variable exists.
try:
    y_pred = run_model(X, y)  # run_model should be defined in the model script
except Exception as e:
    try:
        y_pred = model.predict(X)
    except Exception as e:
        print(f"Error: unable to get predictions for model {model_name} - {{str(e)}}")
        sys.exit(1)

# Compute metrics
metrics = {{
    'accuracy': accuracy_score(y, y_pred),
    'classification_report': classification_report(y, y_pred, output_dict=True)
}}

# Save metrics to file
metrics_file = "{os.path.join(model_results_dir, f'{output_prefix}_{model_name}_metrics.json').replace('\\', '/')}"
with open(metrics_file, 'w') as f_out:
    json.dump(metrics, f_out, indent=2)

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('{model_name} Confusion Matrix ({output_prefix} data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("{os.path.join(model_results_dir, f'{output_prefix}_{model_name}_confusion.png').replace('\\', '/')}")
plt.close()

# Print key metrics
print(f"Accuracy: {{metrics['accuracy']:.4f}}")
cr = metrics['classification_report']
weighted_avg = cr['weighted avg']
print(f"Precision: {{weighted_avg['precision']:.4f}}")
print(f"Recall: {{weighted_avg['recall']:.4f}}")
print(f"F1 Score: {{weighted_avg['f1-score']:.4f}}")

# Append metrics to the CSV file
with open("{metrics_path.replace('\\', '/')}", 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        '{model_name}', 
        f"{{metrics['accuracy']:.4f}}", 
        f"{{weighted_avg['precision']:.4f}}", 
        f"{{weighted_avg['recall']:.4f}}", 
        f"{{weighted_avg['f1-score']:.4f}}"
    ])
""")
            
            # Execute the temporary module
            import subprocess
            result = subprocess.run(
                ["python", temp_module_path], 
                capture_output=True, 
                text=True
            )
            
            logger.info(f"Model output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Model errors: {result.stderr}")
            
            # Try to load metrics from the file
            import json
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
                synthetic_acc.append(synth_accuracy)
                
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
