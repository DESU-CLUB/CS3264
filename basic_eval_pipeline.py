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

def parse_synthetic_df(df,columns):
    """
    Parses a synthetic DataFrame that is returned by generate_data,
    where the DataFrame has a single column containing semicolon-separated values.
    
    The function:
      - Extracts the header from the first row.
      - Splits the remaining rows based on ';' to form individual columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame with one column.
        
    Returns:
        pd.DataFrame: A properly parsed DataFrame with multiple columns.
    """
    # Check that the DataFrame has a single column
    if df.shape[1] != 1:
        raise ValueError("Expected a single-column DataFrame for parsing.")
    
    
    data = df.iloc[:, 0].str.split(';', expand=True)
    data.columns = columns
    
    return data

def run_evaluation_pipeline(dataset_path, output_dir="./results", test_size=0.2, n_samples=None, 
                            models=None, random_state=42, features_dir="./data/features/"):
    """
    Run a complete evaluation pipeline with the following revised workflow:
      1. Split the original dataset into train (80%) and test (20%).
      2. Train models on the original training data and evaluate them on the test set.
      3. Generate synthetic data using the training set (with n_samples defaulting to len(train_df)).
      4. Train models on the synthetic data and evaluate them on the same test set.
      5. Compare the model performance (evaluated on test data) between the original and synthetic setups.
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
    df = pd.read_csv(dataset_path, sep=';')
    if "gender" in df.columns:
        df['gender'] = df['gender'].apply(lambda x: 1 if str(x).strip().lower() == 'male' else 0)
    print(f"First row of loaded data: {df.iloc[0].to_dict()}")
    print(f"Data types: {df.dtypes}")

    label_column = df.columns[-1]
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test, y_test], axis=1)
    assert test_df.isnull().sum().sum() == 0, "test contains missing values"
    assert train_df.isnull().sum().sum() == 0, "train contains missing values"
    train_path = os.path.join(output_dir, "data", "train_data.csv")
    test_path  = os.path.join(output_dir, "data", "test_data.csv")
    logger.info(f"Saved train data ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved test data ({len(test_df)} rows) to {test_path}")
    train_df.to_csv(train_path, index=False, sep=';')
    test_df.to_csv(test_path, index=False, sep=';')
    # Set default number of synthetic samples if not provided (from the original training set)
    if n_samples is None:
        n_samples = len(train_df)

    # Step 2: Train & evaluate models on original training data (evaluated on test set)
    logger.info("Training and evaluating models on original training data")
    original_results = run_model_evaluations(train_df, test_df, output_prefix="original", output_dir=output_dir, models=models)

    # Step 3: Generate synthetic data from the training data
    logger.info(f"Generating {n_samples} synthetic data samples using train data")
    synthetic_path = os.path.join(output_dir, "data", "synthetic_data.csv")
    analyze_csv_features(train_path, features_dir)

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
    logger.info(f"Generated synthetic data with {synthetic_df.head()} rows")
    
    parsed_synthetic_df = parse_synthetic_df(synthetic_df,X_train.columns.tolist()+[label_column])
    print(parsed_synthetic_df.head())
    print(parsed_synthetic_df.info())
    logger.info(f"Generated synthetic data with {len(parsed_synthetic_df)} rows")
    logger.info(f"synthetic_df shape: {parsed_synthetic_df.shape}")
    logger.info(parsed_synthetic_df.head())
    

    # Step 4: Train & evaluate models on synthetic data (evaluated on the same test set)
    logger.info("Training and evaluating models on synthetic data")
    synthetic_results = run_model_evaluations(parsed_synthetic_df, test_df, output_prefix="synthetic", output_dir=output_dir, models=models)

    # Step 5: Compare results (both sets are evaluated on the same test set)
    logger.info("Comparing results between models trained on original vs synthetic data (both evaluated on test data)")
    comparison = compare_results(original_results, synthetic_results, output_dir)

    return {
        "original_results": original_results,
        "synthetic_results": synthetic_results,
        "comparison": comparison
    }

def analyze_csv_features(csv_path, output_dir="./data/features/"):
    """
    Analyze the CSV file to generate feature information for data generation
    """
    try:
        from dquery import analyze_csv_features
        logger.info(f"Analyzing features in {csv_path}")
        analyze_csv_features(csv_path, output_dir)
    except (ImportError, AttributeError):
        logger.warning("Could not import dquery.analyze_csv_features, using basic analysis")
        df = pd.read_csv(csv_path)
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = os.path.basename(csv_path).split('.')[0]
        rows, cols = df.shape
        headers = df.columns.tolist()
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

def run_model_evaluations(train_data, test_data, output_prefix, output_dir="./results", models=None):
    """
    For each model, train using the provided training data and evaluate on the provided test data.
    Saves the results and returns a dictionary of evaluation metrics.
    """
    model_results_dir = os.path.join(output_dir, "model_results")
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Save train and test data temporarily
    temp_train_path = os.path.join(model_results_dir, f"{output_prefix}_train_data.csv")
    temp_test_path = os.path.join(model_results_dir, f"{output_prefix}_test_data.csv")
    train_data.to_csv(temp_train_path, index=False, sep=';')
    test_data.to_csv(temp_test_path, index=False, sep=';')
    
    # Debug info
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Determine models to evaluate
    if models is None:
        models_dir = os.path.join("evals", "models")
        if not os.path.exists(models_dir):
            logger.error(f"Models directory {models_dir} not found")
            return {}
        models = [f[:-3] for f in os.listdir(models_dir)
                  if f.endswith('.py') and not f.startswith('__')]
    
    try:
        from evals.models.data_util import process_data
        logger.info("Successfully imported data processing utility")
    except ImportError:
        logger.warning("Could not import data_util.process_data, skipping processing")
        process_data = lambda df: df
    
    results = {}
    metrics_path = os.path.join(output_dir, f"{output_prefix}_metrics.csv")
    with open(metrics_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])
    
    for model_name in models:
        logger.info(f"Running model: {model_name}")
        try:
            temp_module_name = f"temp_model_{model_name}"
            temp_module_path = os.path.join(model_results_dir, f"{temp_module_name}.py")
            
            project_root = os.path.abspath(os.path.dirname(__file__))
            fixed_project_root = project_root.replace("\\", "/")
            fixed_train_path = temp_train_path.replace("\\", "/")
            fixed_test_path = temp_test_path.replace("\\", "/")
            metrics_file_path = os.path.join(model_results_dir, f'{output_prefix}_{model_name}_metrics.json').replace("\\", "/")
            confusion_file_path = os.path.join(model_results_dir, f'{output_prefix}_{model_name}_confusion.png').replace("\\", "/")
            fixed_metrics_csv_path = metrics_path.replace("\\", "/")
            
            # Create temporary module script that trains on train_data and evaluates on test_data.
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

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)
project_root = "{fixed_project_root}"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "evals", "models"))
print("Python path after additions:", sys.path)

try:
    from evals.models.{model_name} import *
    print("Successfully imported {model_name} model")
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)
""")
            with open(temp_module_path, 'a') as f:
                f.write(f"""
# Load training and test data
df_train = pd.read_csv("{fixed_train_path}", sep=';')
df_test = pd.read_csv("{fixed_test_path}", sep=';')

print("Train data shape:", df_train.shape)
print("Test data shape:", df_test.shape)

# Determine target column (assumed to be the last column)
label_column = df_train.columns[-1]
X_train = df_train.drop(label_column, axis=1)
y_train = df_train[label_column]
X_test = df_test.drop(label_column, axis=1)
y_test = df_test[label_column]

print("X_train columns:", X_train.columns.tolist())
print("X_train shape:", X_train.shape)
print("Test labels count:", len(y_test))

# Train the model and get predictions
try:
    print("Calling run_model with training and test data...")
    # This assumes that run_model accepts (X_train, y_train, X_test, y_test)
    y_pred = run_model(X_train, y_train, X_test, y_test)
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
    'accuracy': accuracy_score(y_test, y_pred),
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}}

with open("{metrics_file_path}", 'w') as f_out:
    json.dump(metrics, f_out, indent=2)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('{model_name} Confusion Matrix ({output_prefix} data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("{confusion_file_path}")
plt.close()

print(f"Accuracy: {{metrics['accuracy']:.4f}}")
cr = metrics['classification_report']
weighted_avg = cr['weighted avg']
print(f"Precision: {{weighted_avg['precision']:.4f}}")
print(f"Recall: {{weighted_avg['recall']:.4f}}")
print(f"F1 Score: {{weighted_avg['f1-score']:.4f}}")

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
            result = subprocess.run(
                ["python", temp_module_path],
                capture_output=True,
                text=True,
                cwd=os.path.abspath(os.path.dirname(__file__))
            )
            
            logger.info(f"Model output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Model errors: {result.stderr}")
            
            if os.path.exists(metrics_file_path):
                with open(metrics_file_path, 'r') as f_in:
                    metrics = json.load(f_in)
                results[model_name] = metrics
            else:
                logger.warning(f"No metrics file found for {model_name}")
            
            if os.path.exists(temp_module_path):
                os.remove(temp_module_path)
                
        except Exception as e:
            logger.error(f"Error running model {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results

def compare_results(original_results, synthetic_results, output_dir):
    """
    Compare results between models trained on original data and synthetic data evaluations.
    Both sets of results are obtained by evaluating on the same test data.
    """
    comparison = {}
    comparison_path = os.path.join(output_dir, "comparison.csv")
    with open(comparison_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Model', 'Original_Accuracy', 'Synthetic_Accuracy', 'Difference', 'Better'])
    
    models = []
    original_acc = []
    synthetic_acc = []
    
    for model_name in set(original_results.keys()).union(synthetic_results.keys()):
        if model_name in original_results and model_name in synthetic_results:
            try:
                orig_acc = original_results[model_name].get('accuracy', 0)
                synth_acc = synthetic_results[model_name].get('accuracy', 0)
                difference = synth_acc - orig_acc
                better = "Synthetic" if difference > 0 else "Original" if difference < 0 else "Equal"
                comparison[model_name] = {
                    "original_accuracy": orig_acc,
                    "synthetic_accuracy": synth_acc,
                    "difference": difference,
                    "better": better
                }
                with open(comparison_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([
                        model_name,
                        f"{orig_acc:.4f}",
                        f"{synth_acc:.4f}",
                        f"{difference:.4f}",
                        better
                    ])
                models.append(model_name)
                original_acc.append(orig_acc)
                synthetic_acc.append(synth_acc)
            except Exception as e:
                logger.error(f"Error comparing results for {model_name}: {str(e)}")
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, original_acc, width, label='Original Data')
    plt.bar(x + width/2, synthetic_acc, width, label='Synthetic Data')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison: Original vs Synthetic Data Training (Evaluated on Test Data)')
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
    dataset_path = "./evals/dataset/andrew_diabetes.csv"
    models = ["knn","mlp", "naivebayes", "randomforest", "sgd", "svm"]
    
    results = run_evaluation_pipeline(
        dataset_path=dataset_path,
        output_dir="./results",
        test_size=0.2,
        n_samples=None,    # Defaults to the size of train data
        models=models,
        random_state=42,
        features_dir="./data/features/"
    )
    
    if results:
        logger.info("Pipeline completed successfully")
        for model_name, comp in results["comparison"].items():
            better = comp["better"]
            diff = abs(comp["difference"]) * 100
            logger.info(f"{model_name}: {better} models performed better by {diff:.2f}%")
    else:
        logger.error("Pipeline failed to complete")
