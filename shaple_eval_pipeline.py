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
import matplotlib as mpl
import seaborn as sns
from dotenv import load_dotenv
import sys
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import time
from tqdm import tqdm
import shap  # Added SHAP library

# Set better matplotlib style
plt.style.use('ggplot')
sns.set_style("whitegrid")
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14

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
                            models=None, random_state=42, features_dir="./data/features/",
                            tune_hyperparams=False, n_trials=30, n_runs=5):
    """
    Run a complete evaluation pipeline with the following revised workflow:
      1. Split the original dataset into train (80%) and test (20%).
      2. Train models on the original training data and evaluate them on the test set.
      3. Generate synthetic data using the training set (with n_samples defaulting to len(train_df)).
      4. Train models on the synthetic data and evaluate them on the same test set.
      5. Compare the model performance (evaluated on test data) between the original and synthetic setups.
      
    Added features:
      - Runs multiple evaluations with different random seeds
      - Computes mean and standard deviation for metrics
      - Produces visualizations with error bars
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs("./data/features/", exist_ok=True)
    os.makedirs("./data/chroma_db", exist_ok=True)
    os.makedirs("./data/generated", exist_ok=True)

    # Storage for multi-run results
    multi_original_results = defaultdict(list)
    multi_synthetic_results = defaultdict(list)
    
    # Storage for Shapley value results
    original_shapley_values = {}
    synthetic_shapley_values = {}
    
    # Run multiple evaluations
    logger.info(f"Running {n_runs} evaluations with different random seeds")
    
    for run_idx in tqdm(range(n_runs), desc="Running multiple evaluations"):
        # Set a different random seed for each run
        current_seed = random_state + run_idx
        run_output_dir = os.path.join(output_dir, f"run_{run_idx}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        # Step 1: Load and split original dataset
        logger.info(f"Run {run_idx+1}/{n_runs}: Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path, sep=';')
        
        # Handle 'gender' column if present
        if 'gender' in df.columns:
            df['gender'] = df['gender'].apply(lambda x: 1 if str(x).strip().lower() == 'male' else 0)
        
        if run_idx == 0:  # Only print once
            print(f"First row of loaded data: {df.iloc[0].to_dict()}")
            print(f"Data types: {df.dtypes}")

        label_column = df.columns[-1]
        X = df.drop(label_column, axis=1)
        y = df[label_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=current_seed
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df  = pd.concat([X_test, y_test], axis=1)
        
        # Verify no missing values
        assert test_df.isnull().sum().sum() == 0, "test contains missing values"
        assert train_df.isnull().sum().sum() == 0, "train contains missing values"
        
        train_path = os.path.join(run_output_dir, "data", "train_data.csv")
        test_path  = os.path.join(run_output_dir, "data", "test_data.csv")
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        
        logger.info(f"Saved train data ({len(train_df)} rows) to {train_path}")
        logger.info(f"Saved test data ({len(test_df)} rows) to {test_path}")
        train_df.to_csv(train_path, index=False, sep=';')
        test_df.to_csv(test_path, index=False, sep=';')
        
        # Set default number of synthetic samples if not provided
        current_n_samples = n_samples
        if current_n_samples is None:
            current_n_samples = len(train_df)

        # Step 2: Train & evaluate models on original training data (evaluated on test set)
        logger.info(f"Run {run_idx+1}/{n_runs}: Training and evaluating models on original training data")
        original_results = run_model_evaluations(
            train_df, 
            test_df, 
            output_prefix="original", 
            output_dir=run_output_dir, 
            models=models,
            tune_hyperparams=tune_hyperparams,
            n_trials=n_trials
        )

        # Step 3: Generate synthetic data from the training data
        logger.info(f"Run {run_idx+1}/{n_runs}: Generating {current_n_samples} synthetic data samples")
        synthetic_path = os.path.join(run_output_dir, "data", "synthetic_data.csv")
        analyze_csv_features(train_path, features_dir)

        synthetic_df = generate_data(
            csv_path=train_path,
            n_samples=current_n_samples,
            persist_dir="./data/chroma_db",
            features_dir=features_dir,
            collection_name="dquery",
            output_path=synthetic_path,
            max_workers=5,
            batch_size=100
        )

        if synthetic_df is None or len(synthetic_df) == 0:
            logger.error(f"Run {run_idx+1}/{n_runs}: Failed to generate synthetic data")
            continue
        
        parsed_synthetic_df = parse_synthetic_df(synthetic_df, X_train.columns.tolist()+[label_column])
        logger.info(f"Run {run_idx+1}/{n_runs}: Generated {len(parsed_synthetic_df)} rows of synthetic data")

        # Step 4: Train & evaluate models on synthetic data (evaluated on the same test set)
        logger.info(f"Run {run_idx+1}/{n_runs}: Training and evaluating models on synthetic data")
        synthetic_results = run_model_evaluations(
            parsed_synthetic_df, test_df, 
            output_prefix="synthetic", 
            output_dir=run_output_dir, 
            models=models, 
            tune_hyperparams=tune_hyperparams,
            n_trials=n_trials
        )

        # Store results from this run
        for model_name, results in original_results.items():
            if 'accuracy' in results:
                multi_original_results[model_name].append(results['accuracy'])
                
            # Collect Shapley data if available
            if 'shapley' in results and results['shapley'].get('available', False):
                shapley_path = results['shapley'].get('path')
                if shapley_path and os.path.exists(shapley_path):
                    try:
                        with open(shapley_path, 'r') as f:
                            shapley_data = json.load(f)
                        if model_name not in original_shapley_values:
                            original_shapley_values[model_name] = shapley_data
                    except Exception as e:
                        logger.error(f"Error loading original Shapley data for {model_name}: {str(e)}")
                
        for model_name, results in synthetic_results.items():
            if 'accuracy' in results:
                multi_synthetic_results[model_name].append(results['accuracy'])
                
            # Collect Shapley data if available
            if 'shapley' in results and results['shapley'].get('available', False):
                shapley_path = results['shapley'].get('path')
                if shapley_path and os.path.exists(shapley_path):
                    try:
                        with open(shapley_path, 'r') as f:
                            shapley_data = json.load(f)
                        if model_name not in synthetic_shapley_values:
                            synthetic_shapley_values[model_name] = shapley_data
                    except Exception as e:
                        logger.error(f"Error loading synthetic Shapley data for {model_name}: {str(e)}")
    
    # Combine results from all runs
    combined_original_results = {}
    combined_synthetic_results = {}
    
    for model_name, accuracies in multi_original_results.items():
        if accuracies:
            combined_original_results[model_name] = {
                'accuracy': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'accuracy_values': accuracies
            }
    
    for model_name, accuracies in multi_synthetic_results.items():
        if accuracies:
            combined_synthetic_results[model_name] = {
                'accuracy': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'accuracy_values': accuracies
            }
    
    # Compare and visualize results
    logger.info("Comparing results between models trained on original vs synthetic data")
    comparison = compare_results_with_error_bars(
        combined_original_results, 
        combined_synthetic_results, 
        output_dir
    )
    
    # Create detailed result visualizations
    create_detailed_visualizations(
        combined_original_results, 
        combined_synthetic_results, 
        output_dir
    )
    
    # Create Shapley value visualizations
    logger.info("Creating Shapley value comparison visualizations")
    shapley_dir = os.path.join(output_dir, "shapley_plots")
    os.makedirs(shapley_dir, exist_ok=True)
    
    # Create Shapley visualizations for each model
    for model_name in set(original_shapley_values.keys()) & set(synthetic_shapley_values.keys()):
        logger.info(f"Creating Shapley comparison for {model_name}")
        
        # Create dictionary with needed structure for visualization function
        orig_shapley = {
            "shap_values": None,  # We don't have the actual SHAP values, just the importance
            "explainer": None,
            "feature_importance": original_shapley_values[model_name]["feature_importance"],
            "sample_data": None
        }
        
        synth_shapley = {
            "shap_values": None,
            "explainer": None,
            "feature_importance": synthetic_shapley_values[model_name]["feature_importance"],
            "sample_data": None
        }
        
        # Create comparison visualizations
        visualize_shapley_comparison(orig_shapley, synth_shapley, output_dir, model_name)
    
    # Create consolidated Shapley comparison visualization
    create_consolidated_shapley_visualization(original_shapley_values, synthetic_shapley_values, output_dir)

    return {
        "original_results": combined_original_results,
        "synthetic_results": combined_synthetic_results,
        "comparison": comparison,
        "original_shapley": original_shapley_values,
        "synthetic_shapley": synthetic_shapley_values,
        "n_runs": n_runs
    }

def create_detailed_visualizations(original_results, synthetic_results, output_dir):
    """
    Create more detailed visualizations of the results
    """
    # 1. Violin plots of model accuracies
    plt.figure(figsize=(12, 8))
    
    model_names = []
    orig_data = []
    synth_data = []
    
    for model_name in sorted(set(original_results.keys()) & set(synthetic_results.keys())):
        model_names.append(model_name)
        orig_data.append(original_results[model_name]['accuracy_values'])
        synth_data.append(synthetic_results[model_name]['accuracy_values'])
    
    # Prepare data for violin plots
    plot_data = []
    for i, model in enumerate(model_names):
        for val in orig_data[i]:
            plot_data.append({'Model': model, 'Accuracy': val, 'Data Source': 'Original'})
        for val in synth_data[i]:
            plot_data.append({'Model': model, 'Accuracy': val, 'Data Source': 'Synthetic'})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create violin plot
    plt.figure(figsize=(14, 8))
    ax = sns.violinplot(x='Model', y='Accuracy', hue='Data Source', 
                    data=plot_df, palette='Set2', split=True, inner='quart')
    
    # Add individual data points
    sns.stripplot(x='Model', y='Accuracy', hue='Data Source', data=plot_df, 
                 dodge=True, alpha=0.3, jitter=True, size=4, legend=False)
    
    plt.title('Distribution of Model Accuracies Across Multiple Runs', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title='Data Source', fontsize=12, 
              title_fontsize=13, loc='lower right')
    
    plt.tight_layout()
    violin_path = os.path.join(output_dir, "plots", "accuracy_distributions.png")
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Boxplot comparing original vs synthetic for each model
    plt.figure(figsize=(12, 8))
    
    fig, axes = plt.subplots(nrows=len(model_names), figsize=(12, 3*len(model_names)))
    if len(model_names) == 1:
        axes = [axes]
    
    for i, model in enumerate(model_names):
        data = [original_results[model]['accuracy_values'], 
                synthetic_results[model]['accuracy_values']]
        
        ax = axes[i]
        box = ax.boxplot(data, patch_artist=True, labels=['Original', 'Synthetic'], 
                     widths=0.6, notch=True)
        
        # Add a scatter plot of individual points
        for j, d in enumerate(data):
            # Add jitter to x position
            x = np.random.normal(j+1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.6, s=30, c='black', zorder=3)
        
        # Customize box colors
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(f'{model.upper()} Accuracy Comparison', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylabel('Accuracy')
        
        # Add means as diamond markers
        means = [np.mean(d) for d in data]
        ax.scatter([1, 2], means, marker='D', s=100, c='yellow', 
                  edgecolor='black', zorder=10, label='Mean')
        
        # Add text for mean and std
        for j, (m, d) in enumerate(zip(means, data)):
            std = np.std(d)
            ax.annotate(f'Mean: {m:.4f}\nStd: {std:.4f}', 
                       xy=(j+1, m), xytext=(20, 0), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "plots", "model_boxplots.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Radar chart for model comparison
    plt.figure(figsize=(12, 10))
    
    # Compute model performances as percentage of max possible (normalized to [0,1])
    orig_means = [original_results[m]['accuracy'] for m in model_names]
    synth_means = [synthetic_results[m]['accuracy'] for m in model_names]
    
    # Set up radar chart
    num_models = len(model_names)
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    orig_means += orig_means[:1]  # Close the loop
    synth_means += synth_means[:1]  # Close the loop
    model_names_plot = model_names + [model_names[0]]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, orig_means, 'o-', linewidth=2, label='Original Data', color='#3498db')
    ax.fill(angles, orig_means, alpha=0.25, color='#3498db')
    
    ax.plot(angles, synth_means, 'o-', linewidth=2, label='Synthetic Data', color='#e74c3c')
    ax.fill(angles, synth_means, alpha=0.25, color='#e74c3c')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(model_names)
    
    # Customize radar chart
    ax.set_ylim(min(min(orig_means), min(synth_means)) * 0.9, 
                max(max(orig_means), max(synth_means)) * 1.1)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.title('Model Performance Comparison: Original vs Synthetic', fontsize=16, y=1.08)
    
    radar_path = os.path.join(output_dir, "plots", "radar_comparison.png")
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created detailed visualizations in {output_dir}/plots/")


def compare_results_with_error_bars(original_results, synthetic_results, output_dir):
    """
    Compare results between models trained on original data and synthetic data evaluations.
    Both sets of results are obtained by evaluating on the same test data.
    Includes error bars based on standard deviation across multiple runs.
    """
    comparison = {}
    comparison_path = os.path.join(output_dir, "comparison.csv")
    with open(comparison_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Model', 'Original_Accuracy', 'Original_Std', 'Synthetic_Accuracy', 'Synthetic_Std', 'Difference', 'Better'])
    
    models = []
    original_acc = []
    original_std = []
    synthetic_acc = []
    synthetic_std = []
    
    for model_name in set(original_results.keys()).union(synthetic_results.keys()):
        if model_name in original_results and model_name in synthetic_results:
            try:
                orig_acc = original_results[model_name]['accuracy']
                orig_std = original_results[model_name]['accuracy_std']
                synth_acc = synthetic_results[model_name]['accuracy']
                synth_std = synthetic_results[model_name]['accuracy_std']
                
                difference = synth_acc - orig_acc
                better = "Synthetic" if difference > 0 else "Original" if difference < 0 else "Equal"
                
                comparison[model_name] = {
                    "original_accuracy": orig_acc,
                    "original_std": orig_std,
                    "synthetic_accuracy": synth_acc,
                    "synthetic_std": synth_std,
                    "difference": difference,
                    "better": better
                }
                
                with open(comparison_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([
                        model_name,
                        f"{orig_acc:.4f}",
                        f"{orig_std:.4f}",
                        f"{synth_acc:.4f}",
                        f"{synth_std:.4f}",
                        f"{difference:.4f}",
                        better
                    ])
                
                models.append(model_name)
                original_acc.append(orig_acc)
                original_std.append(orig_std)
                synthetic_acc.append(synth_acc)
                synthetic_std.append(synth_std)
                
            except Exception as e:
                logger.error(f"Error comparing results for {model_name}: {str(e)}")
    
    # Create enhanced bar plot with error bars
    plt.figure(figsize=(14, 9))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars with error bars
    orig_bars = plt.bar(x - width/2, original_acc, width, yerr=original_std, 
                       label='Original Data', color='#3498db', capsize=5,
                       error_kw=dict(elinewidth=2, ecolor='#2c3e50', capthick=2))
    
    synth_bars = plt.bar(x + width/2, synthetic_acc, width, yerr=synthetic_std,
                         label='Synthetic Data', color='#e74c3c', capsize=5,
                         error_kw=dict(elinewidth=2, ecolor='#2c3e50', capthick=2))
    
    # Add text annotations for each bar
    def add_labels(bars, values, stds):
        for bar, val, std in zip(bars, values, stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + std + 0.01,
                   f'{val:.3f}±{std:.3f}', ha='center', va='bottom', 
                   fontsize=9, rotation=0, color='#2c3e50')
    
    add_labels(orig_bars, original_acc, original_std)
    add_labels(synth_bars, synthetic_acc, synthetic_std)
    
    # Add statistical significance indicators
    for i, model in enumerate(models):
        orig_val = original_acc[i]
        orig_err = original_std[i]
        synth_val = synthetic_acc[i]
        synth_err = synthetic_std[i]
        
        # Calculate t-statistic for simple significance test
        # Using a very simple approach here - more sophisticated tests could be used
        if orig_val > synth_val and orig_val - orig_err > synth_val + synth_err:
            # Original significantly better
            plt.text(i, max(orig_val, synth_val) + max(orig_err, synth_err) + 0.02,
                   '*', ha='center', va='bottom', fontsize=24, color='blue')
        elif synth_val > orig_val and synth_val - synth_err > orig_val + orig_err:
            # Synthetic significantly better
            plt.text(i, max(orig_val, synth_val) + max(orig_err, synth_err) + 0.02,
                   '*', ha='center', va='bottom', fontsize=24, color='red')
    
    # Enhance the plot
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy (with Standard Deviation)', fontsize=14)
    plt.title('Model Accuracy Comparison: Original vs Synthetic Data (with Error Bars)', fontsize=16)
    plt.xticks(x, [model.upper() for model in models], rotation=45, fontsize=12)
    plt.legend(fontsize=12)
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Improve y-axis range
    ymin = min([a - e for a, e in zip(original_acc + synthetic_acc, original_std + synthetic_std)])
    ymax = max([a + e for a, e in zip(original_acc + synthetic_acc, original_std + synthetic_std)])
    plt.ylim(max(0, ymin - 0.05), min(1.0, ymax + 0.1))
    
    plt.tight_layout()
    
    comparison_plot_path = os.path.join(output_dir, "plots", "accuracy_comparison_with_error_bars.png")
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison results to {comparison_path}")
    logger.info(f"Saved comparison plot to {comparison_plot_path}")
    
    return comparison

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

def calculate_shapley_values(model, X, max_display=10, feature_names=None):
    """
    Calculate and return SHAP values for a given model and dataset
    
    Args:
        model: Trained model object
        X: Feature data to explain (DataFrame)
        max_display: Maximum number of features to display in plots
        feature_names: Optional list of feature names
        
    Returns:
        dict: Dictionary containing shap values and explainer
    """
    try:
        # Select appropriate explainer based on model type
        model_type = type(model).__name__
        
        if hasattr(model, "predict_proba"):
            has_predict_proba = True
        else:
            has_predict_proba = False
        
        # Create SHAP explainer based on model type
        if model_type == "RandomForestClassifier":
            explainer = shap.TreeExplainer(model)
        elif model_type in ["SVC", "SVR", "LinearSVC", "LinearSVR"]:
            # Use KernelExplainer for SVM models
            if has_predict_proba:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
        elif model_type in ["KNeighborsClassifier", "MLPClassifier", "SGDClassifier"]:
            # Use KernelExplainer for blackbox models
            if has_predict_proba:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
        elif model_type in ["GaussianNB"]:
            # Use KernelExplainer for Naive Bayes
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
        else:
            # Default to KernelExplainer for unknown models
            if has_predict_proba:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
                
        # Calculate SHAP values
        sample_for_shap = X.iloc[:min(len(X), 100)]  # Limit to 100 samples for performance
        shap_values = explainer.shap_values(sample_for_shap)
        
        # Handle shap_values format differences
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For multi-class classification, take the mean abs across classes
            mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=0)
            feature_importance = np.mean(mean_abs_shap, axis=0)
        else:
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for binary classification
            feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance dictionary
        if feature_names is None:
            feature_names = X.columns.tolist()
            
        importance_dict = dict(zip(feature_names, feature_importance))
        
        return {
            "shap_values": shap_values,
            "explainer": explainer,
            "feature_importance": importance_dict,
            "sample_data": sample_for_shap
        }
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {str(e)}")
        return None

def create_consolidated_shapley_visualization(original_shapley_values, synthetic_shapley_values, output_dir):
    """
    Create a consolidated visualization comparing feature importance across models
    
    Args:
        original_shapley_values: Dictionary of Shapley data for original models
        synthetic_shapley_values: Dictionary of Shapley data for synthetic models
        output_dir: Directory to save visualizations
    """
    try:
        # Create directory for shapley plots
        shapley_dir = os.path.join(output_dir, "shapley_plots")
        os.makedirs(shapley_dir, exist_ok=True)
        
        # Check if we have any data to visualize
        if not original_shapley_values or not synthetic_shapley_values:
            logger.warning("No Shapley data available for consolidated visualization")
            return
        
        # Get all models with both original and synthetic data
        common_models = set(original_shapley_values.keys()) & set(synthetic_shapley_values.keys())
        if not common_models:
            logger.warning("No common models with Shapley data for both original and synthetic")
            return
            
        # Get all features across all models
        all_features = set()
        for model in common_models:
            all_features.update(original_shapley_values[model]["feature_importance"].keys())
            all_features.update(synthetic_shapley_values[model]["feature_importance"].keys())
        
        # Limit to top 20 features by average importance across all models
        feature_avg_importance = {}
        for feature in all_features:
            total_importance = 0
            count = 0
            
            # Sum importance from original models
            for model in common_models:
                if feature in original_shapley_values[model]["feature_importance"]:
                    total_importance += original_shapley_values[model]["feature_importance"][feature]
                    count += 1
                    
            # Sum importance from synthetic models
            for model in common_models:
                if feature in synthetic_shapley_values[model]["feature_importance"]:
                    total_importance += synthetic_shapley_values[model]["feature_importance"][feature]
                    count += 1
                    
            if count > 0:
                feature_avg_importance[feature] = total_importance / count
        
        # Sort features by average importance
        top_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        top_feature_names = [f[0] for f in top_features]
        
        # Create a heatmap of feature importance by model
        # Prepare data for the heatmap
        heatmap_data = []
        
        for model in sorted(common_models):
            # Original model
            row_orig = {'Model': f"{model} (Original)"}
            for feature in top_feature_names:
                value = original_shapley_values[model]["feature_importance"].get(feature, 0)
                row_orig[feature] = value
            heatmap_data.append(row_orig)
            
            # Synthetic model
            row_synth = {'Model': f"{model} (Synthetic)"}
            for feature in top_feature_names:
                value = synthetic_shapley_values[model]["feature_importance"].get(feature, 0)
                row_synth[feature] = value
            heatmap_data.append(row_synth)
        
        # Convert to DataFrame for easier visualization
        df_heatmap = pd.DataFrame(heatmap_data)
        df_heatmap = df_heatmap.set_index('Model')
        
        # Create heatmap
        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(df_heatmap, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
        
        # Improve readability of x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance Comparison Across Models', fontsize=16)
        plt.tight_layout()
        
        # Save the heatmap
        heatmap_path = os.path.join(shapley_dir, "consolidated_feature_importance.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a radar chart for the top 10 features
        top_10_features = top_feature_names[:10]
        
        # Function to get importance values for a model
        def get_importance_values(model_data, features):
            return [model_data["feature_importance"].get(f, 0) for f in features]
        
        # Setup radar chart
        angles = np.linspace(0, 2*np.pi, len(top_10_features), endpoint=False).tolist()
        
        # Add the first angle again to close the circle
        angles += angles[:1]
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        
        # Plot each model's feature importance as a separate line
        for i, model in enumerate(sorted(common_models)):
            # Original model values
            values_orig = get_importance_values(original_shapley_values[model], top_10_features)
            values_orig += values_orig[:1]  # Close the loop
            
            # Synthetic model values
            values_synth = get_importance_values(synthetic_shapley_values[model], top_10_features)
            values_synth += values_synth[:1]  # Close the loop
            
            # Plot with different line styles for original vs synthetic
            ax.plot(angles, values_orig, 'o-', linewidth=2, 
                   label=f'{model} (Original)', color=f'C{i}', alpha=0.8)
            ax.plot(angles, values_synth, 's--', linewidth=2, 
                   label=f'{model} (Synthetic)', color=f'C{i}', alpha=0.6)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_10_features, fontsize=12)
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Feature Importance by Model Type (Original vs Synthetic)', fontsize=16, y=1.08)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save the radar chart
        radar_path = os.path.join(shapley_dir, "feature_importance_radar.png")
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created consolidated Shapley visualizations in {shapley_dir}")
        
        # Create table of top features by importance difference
        feature_diff_data = []
        
        for feature in all_features:
            # Calculate average importance in original models
            orig_values = []
            for model in common_models:
                if feature in original_shapley_values[model]["feature_importance"]:
                    orig_values.append(original_shapley_values[model]["feature_importance"][feature])
            
            # Calculate average importance in synthetic models
            synth_values = []
            for model in common_models:
                if feature in synthetic_shapley_values[model]["feature_importance"]:
                    synth_values.append(synthetic_shapley_values[model]["feature_importance"][feature])
            
            # Only include features that appear in both original and synthetic
            if orig_values and synth_values:
                avg_orig = sum(orig_values) / len(orig_values)
                avg_synth = sum(synth_values) / len(synth_values)
                diff = avg_synth - avg_orig
                abs_diff = abs(diff)
                
                feature_diff_data.append({
                    'Feature': feature,
                    'Avg_Original': avg_orig,
                    'Avg_Synthetic': avg_synth,
                    'Difference': diff,
                    'Abs_Difference': abs_diff
                })
        
        # Sort by absolute difference
        feature_diff_data.sort(key=lambda x: x['Abs_Difference'], reverse=True)
        
        # Create DataFrame for top differences
        top_diffs = feature_diff_data[:15]
        df_diffs = pd.DataFrame(top_diffs)
        
        # Create bar chart of top differences
        plt.figure(figsize=(14, 10))
        
        # Plot bars
        features = df_diffs['Feature'].tolist()
        diffs = df_diffs['Difference'].tolist()
        colors = ['#e74c3c' if d > 0 else '#3498db' for d in diffs]
        
        bars = plt.barh(features, diffs, color=colors)
        
        # Add a vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x = width + 0.01 if width > 0 else width - 0.01
            align = 'left' if width > 0 else 'right'
            plt.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{diffs[i]:.4f}', 
                   va='center', ha=align, fontsize=10)
        
        # Set labels and title
        plt.xlabel('Difference in Importance (Synthetic - Original)', fontsize=14)
        plt.title('Top Features by Difference in Importance between Synthetic and Original Models', fontsize=16)
        
        # Add a legend for color meaning
        synthetic_patch = plt.Rectangle((0, 0), 1, 1, fc='#e74c3c')
        original_patch = plt.Rectangle((0, 0), 1, 1, fc='#3498db')
        plt.legend([synthetic_patch, original_patch], ['Higher in Synthetic', 'Higher in Original'], 
                 loc='upper right')
        
        # Add grid
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the difference chart
        diff_path = os.path.join(shapley_dir, "feature_importance_differences.png")
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the difference data to CSV
        diff_csv_path = os.path.join(shapley_dir, "feature_importance_differences.csv")
        df_diffs.to_csv(diff_csv_path, index=False)
        
        logger.info(f"Saved feature importance difference analysis to {diff_path}")
        
    except Exception as e:
        logger.error(f"Error creating consolidated Shapley visualization: {str(e)}")

def visualize_shapley_comparison(original_shapley, synthetic_shapley, output_dir, model_name):
    """
    Create visualizations comparing Shapley values between original and synthetic models
    
    Args:
        original_shapley: Shapley values for original data model
        synthetic_shapley: Shapley values for synthetic data model
        output_dir: Directory to save visualizations
        model_name: Name of the model being compared
    """
    try:
        if original_shapley is None or synthetic_shapley is None:
            logger.warning(f"Missing Shapley values for comparison: Original={original_shapley is not None}, Synthetic={synthetic_shapley is not None}")
            return
        
        # Create directory for shapley plots
        shapley_dir = os.path.join(output_dir, "shapley_plots")
        os.makedirs(shapley_dir, exist_ok=True)
        
        # Get feature importance values
        orig_importance = original_shapley["feature_importance"]
        synth_importance = synthetic_shapley["feature_importance"]
        
        # Ensure we have the same features in both
        common_features = set(orig_importance.keys()).intersection(set(synth_importance.keys()))
        features = list(common_features)
        
        # Sort features by average importance
        features.sort(key=lambda x: (orig_importance.get(x, 0) + synth_importance.get(x, 0))/2, reverse=True)
        features = features[:15]  # Limit to top 15 features
        
        # Get importance values for these features
        orig_values = [orig_importance.get(f, 0) for f in features]
        synth_values = [synth_importance.get(f, 0) for f in features]
        
        # Create bar plot comparing feature importance
        plt.figure(figsize=(12, 10))
        
        x = np.arange(len(features))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        orig_bars = ax.barh(x - width/2, orig_values, width, label='Original Data', color='#3498db')
        synth_bars = ax.barh(x + width/2, synth_values, width, label='Synthetic Data', color='#e74c3c')
        
        ax.set_yticks(x)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Highest values at the top
        
        ax.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=14)
        ax.set_title(f'Feature Importance Comparison: {model_name.upper()}', fontsize=16)
        ax.legend(fontsize=12)
        
        # Add grid lines for readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value annotations
        for i, v in enumerate(orig_values):
            ax.text(v + 0.01, i - width/2, f'{v:.4f}', va='center', fontsize=8)
        
        for i, v in enumerate(synth_values):
            ax.text(v + 0.01, i + width/2, f'{v:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save the figure
        feature_imp_path = os.path.join(shapley_dir, f"feature_importance_{model_name}.png")
        plt.savefig(feature_imp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot comparing feature importances
        plt.figure(figsize=(10, 10))
        
        # Add y=x line
        max_val = max(max(orig_values), max(synth_values)) * 1.1
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        # Plot each feature
        plt.scatter(orig_values, synth_values, s=100, alpha=0.7)
        
        # Add feature names as annotations
        for i, feature in enumerate(features):
            plt.annotate(
                feature, 
                (orig_values[i], synth_values[i]),
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center'
            )
            
        plt.xlabel('Original Data Feature Importance', fontsize=14)
        plt.ylabel('Synthetic Data Feature Importance', fontsize=14)
        plt.title(f'Feature Importance Correlation: {model_name.upper()}', fontsize=16)
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set equal aspect ratio
        plt.axis('equal')
        plt.tight_layout()
        
        # Save the figure
        scatter_path = os.path.join(shapley_dir, f"feature_importance_scatter_{model_name}.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual SHAP summary plots for comparison
        plt.figure(figsize=(14, 8))
        
        # Original data summary plot
        plt.subplot(1, 2, 1)
        if isinstance(original_shapley["shap_values"], list):
            shap.summary_plot(
                original_shapley["shap_values"][0], 
                original_shapley["sample_data"],
                feature_names=features,
                max_display=10,
                show=False,
                plot_size=(6, 8)
            )
        else:
            shap.summary_plot(
                original_shapley["shap_values"], 
                original_shapley["sample_data"],
                feature_names=features,
                max_display=10,
                show=False,
                plot_size=(6, 8)
            )
        plt.title("Original Data SHAP Values", fontsize=14)
        
        # Synthetic data summary plot  
        plt.subplot(1, 2, 2)
        if isinstance(synthetic_shapley["shap_values"], list):
            shap.summary_plot(
                synthetic_shapley["shap_values"][0], 
                synthetic_shapley["sample_data"],
                feature_names=features,
                max_display=10,
                show=False,
                plot_size=(6, 8)
            )
        else:
            shap.summary_plot(
                synthetic_shapley["shap_values"], 
                synthetic_shapley["sample_data"],
                feature_names=features,
                max_display=10, 
                show=False,
                plot_size=(6, 8)
            )
        plt.title("Synthetic Data SHAP Values", fontsize=14)
        
        # Save summary plots
        summary_path = os.path.join(shapley_dir, f"shap_summary_{model_name}.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Shapley comparison plots for {model_name} in {shapley_dir}")
        
    except Exception as e:
        logger.error(f"Error creating Shapley comparison visualizations: {str(e)}")

def run_model_evaluations(train_data, test_data, output_prefix, output_dir="./results", models=None, tune_hyperparams=False, n_trials=30):
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
    
    # Hyperparameter dictionaries for each model
    best_params = {}
    
    # If hyperparameter tuning is enabled (for either original or synthetic data)
    if tune_hyperparams:
        logger.info(f"Running hyperparameter tuning for {output_prefix} data")
        
        # Split training data into train and validation for hyperparameter tuning
        train_label_column = train_data.columns[-1]
        X_tune = train_data.drop(train_label_column, axis=1)
        y_tune = train_data[train_label_column]
        
        # Using a stratified split to maintain class distribution
        from sklearn.model_selection import train_test_split
        X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
            X_tune, y_tune, test_size=0.3, random_state=42, stratify=y_tune
        )
        
        # Tune hyperparameters for each model
        for model_name in models:
            try:
                logger.info(f"Tuning {model_name} for {output_prefix} data")
                best_params[model_name] = tune_hyperparameters(
                    model_name, X_tune_train, y_tune_train, X_tune_val, y_tune_val, n_trials=n_trials
                )
            except Exception as e:
                logger.error(f"Error tuning hyperparameters for {model_name}: {str(e)}")
                best_params[model_name] = {}

    for model_name in models:
        logger.info(f"Running model: {model_name}")
        try:
            temp_module_name = f"temp_model_{model_name}"
            temp_module_path = os.path.join(model_results_dir, f"{temp_module_name}.py")
            
            project_root = os.path.abspath(os.path.dirname(__file__))
            fixed_project_root = project_root.replace("\\", "/")
            fixed_train_path = temp_train_path.replace("\\", "/")
            fixed_test_path = temp_test_path.replace("\\", "/")
            metrics_file_path = os.path.join(model_results_dir, f'{output_prefix}_{model_name}_metrics.json').replace("\\\\", "/")
            confusion_file_path = os.path.join(model_results_dir, f'{output_prefix}_{model_name}_confusion.png').replace("\\\\", "/")
            fixed_metrics_csv_path = metrics_path.replace("\\", "/")
            
            # Add model parameters to the temporary script
            model_params = best_params.get(model_name, {})
            model_params_str = json.dumps(model_params)
            
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
import shap  # Import SHAP

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)
project_root = "{fixed_project_root}"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "evals", "models"))
print("Python path after additions:", sys.path)

# Define model parameters from Optuna tuning
model_params = {model_params_str}
print("Model parameters:", model_params)

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

# For storing the trained model
model = None

# Apply hyperparameters if available
try:
    if model_params and '{model_name}' == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**model_params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using optimized RandomForest with parameters:", model_params)
    elif model_params and '{model_name}' == 'svm':
        from sklearn.svm import SVC
        model = SVC(**model_params, random_state=42, probability=True)  # Enable probability for SHAP
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using optimized SVM with parameters:", model_params)
    elif model_params and '{model_name}' == 'mlp':
        from sklearn.neural_network import MLPClassifier
        # Handle the case where parameter name might be mismatched
        mlp_params = model_params.copy()
        if 'hidden_layer_size' in mlp_params:
            # Convert hidden_layer_size to hidden_layer_sizes tuple
            mlp_params['hidden_layer_sizes'] = (mlp_params.pop('hidden_layer_size'),)
        elif 'hidden_layer_sizes' in mlp_params and not isinstance(mlp_params['hidden_layer_sizes'], tuple):
            # Make sure hidden_layer_sizes is a tuple
            mlp_params['hidden_layer_sizes'] = (mlp_params['hidden_layer_sizes'],)
        
        model = MLPClassifier(**mlp_params, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using optimized MLP with parameters:", mlp_params)
    elif model_params and '{model_name}' == 'sgd':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(**model_params, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using optimized SGD with parameters:", model_params)
    elif model_params and '{model_name}' == 'naivebayes':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using optimized Naive Bayes with parameters:", model_params)
    elif model_params and '{model_name}' == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using optimized KNN with parameters:", model_params)
    else:
        print("No optimized parameters found, using default model")
        # This assumes that run_model returns y_pred
        try:
            # First try to see if run_model returns a tuple of (y_pred, model)
            result = run_model(X_train, y_train, X_test, y_test)
            if isinstance(result, tuple) and len(result) == 2:
                y_pred, model = result
            else:
                y_pred = result
                # If no model is returned, we need to train one for Shapley values
                if '{model_name}' == 'randomforest':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)
                elif '{model_name}' == 'svm':
                    from sklearn.svm import SVC
                    model = SVC(random_state=42, probability=True)
                    model.fit(X_train, y_train)
                elif '{model_name}' == 'mlp':
                    from sklearn.neural_network import MLPClassifier
                    model = MLPClassifier(random_state=42, max_iter=1000)
                    model.fit(X_train, y_train)
                elif '{model_name}' == 'sgd':
                    from sklearn.linear_model import SGDClassifier
                    model = SGDClassifier(random_state=42, max_iter=1000)
                    model.fit(X_train, y_train)
                elif '{model_name}' == 'naivebayes':
                    from sklearn.naive_bayes import GaussianNB
                    model = GaussianNB()
                    model.fit(X_train, y_train)
                elif '{model_name}' == 'knn':
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier()
                    model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error in default run_model execution: {{e}}")
            raise
except Exception as e:
    import traceback
    print(f"Error using optimized model: {{type(e).__name__}}: {{str(e)}}")
    traceback.print_exc()
    print("Falling back to default model")
    # This assumes that run_model returns y_pred
    try:
        # First try to see if run_model returns a tuple of (y_pred, model)
        result = run_model(X_train, y_train, X_test, y_test)
        if isinstance(result, tuple) and len(result) == 2:
            y_pred, model = result
        else:
            y_pred = result
            # If no model is returned, we need to train one for Shapley values
            if '{model_name}' == 'randomforest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
            elif '{model_name}' == 'svm':
                from sklearn.svm import SVC
                model = SVC(random_state=42, probability=True)
                model.fit(X_train, y_train)
            elif '{model_name}' == 'mlp':
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(random_state=42, max_iter=1000)
                model.fit(X_train, y_train)
            elif '{model_name}' == 'sgd':
                from sklearn.linear_model import SGDClassifier
                model = SGDClassifier(random_state=42, max_iter=1000)
                model.fit(X_train, y_train)
            elif '{model_name}' == 'naivebayes':
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB()
                model.fit(X_train, y_train)
            elif '{model_name}' == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier()
                model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error in fallback run_model execution: {{e}}")
        raise
""")
            with open(temp_module_path, 'a') as f:
                f.write(f"""
# Calculate metrics
metrics = {{
    'accuracy': accuracy_score(y_test, y_pred),
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}}

# Calculate SHAP values if we have a model
shapley_file_path = os.path.join('{model_results_dir}', f'{output_prefix}_{model_name}_shapley.json').replace("\\\\", "/")
shapley_plot_path = os.path.join('{model_results_dir}', f'{output_prefix}_{model_name}_shapley_summary.png').replace("\\\\", "/")

if model is not None:
    try:
        print("Calculating SHAP values...")
        
        # Get a limited sample for SHAP calculations
        sample_size = min(100, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        
        # Select appropriate explainer based on model type
        model_type = type(model).__name__
        
        if hasattr(model, "predict_proba"):
            has_predict_proba = True
        else:
            has_predict_proba = False
        
        # Create SHAP explainer based on model type
        if model_type == "RandomForestClassifier":
            explainer = shap.TreeExplainer(model)
        elif model_type in ["SVC", "SVR", "LinearSVC", "LinearSVR"]:
            # Use KernelExplainer for SVM models
            if has_predict_proba:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        elif model_type in ["KNeighborsClassifier", "MLPClassifier", "SGDClassifier"]:
            # Use KernelExplainer for blackbox models
            if has_predict_proba:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        elif model_type in ["GaussianNB"]:
            # Use KernelExplainer for Naive Bayes
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
        else:
            # Default to KernelExplainer for unknown models
            if has_predict_proba:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
                
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle shap_values format differences
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For multi-class classification, take the mean abs across classes
            mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=0)
            feature_importance = np.mean(mean_abs_shap, axis=0)
        else:
            if isinstance(shap_values, list):
                shap_values_for_importance = shap_values[0]  # Take first class for binary classification
            else:
                shap_values_for_importance = shap_values
            feature_importance = np.mean(np.abs(shap_values_for_importance), axis=0)
        
        # Create feature importance dictionary
        feature_names = X_sample.columns.tolist()
        importance_dict = dict(zip(feature_names, feature_importance.tolist()))
        
        # Save top 10 features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_features[:10])
        
        # Generate basic summary plot
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_sample, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(shapley_plot_path)
        plt.close()
        
        # Save Shapley data
        shapley_data = {{
            'feature_importance': importance_dict,
            'top_features': top_features,
            'model_type': model_type
        }}
        
        with open(shapley_file_path, 'w') as f_shapley:
            json.dump(shapley_data, f_shapley, indent=2)
            
        print(f"Saved Shapley values to {{shapley_file_path}}")
        metrics['shapley'] = {{'available': True, 'path': shapley_file_path}}
    except Exception as e:
        print(f"Error calculating SHAP values: {{str(e)}}")
        metrics['shapley'] = {{'available': False, 'error': str(e)}}
else:
    print("No model available for SHAP calculations")
    metrics['shapley'] = {{'available': False, 'error': 'No model available'}}

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

def tune_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=50):
    """
    Tune hyperparameters for a given model using Optuna
    
    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials
        
    Returns:
        dict: Best hyperparameters
    """
    logger.info(f"Tuning hyperparameters for {model_name} using Optuna with {n_trials} trials")
    
    # Define the objective function for different models
    if model_name == "randomforest":
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.5)
            
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            clf.fit(X_train, y_train)
            return clf.score(X_val, y_val)
            
    elif model_name == "svm":
        def objective(trial):
            C = trial.suggest_float('C', 0.01, 100, log=True)
            gamma = trial.suggest_float('gamma', 0.001, 10, log=True)
            
            from sklearn.svm import SVC
            clf = SVC(
                C=C,
                gamma=gamma,
                kernel='rbf',
                random_state=42
            )
            clf.fit(X_train, y_train)
            return clf.score(X_val, y_val)
            
    elif model_name == "mlp":
        def objective(trial):
            hidden_layer_size = trial.suggest_int('hidden_layer_sizes', 5, 100)
            alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
            learning_rate_init = trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True)
            
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(
                hidden_layer_sizes=(hidden_layer_size,),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=1000,
                random_state=42
            )
            clf.fit(X_train, y_train)
            return clf.score(X_val, y_val)
            
    elif model_name == "sgd":
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
            learning_rate = trial.suggest_categorical('learning_rate', ['optimal', 'constant', 'invscaling', 'adaptive'])
            eta0 = trial.suggest_float('eta0', 0.001, 0.1, log=True)
            
            from sklearn.linear_model import SGDClassifier
            clf = SGDClassifier(
                loss='hinge',
                alpha=alpha,
                learning_rate=learning_rate,
                eta0=eta0,
                max_iter=1000,
                random_state=42
            )
            clf.fit(X_train, y_train)
            return clf.score(X_val, y_val)
            
    elif model_name == "naivebayes":
        def objective(trial):
            var_smoothing = trial.suggest_float('var_smoothing', 1e-10, 1e-7, log=True)
            
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB(var_smoothing=var_smoothing)
            clf.fit(X_train, y_train)
            return clf.score(X_val, y_val)
            
    elif model_name == "knn":
        def objective(trial):
            n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights
            )
            clf.fit(X_train, y_train)
            return clf.score(X_val, y_val)
    else:
        logger.warning(f"No hyperparameter tuning defined for {model_name}")
        return {}
    
    # Create and run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best parameters for {model_name}: {study.best_params}")
    logger.info(f"Best score: {study.best_value:.4f}")
    
    return study.best_params

if __name__ == "__main__":
    dataset_path = "./evals/dataset/andrew_diabetes.csv"
    models = ["knn", "mlp", "naivebayes", "randomforest", "sgd", "svm"]
    
    results = run_evaluation_pipeline(
        dataset_path=dataset_path,
        output_dir="./results",
        test_size=0.2,
        n_samples=None,    # Defaults to the size of train data
        models=models,
        random_state=42,
        features_dir="./data/features/",
        tune_hyperparams=True,  # Enable hyperparameter tuning for both original and synthetic
        n_trials=20,            # Number of Optuna trials
        n_runs=1                # Number of runs for error bars (reduced to 1)
    )
    
    if results:
        logger.info("Pipeline completed successfully")
        print("\n=== DETAILED COMPARISON WITH STATISTICAL RESULTS ===")
        print(f"{'Model':<15} {'Original':<18} {'Synthetic':<18} {'Diff':<10} {'Better':<10}")
        print("-" * 75)
        
        for model_name, comp in results["comparison"].items():
            orig_acc = comp["original_accuracy"]
            orig_std = comp["original_std"]
            synth_acc = comp["synthetic_accuracy"]
            synth_std = comp["synthetic_std"]
            diff = synth_acc - orig_acc
            better = comp["better"]
            diff_pct = abs(diff) * 100
            
            orig_str = f"{orig_acc:.4f}±{orig_std:.4f}"
            synth_str = f"{synth_acc:.4f}±{synth_std:.4f}"
            
            print(f"{model_name:<15} {orig_str:<18} {synth_str:<18} {diff:.4f}      {better}")
            
            # Calculate if difference is statistically significant (simple approach)
            if abs(diff) > (orig_std + synth_std):
                sig_str = "statistically significant"
            else:
                sig_str = "not statistically significant"
                
            logger.info(f"{model_name}: {better} models performed better by {diff_pct:.2f}% ({sig_str})")
        
        # Display Shapley information if available
        if "original_shapley" in results and "synthetic_shapley" in results:
            print("\n=== SHAPLEY FEATURE IMPORTANCE COMPARISON ===")
            print(f"Shapley analysis is available in the 'shapley_plots' directory")
            
            # Display top features with the largest difference in importance
            common_models = set(results["original_shapley"].keys()) & set(results["synthetic_shapley"].keys())
            if common_models:
                print("\nTop features with largest importance difference between original and synthetic models:")
                print(f"{'Feature':<20} {'Original':<10} {'Synthetic':<10} {'Diff':<10}")
                print("-" * 70)
                
                # For each model that has Shapley data
                for model in common_models:
                    print(f"\n{model.upper()} MODEL:")
                    orig_shapley = results["original_shapley"][model]["feature_importance"]
                    synth_shapley = results["synthetic_shapley"][model]["feature_importance"]
                    
                    # Get common features
                    common_features = set(orig_shapley.keys()) & set(synth_shapley.keys())
                    
                    # Calculate differences
                    diffs = []
                    for feature in common_features:
                        orig_val = orig_shapley[feature]
                        synth_val = synth_shapley[feature]
                        # Check if values are lists and handle appropriately
                        if isinstance(orig_val, list) and isinstance(synth_val, list):
                            # If both are lists of the same length, calculate element-wise difference
                            if len(orig_val) == len(synth_val):
                                diff = sum(s - o for s, o in zip(synth_val, orig_val)) / len(orig_val)
                            else:
                                # If different lengths, use average values
                                diff = sum(synth_val) / len(synth_val) - sum(orig_val) / len(orig_val)
                        else:
                            # For scalar values, simple subtraction
                            diff = synth_val - orig_val
                        diffs.append((feature, orig_val, synth_val, diff))
                    
                    # Sort by absolute difference
                    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
                    
                    # Print top 5 differences
                    for feature, orig_val, synth_val, diff in diffs[:5]:
                        # Format the values properly for display
                        if isinstance(orig_val, list):
                            orig_val_display = f"{sum(orig_val)/len(orig_val):.4f}"
                        else:
                            orig_val_display = f"{orig_val:.4f}"
                            
                        if isinstance(synth_val, list):
                            synth_val_display = f"{sum(synth_val)/len(synth_val):.4f}"
                        else:
                            synth_val_display = f"{synth_val:.4f}"
                            
                        print(f"{feature:<20} {orig_val_display:<10} {synth_val_display:<10} {diff:<10.4f}")
    else:
        logger.error("Pipeline failed to complete")
