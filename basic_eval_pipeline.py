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
                
        for model_name, results in synthetic_results.items():
            if 'accuracy' in results:
                multi_synthetic_results[model_name].append(results['accuracy'])
    
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

    return {
        "original_results": combined_original_results,
        "synthetic_results": combined_synthetic_results,
        "comparison": comparison,
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
            metrics_file_path = os.path.join(model_results_dir, f'{output_prefix}_{model_name}_metrics.json').replace("\\", "/")
            confusion_file_path = os.path.join(model_results_dir, f'{output_prefix}_{model_name}_confusion.png').replace("\\", "/")
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
        model = SVC(**model_params, random_state=42)
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
        # This assumes that run_model accepts (X_train, y_train, X_test, y_test)
        y_pred = run_model(X_train, y_train, X_test, y_test)
except Exception as e:
    import traceback
    print(f"Error using optimized model: {{type(e).__name__}}: {{str(e)}}")
    traceback.print_exc()
    print("Falling back to default model")
    y_pred = run_model(X_train, y_train, X_test, y_test)
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
        n_runs=5                # Number of runs for error bars
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
    else:
        logger.error("Pipeline failed to complete")
