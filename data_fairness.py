#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from pathlib import Path

# For fairness metrics
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set better matplotlib style
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Define a custom color palette for visualizations
COLORS = {
    'synthetic': '#2980b9',  # Blue shade
    'original': '#27ae60',   # Green shade
    'male': '#3498db',       # Light blue
    'female': '#e74c3c',     # Red shade
    'difference': '#9b59b6', # Purple shade
    'threshold': '#f39c12',  # Orange shade
    'fair': '#2ecc71',       # Green
    'unfair': '#e74c3c'      # Red
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fairness metrics on synthetic data")
    parser.add_argument("--original", type=str, required=False, help="Path to original dataset CSV")
    parser.add_argument("--synthetic", type=str, required=True, help="Path to synthetic dataset CSV")
    parser.add_argument("--output-dir", type=str, default="./fairness_results", help="Directory to save results")
    parser.add_argument("--separator", type=str, default=";", help="CSV separator")
    parser.add_argument("--protected-attribute", type=str, default="gender", 
                        help="Protected attribute to analyze for fairness")
    return parser.parse_args()

def load_and_preprocess_data(file_path, separator=";"):
    """
    Load and preprocess the dataset.
    
    Args:
        file_path: Path to the dataset
        separator: CSV separator
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    
    # Read CSV with more robust NA handling
    df = pd.read_csv(file_path, sep=separator, na_values=['', 'NA', 'N/A', 'nan', 'NaN', 'None', '?'])
    
    # Print column information and data types
    logger.info("Column information before preprocessing:")
    for col in df.columns:
        logger.info(f"Column '{col}': dtype={df[col].dtype}, unique values={len(df[col].unique())}, null count={df[col].isna().sum()}")
    
    # Handle NA values - AIF360's BinaryLabelDataset doesn't accept NA values
    na_count_before = df.isna().sum().sum()
    if na_count_before > 0:
        logger.warning(f"Found {na_count_before} NA values in the dataset")
        
        # For categorical columns, fill NA with the most frequent value
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                most_freq = df[col].mode()[0]
                df[col] = df[col].fillna(most_freq)
                logger.info(f"Filled NA values in column '{col}' with most frequent value: {most_freq}")
        
        # For numeric columns, fill NA with the median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled NA values in column '{col}' with median: {median_val}")
        
        # Check if any NA values remain
        na_count_after = df.isna().sum().sum()
        if na_count_after > 0:
            logger.warning(f"Still have {na_count_after} NA values after filling. Dropping rows with NA values.")
            df = df.dropna()
            logger.info(f"Dropped rows with NA values. New shape: {df.shape}")
    
    # Ensure all columns are properly typed for AIF360
    # Convert all string/object columns that should be numeric
    for col in df.columns:
        if col != 'gender' and df[col].dtype == 'object':
            try:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col])
                logger.info(f"Converted column '{col}' to numeric")
            except:
                # If conversion fails, keep as is but ensure no empty values
                df[col] = df[col].fillna(df[col].mode()[0])
                logger.info(f"Kept column '{col}' as categorical and filled any remaining empty values")
    
    # Convert gender to string categorical values if it exists as numeric
    if 'gender' in df.columns:
        # First check if gender has any problematic values
        unique_gender_values = df['gender'].unique()
        logger.info(f"Unique gender values before mapping: {unique_gender_values}")
        
        # Map gender values consistently
        gender_mapping = {0: "female", 1: "male", "0": "female", "1": "male", 
                         "female": "female", "male": "male", "Female": "female", "Male": "male",
                         "f": "female", "m": "male", "F": "female", "M": "male"}
        
        df['gender'] = df['gender'].map(lambda x: gender_mapping.get(x, x))
        
        # Check if there are any unmapped values
        unmapped_values = [v for v in df['gender'].unique() if v not in ["female", "male"]]
        if unmapped_values:
            logger.warning(f"Found unmapped gender values: {unmapped_values}. Converting to most frequent value.")
            most_freq = df['gender'].value_counts().index[0]
            df['gender'] = df['gender'].map(lambda x: x if x in ["female", "male"] else most_freq)
        
        logger.info(f"Unique gender values after mapping: {df['gender'].unique()}")
    
    # Make a final check for any NaN values
    if df.isna().any().any():
        logger.warning("Still found NA values after preprocessing. Dropping these rows.")
        df = df.dropna()
        logger.info(f"Final shape after dropping all NA values: {df.shape}")
    
    logger.info(f"Data shape: {df.shape}")
    return df

def create_aif360_dataset(df, protected_attribute="gender", label_name="class"):
    """
    Convert a pandas DataFrame to an aif360 BinaryLabelDataset.
    
    Args:
        df: Input DataFrame
        protected_attribute: Name of the protected attribute column
        label_name: Name of the label column
        
    Returns:
        aif360 BinaryLabelDataset
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Verify all columns are numeric except protected attribute
    for col in df_copy.columns:
        if col != protected_attribute and df_copy[col].dtype == object:
            logger.warning(f"Column {col} is not numeric. Converting to numeric.")
            # Try to convert to numeric
            try:
                df_copy[col] = pd.to_numeric(df_copy[col])
            except:
                logger.error(f"Failed to convert column {col} to numeric. This may cause issues.")
    
    # Identify privileged and unprivileged groups
    if protected_attribute == "gender":
        # Map gender to binary values if it's not already
        if df_copy[protected_attribute].dtype == object:
            logger.info(f"Converting gender values to binary. Current values: {df_copy[protected_attribute].unique()}")
            # Ensure all values are either 'female' or 'male'
            valid_mask = df_copy[protected_attribute].isin(['female', 'male'])
            if not valid_mask.all():
                logger.warning(f"Found invalid gender values. Setting to most frequent.")
                most_freq = df_copy[protected_attribute].value_counts().index[0]
                df_copy.loc[~valid_mask, protected_attribute] = most_freq
            
            df_copy[protected_attribute] = df_copy[protected_attribute].map({"female": 0, "male": 1})
            logger.info(f"Gender values after conversion: {df_copy[protected_attribute].unique()}")
            
            privileged_groups = [{'gender': 1}]  # male
            unprivileged_groups = [{'gender': 0}]  # female
        else:
            # Ensure gender values are either 0 or 1
            valid_mask = df_copy[protected_attribute].isin([0, 1])
            if not valid_mask.all():
                logger.warning(f"Found invalid numeric gender values. Setting to binary.")
                df_copy.loc[~valid_mask, protected_attribute] = df_copy[protected_attribute].map(lambda x: 1 if x > 0.5 else 0)
            
            privileged_groups = [{'gender': 1}]  # male
            unprivileged_groups = [{'gender': 0}]  # female
    else:
        # For other protected attributes, we need to determine the privileged group
        unique_values = df_copy[protected_attribute].unique()
        if len(unique_values) == 2:
            # Ensure values are 0 and 1
            if not set(unique_values).issubset({0, 1}):
                logger.warning(f"Converting protected attribute values to binary. Current values: {unique_values}")
                # Map smallest value to 0, largest to 1
                min_val, max_val = min(unique_values), max(unique_values)
                df_copy[protected_attribute] = df_copy[protected_attribute].map({min_val: 0, max_val: 1})
            
            privileged_groups = [{protected_attribute: 1}]
            unprivileged_groups = [{protected_attribute: 0}]
        else:
            logger.warning(f"Protected attribute {protected_attribute} has more than two values: {unique_values}. Using first value as unprivileged.")
            # Convert to binary - first value as 0, all others as 1
            first_value = unique_values[0]
            df_copy[protected_attribute] = df_copy[protected_attribute].map(lambda x: 0 if x == first_value else 1)
            privileged_groups = [{protected_attribute: 1}]
            unprivileged_groups = [{protected_attribute: 0}]
    
    # Ensure label is binary (0 or 1)
    unique_labels = df_copy[label_name].unique()
    if not set(unique_labels).issubset({0, 1}):
        logger.warning(f"Label values are not binary: {unique_labels}. Converting to binary.")
        # Map smallest value to 0, others to 1 if more than two values
        if len(unique_labels) > 2:
            min_val = min(unique_labels)
            df_copy[label_name] = df_copy[label_name].map(lambda x: 0 if x == min_val else 1)
        else:
            # If exactly two values, map smallest to 0, largest to 1
            min_val, max_val = min(unique_labels), max(unique_labels)
            df_copy[label_name] = df_copy[label_name].map({min_val: 0, max_val: 1})
    
    # Final check for any NaN values
    if df_copy.isna().any().any():
        logger.error("Found NaN values before creating AIF360 dataset. This will cause an error.")
        na_cols = df_copy.columns[df_copy.isna().any()].tolist()
        logger.error(f"Columns with NaN values: {na_cols}")
        for col in na_cols:
            na_count = df_copy[col].isna().sum()
            logger.error(f"Column '{col}' has {na_count} NaN values")
        
        # Drop NaN values as a last resort
        df_copy = df_copy.dropna()
        logger.warning(f"Dropped rows with NaN values. New shape: {df_copy.shape}")
    
    # Convert all datatypes to numeric for AIF360
    for col in df_copy.columns:
        if not np.issubdtype(df_copy[col].dtype, np.number):
            logger.warning(f"Column {col} has non-numeric dtype: {df_copy[col].dtype}. Attempting conversion.")
            try:
                df_copy[col] = pd.to_numeric(df_copy[col])
            except:
                logger.error(f"Failed to convert {col} to numeric. This will likely cause an error.")
    
    logger.info(f"Creating AIF360 dataset with shape: {df_copy.shape}")
    
    try:
        # Convert to aif360 dataset format
        aif_dataset = BinaryLabelDataset(
            df=df_copy,
            label_names=[label_name],
            protected_attribute_names=[protected_attribute],
            privileged_protected_attributes=[1],  # 1 is male, 0 is female
            favorable_label=1,  # positive outcome (has the condition)
            unfavorable_label=0   # negative outcome (doesn't have the condition)
        )
        logger.info("Successfully created AIF360 dataset")
        return aif_dataset, privileged_groups, unprivileged_groups
    except Exception as e:
        logger.error(f"Error creating AIF360 dataset: {e}")
        # Print the first few rows to help diagnose
        logger.error(f"First 5 rows of processed dataframe:\n{df_copy.head()}")
        logger.error(f"DataFrame info:\n{df_copy.info()}")
        raise

def compute_fairness_metrics(dataset, privileged_groups, unprivileged_groups):
    """
    Compute fairness metrics on the dataset.
    
    Args:
        dataset: aif360 BinaryLabelDataset
        privileged_groups: List of privileged group dictionaries
        unprivileged_groups: List of unprivileged group dictionaries
        
    Returns:
        Dictionary with fairness metrics
    """
    metrics = {}
    
    # Create a metric object
    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # Calculate fairness metrics
    metrics['disparate_impact'] = metric.disparate_impact()
    metrics['statistical_parity_difference'] = metric.statistical_parity_difference()
    metrics['group_fairness'] = metric.consistency()[0]
    
    return metrics

def train_and_evaluate_model(dataset, privileged_groups, unprivileged_groups):
    """
    Train a simple model on the dataset and evaluate its fairness.
    
    Args:
        dataset: aif360 BinaryLabelDataset
        privileged_groups: List of privileged group dictionaries
        unprivileged_groups: List of unprivileged group dictionaries
        
    Returns:
        Dictionary with model performance and fairness metrics
    """
    results = {}
    
    # Split the dataset
    train_data = dataset
    
    # Get the features and labels
    X = train_data.features
    y = train_data.labels.ravel()
    
    # Standardize features
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # Train a logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate performance metrics
    results['accuracy'] = accuracy_score(y, y_pred)
    
    # Create a dataset with predictions
    pred_dataset = dataset.copy()
    pred_dataset.labels = y_pred.reshape(-1, 1)
    
    # Calculate fairness metrics
    metric = ClassificationMetric(
        dataset=dataset,
        classified_dataset=pred_dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # Calculate fairness metrics
    results['equal_opportunity_difference'] = metric.equal_opportunity_difference()
    results['average_odds_difference'] = metric.average_odds_difference()
    results['disparate_impact'] = metric.disparate_impact()
    results['statistical_parity_difference'] = metric.statistical_parity_difference()
    
    # Group-specific metrics
    results['privileged_group_metrics'] = {
        'true_positive_rate': metric.true_positive_rate(privileged=True),
        'false_positive_rate': metric.false_positive_rate(privileged=True),
        'true_negative_rate': metric.true_negative_rate(privileged=True),
        'false_negative_rate': metric.false_negative_rate(privileged=True)
    }
    
    results['unprivileged_group_metrics'] = {
        'true_positive_rate': metric.true_positive_rate(privileged=False),
        'false_positive_rate': metric.false_positive_rate(privileged=False),
        'true_negative_rate': metric.true_negative_rate(privileged=False),
        'false_negative_rate': metric.false_negative_rate(privileged=False)
    }
    
    return results

def plot_fairness_metrics(metrics, output_dir, compare=False):
    """
    Create visualizations of fairness metrics.
    
    Args:
        metrics: Dictionary with fairness metrics
        output_dir: Directory to save visualizations
        compare: Whether we're comparing original vs synthetic data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if compare:
        # Comparison plot for disparate impact and statistical parity
        plt.figure(figsize=(12, 8))
        
        # Data to plot
        labels = ['Disparate Impact', 'Statistical Parity Difference']
        
        original_values = [
            metrics['original']['dataset_metrics']['disparate_impact'],
            metrics['original']['dataset_metrics']['statistical_parity_difference']
        ]
        
        synthetic_values = [
            metrics['synthetic']['dataset_metrics']['disparate_impact'],
            metrics['synthetic']['dataset_metrics']['statistical_parity_difference']
        ]
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        
        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(x - width/2, original_values, width, label='Original', color=COLORS['original'], alpha=0.8)
        rects2 = ax.bar(x + width/2, synthetic_values, width, label='Synthetic', color=COLORS['synthetic'], alpha=0.8)
        
        # Add fairness thresholds
        if x[0] >= 0:  # For disparate impact
            ax.axhline(y=1, xmin=x[0]-width, xmax=x[0]+width, 
                      color=COLORS['threshold'], linestyle='--', alpha=0.7, 
                      label='Fairness Threshold (DI=1)')
        if x[1] >= 0:  # For statistical parity
            ax.axhline(y=0, xmin=x[1]-width, xmax=x[1]+width, 
                      color=COLORS['threshold'], linestyle='--', alpha=0.7, 
                      label='Fairness Threshold (SPD=0)')
        
        # Customize plot
        ax.set_ylabel('Metric Value', fontsize=14)
        ax.set_title('Fairness Metrics Comparison: Original vs Synthetic', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=12)
        
        # Add value labels on bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
        
        add_labels(rects1)
        add_labels(rects2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fairness_metrics_comparison.png'), dpi=300)
        plt.close()
        
        # Plot differences in metrics
        plt.figure(figsize=(10, 6))
        
        # Calculate absolute differences
        diff_di = abs(original_values[0] - synthetic_values[0])
        diff_spd = abs(original_values[1] - synthetic_values[1])
        
        # Create bar chart for differences
        diff_metrics = {
            'Disparate Impact': diff_di,
            'Statistical Parity Difference': diff_spd
        }
        
        # Decide colors based on how close the metrics are
        di_color = COLORS['fair'] if diff_di < 0.1 else COLORS['unfair']
        spd_color = COLORS['fair'] if diff_spd < 0.05 else COLORS['unfair']
        colors = [di_color, spd_color]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(diff_metrics.keys(), diff_metrics.values(), color=colors, alpha=0.8)
        
        # Add threshold line for acceptable difference
        plt.axhline(y=0.1, color=COLORS['threshold'], linestyle='--', alpha=0.7, 
                    label='Acceptable Difference Threshold')
        
        # Add labels
        for i, v in enumerate(diff_metrics.values()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)
        
        plt.title('Absolute Differences in Fairness Metrics\nOriginal vs Synthetic', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Absolute Difference', fontsize=14)
        plt.ylim(0, max(list(diff_metrics.values()) + [0.15]))  # Add some room at the top
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fairness_metrics_differences.png'), dpi=300)
        plt.close()
        
        # Compare group statistics
        if 'group_stats' in metrics['original'] and 'group_stats' in metrics['synthetic']:
            # Compare positive outcome rates
            plt.figure(figsize=(12, 7))
            
            orig_rates = metrics['original']['group_stats']['positive_outcome_rates']
            synth_rates = metrics['synthetic']['group_stats']['positive_outcome_rates']
            
            groups = list(orig_rates.keys())
            
            # Get rates for each group from both datasets
            orig_values = [orig_rates[g] for g in groups]
            synth_values = [synth_rates[g] for g in groups]
            
            x = np.arange(len(groups))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 7))
            rects1 = ax.bar(x - width/2, orig_values, width, label='Original', color=COLORS['original'], alpha=0.8)
            rects2 = ax.bar(x + width/2, synth_values, width, label='Synthetic', color=COLORS['synthetic'], alpha=0.8)
            
            # Customize plot
            ax.set_ylabel('Positive Outcome Rate', fontsize=14)
            ax.set_title('Positive Outcome Rates by Group: Original vs Synthetic', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(groups, fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=12)
            
            # Add value labels
            add_labels(rects1)
            add_labels(rects2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'positive_rates_comparison.png'), dpi=300)
            plt.close()
            
            # Compare group proportions
            plt.figure(figsize=(12, 7))
            
            orig_counts = metrics['original']['group_stats']['group_percentages']
            synth_counts = metrics['synthetic']['group_stats']['group_percentages']
            
            groups = list(orig_counts.keys())
            
            # Get percentages for each group from both datasets
            orig_pct = [orig_counts[g] for g in groups]
            synth_pct = [synth_counts[g] for g in groups]
            
            x = np.arange(len(groups))
            
            fig, ax = plt.subplots(figsize=(12, 7))
            rects1 = ax.bar(x - width/2, orig_pct, width, label='Original', color=COLORS['original'], alpha=0.8)
            rects2 = ax.bar(x + width/2, synth_pct, width, label='Synthetic', color=COLORS['synthetic'], alpha=0.8)
            
            # Customize plot
            ax.set_ylabel('Percentage (%)', fontsize=14)
            ax.set_title('Group Distribution: Original vs Synthetic', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(groups, fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=12)
            
            # Add value labels
            for rect in rects1:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
            
            for rect in rects2:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'group_distribution_comparison.png'), dpi=300)
            plt.close()
            
    else:
        # For single dataset analysis (original code with improved style)
        plt.figure(figsize=(10, 7))
        metrics_to_plot = {
            'Disparate Impact': metrics['dataset_metrics']['disparate_impact'],
            'Statistical Parity Difference': metrics['dataset_metrics']['statistical_parity_difference']
        }
        
        # Create bar chart with gradient colors
        bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), 
                      color=[COLORS['fair'] if 0.8 <= metrics['dataset_metrics']['disparate_impact'] <= 1.2 else COLORS['unfair'],
                             COLORS['fair'] if abs(metrics['dataset_metrics']['statistical_parity_difference']) <= 0.1 else COLORS['unfair']])
        
        # Add a horizontal line at y=1 for disparate impact and y=0 for statistical parity
        plt.axhline(y=1, color=COLORS['threshold'], linestyle='--', alpha=0.7, label='Fairness Threshold (DI=1)')
        plt.axhline(y=0, color=COLORS['threshold'], linestyle='--', alpha=0.7, label='Fairness Threshold (SPD=0)')
        
        # Add value labels
        for i, v in enumerate(metrics_to_plot.values()):
            plt.text(i, v + 0.02 * (1 if v >= 0 else -1), f'{v:.3f}', ha='center', fontsize=11)
        
        plt.title('Key Fairness Metrics', fontsize=16, fontweight='bold')
        plt.ylabel('Metric Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'key_fairness_metrics.png'), dpi=300)
        plt.close()
        
        # For model fairness metrics
        if 'model_metrics' in metrics:
            model_metrics = metrics['model_metrics']
            
            # Plot TPR and FPR by group with improved design
            metrics_by_group = {
                'True Positive Rate': [
                    model_metrics['privileged_group_metrics']['true_positive_rate'],
                    model_metrics['unprivileged_group_metrics']['true_positive_rate']
                ],
                'False Positive Rate': [
                    model_metrics['privileged_group_metrics']['false_positive_rate'],
                    model_metrics['unprivileged_group_metrics']['false_positive_rate']
                ],
                'True Negative Rate': [
                    model_metrics['privileged_group_metrics']['true_negative_rate'],
                    model_metrics['unprivileged_group_metrics']['true_negative_rate']
                ],
                'False Negative Rate': [
                    model_metrics['privileged_group_metrics']['false_negative_rate'],
                    model_metrics['unprivileged_group_metrics']['false_negative_rate']
                ]
            }
            
            # Set up the figure with subplots in a more attractive layout
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Plot each metric with improved styling
            for i, (metric_name, values) in enumerate(metrics_by_group.items()):
                # Create more visually appealing bars
                bars = axes[i].bar(['Privileged (Male)', 'Unprivileged (Female)'], 
                                  values, color=[COLORS['male'], COLORS['female']], alpha=0.8)
                
                # Set title and labels
                axes[i].set_title(metric_name, fontsize=14, fontweight='bold')
                axes[i].set_ylabel('Rate', fontsize=12)
                
                # Remove top and right spines for cleaner look
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                
                # Add the actual values as text on the bars
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', fontsize=11)
                
                # Set consistent y-axis limits for better comparison
                axes[i].set_ylim(0, max(1.0, max(values) * 1.15))
            
            # Add a title for the entire figure
            fig.suptitle('Model Performance Metrics by Group', fontsize=18, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
            plt.savefig(os.path.join(output_dir, 'model_metrics_by_group.png'), dpi=300)
            plt.close()
            
            # Plot fairness metrics with improved style
            plt.figure(figsize=(12, 7))
            model_fairness_metrics = {
                'Equal Opportunity\nDifference': model_metrics['equal_opportunity_difference'],
                'Average Odds\nDifference': model_metrics['average_odds_difference'],
                'Statistical Parity\nDifference': model_metrics['statistical_parity_difference']
            }
            
            # Determine color based on fairness (green if fair, red if not)
            colors = [
                COLORS['fair'] if abs(model_metrics['equal_opportunity_difference']) <= 0.1 else COLORS['unfair'],
                COLORS['fair'] if abs(model_metrics['average_odds_difference']) <= 0.1 else COLORS['unfair'],
                COLORS['fair'] if abs(model_metrics['statistical_parity_difference']) <= 0.1 else COLORS['unfair']
            ]
            
            bars = plt.bar(model_fairness_metrics.keys(), model_fairness_metrics.values(), color=colors, alpha=0.8)
            
            # Add a horizontal line at y=0 for fairness
            plt.axhline(y=0, color=COLORS['threshold'], linestyle='--', alpha=0.7, label='Fairness Threshold')
            
            # Add value labels
            for i, v in enumerate(model_fairness_metrics.values()):
                plt.text(i, v + 0.02 * (1 if v >= 0 else -1), f'{v:.3f}', ha='center', fontsize=11)
            
            plt.title('Model Fairness Metrics', fontsize=16, fontweight='bold')
            plt.ylabel('Metric Value', fontsize=14)
            plt.legend(fontsize=12)
            
            # Remove top and right spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'model_fairness_metrics.png'), dpi=300)
            plt.close()

def analyze_group_statistics(df, protected_attribute="gender", label_name="class"):
    """
    Analyze statistics by protected attribute group.
    
    Args:
        df: DataFrame with the data
        protected_attribute: Name of the protected attribute column
        label_name: Name of the label column
        
    Returns:
        Dictionary with group statistics
    """
    stats = {}
    
    # Ensure gender is a string category
    if protected_attribute == "gender" and df[protected_attribute].dtype != object:
        gender_mapping = {0: "female", 1: "male"}
        df[protected_attribute] = df[protected_attribute].map(lambda x: gender_mapping.get(x, x))
    
    # Group statistics
    group_counts = df[protected_attribute].value_counts()
    stats['group_counts'] = group_counts.to_dict()
    stats['group_percentages'] = (group_counts / len(df) * 100).to_dict()
    
    # Positive outcome rates by group
    positive_rates = df.groupby(protected_attribute)[label_name].mean()
    stats['positive_outcome_rates'] = positive_rates.to_dict()
    
    # Difference in positive outcome rates
    if len(positive_rates) == 2:
        groups = list(positive_rates.index)
        stats['positive_rate_difference'] = positive_rates[groups[0]] - positive_rates[groups[1]]
    
    return stats

def plot_group_statistics(stats, output_dir):
    """
    Create visualizations of group statistics.
    
    Args:
        stats: Dictionary with group statistics
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot group counts with improved design
    plt.figure(figsize=(10, 7))
    
    # Get groups and counts
    groups = list(stats['group_counts'].keys())
    counts = list(stats['group_counts'].values())
    
    # Create visually appealing bar chart
    colors = [COLORS['male'] if g.lower() == 'male' else COLORS['female'] for g in groups]
    bars = plt.bar(groups, counts, color=colors, alpha=0.8)
    
    # Add count labels on bars
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha='center', fontsize=11, fontweight='bold')
    
    # Customize plot
    plt.title('Protected Attribute Group Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('Count', fontsize=14)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'group_counts.png'), dpi=300)
    plt.close()
    
    # Plot positive outcome rates by group with improved design
    plt.figure(figsize=(10, 7))
    
    # Get groups and rates
    groups = list(stats['positive_outcome_rates'].keys())
    rates = list(stats['positive_outcome_rates'].values())
    
    # Create visually appealing bar chart
    colors = [COLORS['male'] if g.lower() == 'male' else COLORS['female'] for g in groups]
    bars = plt.bar(groups, rates, color=colors, alpha=0.8)
    
    # Add rate labels on bars
    for i, v in enumerate(rates):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # Customize plot
    plt.title('Positive Outcome Rate by Group', fontsize=16, fontweight='bold')
    plt.ylabel('Positive Outcome Rate', fontsize=14)
    plt.ylim(0, max(rates) * 1.15)  # Add some space for the text
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line to show overall average 
    overall_rate = sum(rates) / len(rates) if rates else 0
    plt.axhline(y=overall_rate, color='gray', linestyle='--', alpha=0.7, 
               label=f'Overall Average: {overall_rate:.3f}')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positive_outcome_rates.png'), dpi=300)
    plt.close()
    
    # Add a radar chart for comparing multiple metrics between groups if more than one group
    if len(groups) >= 2:
        # Create a more creative visualization - radar chart to compare groups
        metrics = ['Positive Rate', 'Representation', 'Impact']
        
        # Calculate metrics for radar chart
        group_data = {}
        for i, group in enumerate(groups):
            group_data[group] = [
                stats['positive_outcome_rates'][group],  # Positive rate
                stats['group_percentages'][group] / 100,  # Representation (normalized)
                stats['positive_outcome_rates'][group] * (stats['group_percentages'][group] / 100)  # Impact
            ]
        
        # Number of variables
        N = len(metrics)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], metrics, fontsize=12)
        
        # Draw the chart for each group
        for i, group in enumerate(groups):
            values = group_data[group]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=group, 
                   color=COLORS['male'] if group.lower() == 'male' else COLORS['female'])
            ax.fill(angles, values, alpha=0.1, 
                   color=COLORS['male'] if group.lower() == 'male' else COLORS['female'])
        
        # Customize chart
        plt.title('Group Comparison Across Multiple Metrics', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'group_radar_chart.png'), dpi=300)
        plt.close()

def main():
    """Main function to run fairness evaluation."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load synthetic data
    synthetic_df = load_and_preprocess_data(args.synthetic, args.separator)
    
    # Dictionary to store all results
    all_results = {}
    
    # Analyze synthetic data
    logger.info(f"Analyzing fairness of synthetic data with respect to {args.protected_attribute}")
    
    # Create aif360 dataset for synthetic data
    synthetic_aif, privileged_groups, unprivileged_groups = create_aif360_dataset(
        synthetic_df, 
        protected_attribute=args.protected_attribute
    )
    
    # Compute dataset fairness metrics for synthetic data
    synthetic_fairness_metrics = compute_fairness_metrics(
        synthetic_aif, 
        privileged_groups, 
        unprivileged_groups
    )
    
    # Train and evaluate a model on synthetic data
    synthetic_model_metrics = train_and_evaluate_model(
        synthetic_aif,
        privileged_groups,
        unprivileged_groups
    )
    
    # Analyze group statistics for synthetic data
    synthetic_group_stats = analyze_group_statistics(
        synthetic_df,
        protected_attribute=args.protected_attribute
    )
    
    # Store synthetic results
    all_results['synthetic'] = {
        'dataset_metrics': synthetic_fairness_metrics,
        'model_metrics': synthetic_model_metrics,
        'group_stats': synthetic_group_stats
    }
    
    # Check if original data is provided
    if args.original:
        logger.info(f"Original data provided, analyzing fairness of original data")
        
        # Load original data
        original_df = load_and_preprocess_data(args.original, args.separator)
        
        # Create aif360 dataset for original data
        original_aif, _, _ = create_aif360_dataset(
            original_df, 
            protected_attribute=args.protected_attribute
        )
        
        # Compute dataset fairness metrics for original data
        original_fairness_metrics = compute_fairness_metrics(
            original_aif, 
            privileged_groups, 
            unprivileged_groups
        )
        
        # Train and evaluate a model on original data
        original_model_metrics = train_and_evaluate_model(
            original_aif,
            privileged_groups,
            unprivileged_groups
        )
        
        # Analyze group statistics for original data
        original_group_stats = analyze_group_statistics(
            original_df,
            protected_attribute=args.protected_attribute
        )
        
        # Store original results
        all_results['original'] = {
            'dataset_metrics': original_fairness_metrics,
            'model_metrics': original_model_metrics,
            'group_stats': original_group_stats
        }
        
        # Plot comparison between original and synthetic
        logger.info("Generating comparison visualizations")
        plot_fairness_metrics(all_results, os.path.join(args.output_dir, 'plots'), compare=True)
        
        # Log comparison results
        logger.info("\nFairness Metrics Comparison:")
        logger.info(f"Disparate Impact (Original): {original_fairness_metrics['disparate_impact']:.4f}")
        logger.info(f"Disparate Impact (Synthetic): {synthetic_fairness_metrics['disparate_impact']:.4f}")
        logger.info(f"Statistical Parity Difference (Original): {original_fairness_metrics['statistical_parity_difference']:.4f}")
        logger.info(f"Statistical Parity Difference (Synthetic): {synthetic_fairness_metrics['statistical_parity_difference']:.4f}")
        
        # Calculate preservation of fairness metrics
        di_diff = abs(original_fairness_metrics['disparate_impact'] - synthetic_fairness_metrics['disparate_impact'])
        spd_diff = abs(original_fairness_metrics['statistical_parity_difference'] - synthetic_fairness_metrics['statistical_parity_difference'])
        
        logger.info("\nFairness Preservation:")
        logger.info(f"Disparate Impact Difference: {di_diff:.4f}")
        logger.info(f"Statistical Parity Difference: {spd_diff:.4f}")
        
        # Evaluate if synthetic data preserves fairness properties
        di_preserved = di_diff < 0.1
        spd_preserved = spd_diff < 0.05
        
        if di_preserved and spd_preserved:
            logger.info("✅ Synthetic data successfully preserves fairness properties of the original data")
        elif di_preserved:
            logger.info("⚠️ Synthetic data preserves disparate impact but not statistical parity")
        elif spd_preserved:
            logger.info("⚠️ Synthetic data preserves statistical parity but not disparate impact")
        else:
            logger.info("❌ Synthetic data does not preserve fairness properties of the original data")
        
    else:
        # Just analyze synthetic data
        logger.info("Original data not provided, analyzing synthetic data only")
        
        # Plot fairness metrics for synthetic data only
        plot_fairness_metrics(all_results['synthetic'], os.path.join(args.output_dir, 'plots'))
        
        # Plot group statistics for synthetic data
        plot_group_statistics(synthetic_group_stats, os.path.join(args.output_dir, 'plots'))
        
        # Log results
        logger.info("Fairness Metrics:")
        logger.info(f"Disparate Impact: {synthetic_fairness_metrics['disparate_impact']:.4f}")
        logger.info(f"Statistical Parity Difference: {synthetic_fairness_metrics['statistical_parity_difference']:.4f}")
        logger.info(f"Group Fairness: {synthetic_fairness_metrics['group_fairness']:.4f}")
        
        logger.info("\nGroup Statistics:")
        for group, count in synthetic_group_stats['group_counts'].items():
            logger.info(f"{group}: {count} instances ({synthetic_group_stats['group_percentages'][group]:.2f}%)")
        
        logger.info("\nPositive Outcome Rates:")
        for group, rate in synthetic_group_stats['positive_outcome_rates'].items():
            logger.info(f"{group}: {rate:.4f}")
        
        logger.info(f"Positive Rate Difference: {synthetic_group_stats.get('positive_rate_difference', 'N/A')}")
        
        logger.info("\nModel Fairness Metrics:")
        logger.info(f"Equal Opportunity Difference: {synthetic_model_metrics['equal_opportunity_difference']:.4f}")
        logger.info(f"Average Odds Difference: {synthetic_model_metrics['average_odds_difference']:.4f}")
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 