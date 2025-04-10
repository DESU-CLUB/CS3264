#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings
from tqdm import tqdm
import math
from scipy.stats import entropy
from typing import Dict, List, Tuple, Union, Optional
import json
import logging
from pathlib import Path

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare feature distributions between original and synthetic data")
    parser.add_argument("--original", type=str, required=True, help="Path to original dataset CSV")
    parser.add_argument("--synthetic", type=str, required=True, help="Path to synthetic dataset CSV")
    parser.add_argument("--output-dir", type=str, default="./distribution_results", help="Directory to save results")
    parser.add_argument("--n-bins", type=int, default=20, help="Number of bins for continuous features")
    parser.add_argument("--separator", type=str, default=";", help="CSV separator")
    parser.add_argument("--categorical-threshold", type=int, default=10, 
                       help="Max unique values to consider a feature categorical")
    return parser.parse_args()

def identify_feature_types(df: pd.DataFrame, categorical_threshold: int = 10) -> Dict[str, List[str]]:
    """
    Identify categorical and numerical features in the dataset.
    
    Args:
        df: Input DataFrame
        categorical_threshold: Maximum number of unique values to consider a feature categorical
        
    Returns:
        Dictionary with 'categorical' and 'numerical' lists of feature names
    """
    feature_types = {'categorical': [], 'numerical': []}
    
    # Identify feature types
    for col in df.columns:
        # Check for mixed types that might cause issues
        try:
            # Skip columns with mixed types that can't be sorted
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 0:
                # Test if values can be properly sorted
                sorted(unique_values)
            
            n_unique = df[col].nunique()
            if pd.api.types.is_numeric_dtype(df[col]):
                if n_unique <= categorical_threshold:
                    feature_types['categorical'].append(col)
                else:
                    feature_types['numerical'].append(col)
            else:
                feature_types['categorical'].append(col)
        except (TypeError, ValueError):
            # If sorting fails due to mixed types, treat as numerical
            logger.warning(f"Column {col} has mixed types that can't be sorted. Treating as numerical.")
            feature_types['numerical'].append(col)
            
    logger.info(f"Found {len(feature_types['categorical'])} categorical features and "
                f"{len(feature_types['numerical'])} numerical features")
    
    return feature_types

def calculate_distribution(series: pd.Series, n_bins: int = 20, is_categorical: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate normalized distribution for a feature.
    
    Args:
        series: Feature values
        n_bins: Number of bins for numerical features
        is_categorical: Whether the feature is categorical
        
    Returns:
        Tuple of (bin_edges, normalized_counts)
    """
    if is_categorical:
        value_counts = series.value_counts(normalize=True).sort_index()
        return np.array(value_counts.index), value_counts.values
    else:
        # Handle numerical features
        hist, bin_edges = np.histogram(series.dropna(), bins=n_bins)
        # Normalize counts to get probabilities
        normalized_counts = hist / float(sum(hist))
        return bin_edges, normalized_counts

def smooth_distribution(counts: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Apply smoothing to distribution to avoid zeros which cause issues in KL divergence.
    
    Args:
        counts: Distribution counts
        epsilon: Small value to add
        
    Returns:
        Smoothed distribution
    """
    smoothed = counts + epsilon
    return smoothed / smoothed.sum()  # Renormalize

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate KL divergence between two distributions.
    
    Args:
        p: First distribution (original)
        q: Second distribution (synthetic)
        
    Returns:
        KL divergence value
    """
    # Apply smoothing to avoid division by zero
    p_smoothed = smooth_distribution(p)
    q_smoothed = smooth_distribution(q)
    
    # Calculate KL divergence
    return entropy(p_smoothed, q_smoothed)

def calculate_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.
    This is a symmetric measure based on KL divergence.
    
    Args:
        p: First distribution (original)
        q: Second distribution (synthetic)
        
    Returns:
        JS divergence value
    """
    # Apply smoothing to avoid division by zero
    p_smoothed = smooth_distribution(p)
    q_smoothed = smooth_distribution(q)
    
    # Calculate average distribution
    m = 0.5 * (p_smoothed + q_smoothed)
    
    # Calculate JS divergence
    js_div = 0.5 * (entropy(p_smoothed, m) + entropy(q_smoothed, m))
    return js_div

def calculate_wasserstein_distance(original_vals: np.ndarray, synthetic_vals: np.ndarray, 
                                   original_bins: np.ndarray, synthetic_bins: np.ndarray,
                                   is_categorical: bool = False) -> float:
    """
    Calculate Wasserstein distance (Earth Mover's Distance) between distributions.
    
    Args:
        original_vals: Original distribution values
        synthetic_vals: Synthetic distribution values
        original_bins: Original distribution bin edges
        synthetic_bins: Synthetic distribution bin edges
        is_categorical: Whether feature is categorical
        
    Returns:
        Wasserstein distance
    """
    try:
        if is_categorical:
            # For categorical values, we need point masses
            # Convert to CDFs for Wasserstein distance
            original_cdf = np.cumsum(original_vals)
            synthetic_cdf = np.cumsum(synthetic_vals)
            
            # Calculate Wasserstein as the area between CDFs
            return np.sum(np.abs(original_cdf - synthetic_cdf)) / len(original_cdf)
        else:
            # For numerical features with histograms, calculate bin centers
            original_centers = (original_bins[:-1] + original_bins[1:]) / 2
            
            # Use scipy's implementation for continuous distributions
            return stats.wasserstein_distance(
                original_centers, original_centers,
                original_vals, synthetic_vals
            )
    except Exception as e:
        # If calculation fails, return a default high value
        logger.warning(f"Error calculating Wasserstein distance: {e}")
        return 1.0  # Return a default high value indicating poor match

def plot_feature_distribution(feature: str, original_series: pd.Series, synthetic_series: pd.Series,
                             n_bins: int, is_categorical: bool, output_dir: str,
                             metrics: Dict[str, float]) -> None:
    """
    Plot distribution comparison for a single feature.
    
    Args:
        feature: Feature name
        original_series: Original data values
        synthetic_series: Synthetic data values
        n_bins: Number of bins for histograms
        is_categorical: Whether feature is categorical
        output_dir: Directory to save plots
        metrics: Dictionary of calculated metrics
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if is_categorical:
        try:
            # For categorical features, use bar plots
            original_counts = original_series.value_counts(normalize=True).sort_index()
            synthetic_counts = synthetic_series.value_counts(normalize=True).sort_index()
            
            # Create a mapping between string representations and original values
            orig_map = {str(idx): idx for idx in original_counts.index}
            synth_map = {str(idx): idx for idx in synthetic_counts.index}
            
            # Get union of all categories as strings for safe sorting
            all_categories_str = sorted(set(orig_map.keys()) | set(synth_map.keys()))
            
            # Create aligned distributions
            indices = np.arange(len(all_categories_str))
            width = 0.35  # width of bars
            
            # Fill in missing values with zeros, using original values for lookup
            orig_values = []
            synth_values = []
            display_categories = []  # For axis labels
            
            for cat_str in all_categories_str:
                # Get original category value if it exists
                orig_cat = orig_map.get(cat_str)
                synth_cat = synth_map.get(cat_str)
                
                # Use the original category for lookup if available
                orig_values.append(original_counts.get(orig_cat, 0))
                synth_values.append(synthetic_counts.get(synth_cat, 0))
                
                # For display, use a value that exists or the string representation
                display_categories.append(str(orig_cat if orig_cat is not None else synth_cat))
            
            # Plot bars
            ax.bar(indices - width/2, orig_values, width, label='Original', alpha=0.7, color='#3498db')
            ax.bar(indices + width/2, synth_values, width, label='Synthetic', alpha=0.7, color='#e74c3c')
            
            # Set x-axis labels
            ax.set_xticks(indices)
            if len(display_categories) > 10:
                # If too many categories, rotate labels
                ax.set_xticklabels(display_categories, rotation=45, ha='right')
            else:
                ax.set_xticklabels(display_categories)
        except Exception as e:
            logger.warning(f"Error plotting categorical distribution for {feature}: {e}")
            logger.warning("Falling back to histogram representation")
            is_categorical = False  # Fall back to numerical representation
    else:
        # For numerical features, use histograms
        # Calculate common range to use same bins
        min_val = min(original_series.min(), synthetic_series.min())
        max_val = max(original_series.max(), synthetic_series.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Plot histograms
        ax.hist(original_series, bins=bins, alpha=0.7, density=True, 
                label='Original', color='#3498db')
        ax.hist(synthetic_series, bins=bins, alpha=0.7, density=True, 
                label='Synthetic', color='#e74c3c')
    
    # Add metrics to the plot
    textstr = '\n'.join([
        f'KL Divergence: {metrics["kl_divergence"]:.4f}',
        f'JS Divergence: {metrics["js_divergence"]:.4f}',
        f'Wasserstein: {metrics["wasserstein"]:.4f}'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_title(f'Distribution Comparison: {feature}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Normalized Frequency')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_{feature.replace(" ", "_")}.png'))
    plt.close()

def compare_distributions(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                         output_dir: str, n_bins: int = 20, 
                         categorical_threshold: int = 10) -> Dict:
    """
    Compare feature distributions between original and synthetic data.
    
    Args:
        original_df: Original dataset
        synthetic_df: Synthetic dataset
        output_dir: Directory to save results
        n_bins: Number of bins for histograms
        categorical_threshold: Max unique values to consider categorical
        
    Returns:
        Dictionary with distribution comparison metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Get common columns
    common_columns = list(set(original_df.columns) & set(synthetic_df.columns))
    logger.info(f"Comparing {len(common_columns)} common features")
    
    # Identify feature types
    feature_types = identify_feature_types(original_df[common_columns], categorical_threshold)
    
    # Store results
    results = {
        'overall': {
            'avg_kl_divergence': 0,
            'avg_js_divergence': 0,
            'avg_wasserstein': 0,
        },
        'features': {}
    }
    
    # Analyze each feature
    for feature in tqdm(common_columns, desc="Analyzing features"):
        # Skip features with all missing values
        if original_df[feature].isna().all() or synthetic_df[feature].isna().all():
            logger.warning(f"Skipping {feature} due to missing values")
            continue
            
        is_categorical = feature in feature_types['categorical']
        
        # Calculate distributions
        if is_categorical:
            try:
                # For categorical, get all possible values
                all_values = set(original_df[feature].dropna().unique()) | set(synthetic_df[feature].dropna().unique())
                
                # Create string representation mapping to handle mixed types
                str_to_val = {str(val): val for val in all_values}
                sorted_str_keys = sorted(str_to_val.keys())
                
                # Create distributions with zeros for missing categories
                orig_counts = original_df[feature].value_counts(normalize=True)
                synth_counts = synthetic_df[feature].value_counts(normalize=True)
                
                # Ensure both have same categories using original values to look up
                orig_dist = np.array([orig_counts.get(str_to_val[str_key], 0) for str_key in sorted_str_keys])
                synth_dist = np.array([synth_counts.get(str_to_val[str_key], 0) for str_key in sorted_str_keys])
                
                # Ensure distributions sum to 1
                if np.sum(orig_dist) > 0:
                    orig_dist = orig_dist / np.sum(orig_dist)
                if np.sum(synth_dist) > 0:
                    synth_dist = synth_dist / np.sum(synth_dist)
                
                # Bin edges are the category values in sorted string representation order
                orig_bins = np.array([str_to_val[str_key] for str_key in sorted_str_keys])
                synth_bins = orig_bins.copy()
            except Exception as e:
                # If there's an error with categorical handling, fall back to numerical approach
                logger.warning(f"Error handling categorical feature {feature}: {e}")
                logger.warning(f"Falling back to numerical distribution for {feature}")
                is_categorical = False
                # Process as numerical feature
                min_val = min(original_df[feature].min(), synthetic_df[feature].min())
                max_val = max(original_df[feature].max(), synthetic_df[feature].max())
                
                # Use common bins for both distributions
                bins = np.linspace(min_val, max_val, n_bins + 1)
                
                # Calculate histograms
                orig_hist, orig_bins = np.histogram(original_df[feature].dropna(), bins=bins, density=True)
                synth_hist, synth_bins = np.histogram(synthetic_df[feature].dropna(), bins=bins, density=True)
                
                # Normalize to get probability distributions
                orig_dist = orig_hist / np.sum(orig_hist) if np.sum(orig_hist) > 0 else orig_hist
                synth_dist = synth_hist / np.sum(synth_hist) if np.sum(synth_hist) > 0 else synth_hist
        else:
            # For numerical features, use histograms with same bins
            min_val = min(original_df[feature].min(), synthetic_df[feature].min())
            max_val = max(original_df[feature].max(), synthetic_df[feature].max())
            
            # Use common bins for both distributions
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            # Calculate histograms
            orig_hist, orig_bins = np.histogram(original_df[feature].dropna(), bins=bins, density=True)
            synth_hist, synth_bins = np.histogram(synthetic_df[feature].dropna(), bins=bins, density=True)
            
            # Normalize to get probability distributions
            orig_dist = orig_hist / np.sum(orig_hist) if np.sum(orig_hist) > 0 else orig_hist
            synth_dist = synth_hist / np.sum(synth_hist) if np.sum(synth_hist) > 0 else synth_hist
        
        # Calculate metrics
        kl_div = calculate_kl_divergence(orig_dist, synth_dist)
        js_div = calculate_js_divergence(orig_dist, synth_dist)
        wasserstein = calculate_wasserstein_distance(
            orig_dist, synth_dist, orig_bins, synth_bins, is_categorical
        )
        
        # Store metrics
        results['features'][feature] = {
            'type': 'categorical' if is_categorical else 'numerical',
            'kl_divergence': float(kl_div),
            'js_divergence': float(js_div),
            'wasserstein': float(wasserstein)
        }
        
        # Update overall metrics
        results['overall']['avg_kl_divergence'] += kl_div
        results['overall']['avg_js_divergence'] += js_div
        results['overall']['avg_wasserstein'] += wasserstein
        
        # Plot distribution
        plot_feature_distribution(
            feature, 
            original_df[feature], 
            synthetic_df[feature],
            n_bins, 
            is_categorical,
            os.path.join(output_dir, 'plots'),
            results['features'][feature]
        )
    
    # Calculate averages
    num_features = len(results['features'])
    if num_features > 0:
        results['overall']['avg_kl_divergence'] /= num_features
        results['overall']['avg_js_divergence'] /= num_features
        results['overall']['avg_wasserstein'] /= num_features
    
    # Generate summary visualizations
    create_summary_visualizations(results, os.path.join(output_dir, 'plots'))
    
    # Save results to JSON
    with open(os.path.join(output_dir, 'distribution_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_summary_visualizations(results: Dict, output_dir: str) -> None:
    """
    Create summary visualizations of distribution metrics.
    
    Args:
        results: Results dictionary with metrics
        output_dir: Directory to save visualizations
    """
    # Extract metrics for visualization
    features = []
    kl_values = []
    js_values = []
    wasserstein_values = []
    types = []
    
    for feature, metrics in results['features'].items():
        features.append(feature)
        kl_values.append(metrics['kl_divergence'])
        js_values.append(metrics['js_divergence'])
        wasserstein_values.append(metrics['wasserstein'])
        types.append(metrics['type'])
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Feature': features,
        'KL Divergence': kl_values,
        'JS Divergence': js_values,
        'Wasserstein': wasserstein_values,
        'Type': types
    })
    
    # 1. Bar chart of KL divergence by feature
    plt.figure(figsize=(14, 10))
    # Sort by KL divergence
    df_sorted = df.sort_values('KL Divergence', ascending=False)
    
    # Plot top 20 features with highest divergence
    top_n = min(20, len(df_sorted))
    ax = sns.barplot(x='KL Divergence', y='Feature', data=df_sorted.head(top_n),
                     hue='Type', palette={'categorical': '#3498db', 'numerical': '#e74c3c'})
    
    plt.title('Top Features by KL Divergence', fontsize=16)
    plt.xlabel('KL Divergence (lower is better)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_kl_divergence.png'))
    plt.close()
    
    # 2. Scatterplot comparing JS vs KL divergence
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='KL Divergence', y='JS Divergence', hue='Type', 
                   data=df, s=100, alpha=0.7,
                   palette={'categorical': '#3498db', 'numerical': '#e74c3c'})
    
    # Add feature labels to points
    for i, row in df.iterrows():
        plt.annotate(row['Feature'], 
                    (row['KL Divergence'], row['JS Divergence']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.title('KL Divergence vs JS Divergence by Feature', fontsize=16)
    plt.xlabel('KL Divergence', fontsize=14)
    plt.ylabel('JS Divergence', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kl_vs_js_divergence.png'))
    plt.close()
    
    # 3. Histogram of divergence distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(df['KL Divergence'], kde=True)
    plt.title('Distribution of KL Divergence')
    plt.xlabel('KL Divergence')
    
    plt.subplot(1, 3, 2)
    sns.histplot(df['JS Divergence'], kde=True)
    plt.title('Distribution of JS Divergence')
    plt.xlabel('JS Divergence')
    
    plt.subplot(1, 3, 3)
    sns.histplot(df['Wasserstein'], kde=True)
    plt.title('Distribution of Wasserstein Distance')
    plt.xlabel('Wasserstein Distance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'divergence_distributions.png'))
    plt.close()
    
    # 4. Compare categorical vs numerical features
    plt.figure(figsize=(12, 6))
    
    # Melt the DataFrame for easier grouped boxplots
    df_melted = pd.melt(df, id_vars=['Feature', 'Type'], 
                        value_vars=['KL Divergence', 'JS Divergence', 'Wasserstein'],
                        var_name='Metric', value_name='Value')
    
    # Create boxplots
    sns.boxplot(x='Metric', y='Value', hue='Type', data=df_melted,
                palette={'categorical': '#3498db', 'numerical': '#e74c3c'})
    
    plt.title('Distribution Metrics by Feature Type', fontsize=16)
    plt.ylabel('Value', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.yscale('log')  # Log scale often helps with divergence metrics
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_feature_type.png'))
    plt.close()

def main():
    """Main function to run distribution comparison."""
    args = parse_args()
    
    # Load datasets
    logger.info(f"Loading original dataset from {args.original}")
    original_df = pd.read_csv(args.original, sep=args.separator)
    
    logger.info(f"Loading synthetic dataset from {args.synthetic}")
    synthetic_df = pd.read_csv(args.synthetic, sep=args.separator)
    
    logger.info(f"Original data shape: {original_df.shape}")
    logger.info(f"Synthetic data shape: {synthetic_df.shape}")
    
    # Preprocess: Convert gender from 0/1 back to female/male in synthetic data
    if 'gender' in synthetic_df.columns:
        logger.info("Converting gender values in synthetic data from 0/1 to Female/Male")
        gender_mapping = {0: "Female", 1: "Male"}
        synthetic_df['gender'] = synthetic_df['gender'].map(lambda x: gender_mapping.get(x, x))
        
        # Check original data format to ensure consistency
        if 'gender' in original_df.columns:
            unique_gender_values = original_df['gender'].unique()
            logger.info(f"Original data gender values: {unique_gender_values}")
            logger.info(f"Synthetic data gender values: {synthetic_df['gender'].unique()}")
    
    # Compare distributions
    results = compare_distributions(
        original_df, 
        synthetic_df, 
        args.output_dir,
        n_bins=args.n_bins,
        categorical_threshold=args.categorical_threshold
    )
    
    # Log summary
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")
    logger.info(f"Average KL Divergence: {results['overall']['avg_kl_divergence']:.4f}")
    logger.info(f"Average JS Divergence: {results['overall']['avg_js_divergence']:.4f}")
    logger.info(f"Average Wasserstein Distance: {results['overall']['avg_wasserstein']:.4f}")

if __name__ == "__main__":
    main()
