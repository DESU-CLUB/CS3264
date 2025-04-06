import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import logging
import asyncio
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def analyze_feature(client, feature, df, dataset_name, headers, output_dir):
    """
    Analyze a single feature and save results to a text file.
    
    Args:
        client: OpenAI client
        feature: Feature name
        df: Pandas DataFrame
        dataset_name: Name of the dataset
        headers: List of all column headers
        output_dir: Directory to save output
        
    Returns:
        str: Path to created file
    """
    rows = len(df)
    
    # Create a safe filename
    safe_feature_name = feature.replace(' ', '_').replace('/', '_')
    feature_path = os.path.join(output_dir, f"{dataset_name}_{safe_feature_name}.txt")
    
    # Collect statistics for this feature
    feature_stats = {"type": str(df[feature].dtype)}
    
    with open(feature_path, 'w') as f:
        f.write(f"# Feature: {feature}\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Data type: {df[feature].dtype}\n\n")
        
        # Calculate statistics based on data type
        if pd.api.types.is_numeric_dtype(df[feature]):
            # Five-point summary
            min_val = df[feature].min()
            q1 = df[feature].quantile(0.25)
            median = df[feature].median()
            q3 = df[feature].quantile(0.75)
            max_val = df[feature].max()
            mean = df[feature].mean()
            std = df[feature].std()
            
            f.write("## Five-point summary\n\n")
            f.write(f"- Minimum: {min_val}\n")
            f.write(f"- Q1 (25th percentile): {q1}\n")
            f.write(f"- Median: {median}\n")
            f.write(f"- Q3 (75th percentile): {q3}\n")
            f.write(f"- Maximum: {max_val}\n")
            f.write(f"- Mean: {mean}\n")
            f.write(f"- Standard Deviation: {std}\n\n")
            
            feature_stats.update({
                "min": float(min_val),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(max_val),
                "mean": float(mean),
                "std": float(std)
            })
        else:
            # Categorical statistics
            unique_count = df[feature].nunique()
            top_values = df[feature].value_counts().head(5).to_dict()
            
            f.write("## Categorical statistics\n\n")
            f.write(f"Unique values: {unique_count}\n")
            f.write("Top 5 values:\n")
            for val, count in top_values.items():
                f.write(f"- {val}: {count} ({count/rows*100:.2f}%)\n")
            f.write("\n")
            
            feature_stats.update({
                "unique_count": unique_count,
                "top_values": {str(k): int(v) for k, v in top_values.items()}
            })
        
        # Missing values
        missing = df[feature].isna().sum()
        missing_pct = missing/rows*100
        
        f.write("## Missing values\n\n")
        f.write(f"Missing count: {missing}\n")
        f.write(f"Missing percentage: {missing_pct:.2f}%\n\n")
        
        feature_stats["missing"] = int(missing)
        feature_stats["missing_pct"] = float(missing_pct)
    
    # Get other features for relationship analysis
    other_features = [h for h in headers if h != feature]
    
    # Use OpenAI to analyze this specific feature
    logger.info(f"Generating insights for feature: {feature}")
    
    # Create prompt for OpenAI specific to this feature
    prompt = f"""
    Analyze the feature '{feature}' from dataset '{dataset_name}' based on its statistics:
    
    {feature_stats}
    
    Other features in the dataset: {', '.join(other_features)}
    
    Provide:
    1. A brief interpretation of this feature's statistics
    2. Possible relationships with other features mentioned above
    3. How this feature might influence or correlate with the other features
    
    Be specific, detailed and concise. Base your analysis on the provided statistics.
    """
    
    # Call OpenAI API for this feature
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis expert specializing in feature analysis and relationships."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        
        # Write the analysis to the file
        analysis = response.choices[0].message.content
        
        with open(feature_path, 'a') as f:
            f.write("## Feature Analysis and Relationships\n\n")
            f.write(analysis)
            
        return feature_path
    except Exception as e:
        logger.error(f"Error generating insights for feature {feature}: {str(e)}")
        with open(feature_path, 'a') as f:
            f.write("## Feature Analysis and Relationships\n\n")
            f.write(f"Error generating insights: {str(e)}")
        return feature_path

async def analyze_csv_features_async(csv_path, output_dir="./data/features/"):
    """
    Asynchronously analyzes a CSV file, computes statistics for each feature,
    uses OpenAI to generate insights about feature relationships,
    and saves results to individual text files for each feature.
    
    Args:
        csv_path (str): Path to the CSV file
        output_dir (str): Directory to save the output text files
        
    Returns:
        str: "Success" if operation completed successfully
    """
    try:
        logger.info(f"Analyzing CSV file: {csv_path}")
        start_time = time.time()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get basic dataset information
        rows, cols = df.shape
        headers = df.columns.tolist()
        dataset_name = os.path.basename(csv_path).split('.')[0]
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create an overview file
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
        
        # Process all features asynchronously
        tasks = []
        for feature in headers:
            task = analyze_feature(client, feature, df, dataset_name, headers, output_dir)
            tasks.append(task)
        
        # Wait for all tasks to complete
        feature_files = await asyncio.gather(*tasks)
        
        end_time = time.time()
        logger.info(f"Analysis complete. Results saved to {output_dir}")
        logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
        
        return "Success"
            
    except Exception as e:
        logger.error(f"Error analyzing CSV: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def analyze_csv_features(csv_path, output_dir="./data/features/"):
    """
    Wrapper function to run the async analysis function
    """
    return asyncio.run(analyze_csv_features_async(csv_path, output_dir))

if __name__ == "__main__":
    # Example usage
    result = analyze_csv_features("./datasets/diabetes.csv")
    print(result)
