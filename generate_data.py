import pandas as pd
import numpy as np
import os
import logging
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Optional, Any, Literal
import random
from datetime import datetime
import asyncio
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# 1. Function to load query engine or regenerate embeddings
def load_query_engine(persist_dir="./data/chroma_db", features_dir="./data/features/", collection_name="dquery"):
    """
    Load an existing query engine or create a new one with embeddings
    
    Args:
        persist_dir: Directory where ChromaDB index is persisted
        features_dir: Directory containing feature documents
        collection_name: Name of the ChromaDB collection
        
    Returns:
        query_engine: Query engine for the index
    """
    logger.info(f"Loading or creating query engine from {persist_dir}")
    
    # Check if persist directory exists
    if os.path.exists(persist_dir):
        try:
            # Try to load from persisted client
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            
            # Check if collection exists
            try:
                chroma_collection = chroma_client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
                
                # Set up vector store and index
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Use OpenAI embedding
                embed_model = OpenAIEmbedding()
                
                # Create index from existing vector store
                index = VectorStoreIndex.from_vector_store(
                    vector_store, embed_model=embed_model
                )
                
                return index.as_query_engine()
            except Exception as e:
                logger.warning(f"Collection not found: {str(e)}")
        except Exception as e:
            logger.warning(f"Error loading persisted client: {str(e)}")
    
    # If loading failed or directory doesn't exist, create new index
    logger.info(f"Creating new index from {features_dir}")
    
    # Create persisted client
    os.makedirs(persist_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    # Create new collection
    try:
        # Try to get existing collection first (if it exists)
        chroma_collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except:
        # Create new collection if it doesn't exist
        chroma_collection = chroma_client.create_collection(name=collection_name)
        logger.info(f"Created new collection: {collection_name}")
    
    # Create index with documents
    from llama_index.core import SimpleDirectoryReader
    
    # Load documents
    documents = SimpleDirectoryReader(features_dir).load_data()
    logger.info(f"Loaded {len(documents)} documents from {features_dir}")
    
    # Set up vector store and create index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Use OpenAI embedding
    embed_model = OpenAIEmbedding()
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    
    return index.as_query_engine()

# 2. Get feature explanation and relationships
def get_feature_context(query_engine, feature, label_column, label_value, existing_values=None):
    """
    Get explanation and relationships for a feature
    
    Args:
        query_engine: Query engine for ChromaDB
        feature: Feature name
        label_column: Name of the label column
        label_value: Value of the label
        existing_values: Dictionary of already generated feature values
        
    Returns:
        dict: Context information about the feature
    """
    # Build context query based on available information
    context_parts = []
    
    # Add information about the label
    context_parts.append(f"The target variable {label_column} has value {label_value}.")
    
    # Add information about already generated features
    if existing_values and len(existing_values) > 0:
        context_parts.append("The following features have already been generated:")
        for feat, val in existing_values.items():
            if feat != label_column:
                context_parts.append(f"- {feat}: {val}")
    
    context = " ".join(context_parts)
    
    # Query for feature explanation
    explanation_query = f"""
    Explain the feature '{feature}' in detail, including:
    1. What this feature represents
    2. Its typical range or possible values
    3. Its relationship with the target variable {label_column}
    4. How it might be affected by or related to other features
    
    Context: {context}
    """
    
    explanation = query_engine.query(explanation_query)
    
    # Query for feature relationships with label
    label_relationship_query = f"""
    Describe the relationship between feature '{feature}' and the target variable '{label_column}'.
    When '{label_column}' is {label_value}, what range of values would you expect for '{feature}'?
    
    Context: {context}
    """
    
    label_relationship = query_engine.query(label_relationship_query)
    
    # Query for feature relationships with other features
    if existing_values and len(existing_values) > 0:
        other_relationships_query = f"""
        Given that we know these feature values:
        {json.dumps(existing_values, indent=2)}
        
        Describe how these existing values should influence the value of '{feature}'.
        Be specific about how each of the known features affects '{feature}'.
        """
        
        other_relationships = query_engine.query(other_relationships_query)
    else:
        other_relationships = "No existing feature values to evaluate relationships."
    
    # Return structured context
    return {
        "feature": feature,
        "explanation": str(explanation),
        "label_relationship": str(label_relationship),
        "other_relationships": str(other_relationships)
    }

# 3. Get feature type from CSV
def get_feature_type(csv_path, feature):
    """
    Get the data type and statistics for a feature
    
    Args:
        csv_path: Path to the CSV file
        feature: Feature name
        
    Returns:
        dict: Type information and statistics
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Get the column data
    column_data = df[feature]
    
    # Get the pandas dtype
    dtype = column_data.dtype
    
    # Initialize type info
    type_info = {
        "name": feature,
        "pandas_dtype": str(dtype)
    }
    
    # Determine the Python type and add appropriate stats
    if pd.api.types.is_numeric_dtype(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            type_info["python_type"] = "integer"
        else:
            type_info["python_type"] = "float"
            
        # Add statistics for numeric features
        type_info.update({
            "min": float(column_data.min()),
            "max": float(column_data.max()),
            "mean": float(column_data.mean()),
            "median": float(column_data.median()),
            "std": float(column_data.std())
        })
    elif pd.api.types.is_bool_dtype(dtype):
        type_info["python_type"] = "boolean"
        
        # Add statistics for boolean features
        type_info["true_count"] = int(column_data.sum())
        type_info["false_count"] = int(len(column_data) - column_data.sum())
    elif pd.api.types.is_datetime64_dtype(dtype):
        type_info["python_type"] = "datetime"
        
        # Add statistics for datetime features
        type_info["min"] = column_data.min().isoformat()
        type_info["max"] = column_data.max().isoformat()
    else:
        # Assume string/categorical
        type_info["python_type"] = "string"
        
        # Add statistics for categorical features
        value_counts = column_data.value_counts()
        type_info["unique_count"] = int(len(value_counts))
        
        # Get top categories
        if len(value_counts) <= 20:  # If reasonable number of categories
            categories = {}
            for val, count in value_counts.items():
                categories[str(val)] = int(count)
            type_info["categories"] = categories
    
    # Add missing value info
    type_info["missing_count"] = int(column_data.isna().sum())
    type_info["missing_percentage"] = float(column_data.isna().mean() * 100)
    
    return type_info

# 4. Pydantic models for structured output
class NumericFeatureValue(BaseModel):
    """Model for numeric feature values"""
    value: float
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class CategoricalFeatureValue(BaseModel):
    """Model for categorical feature values"""
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class BooleanFeatureValue(BaseModel):
    """Model for boolean feature values"""
    value: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class DateTimeFeatureValue(BaseModel):
    """Model for datetime feature values"""
    value: str  # ISO format
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class FeatureValueResponse(BaseModel):
    """Model for feature value generation response"""
    feature_name: str
    feature_type: Literal["numeric", "categorical", "boolean", "datetime"]
    generated_value: Union[NumericFeatureValue, CategoricalFeatureValue, BooleanFeatureValue, DateTimeFeatureValue]

# 5. Generate value for a feature
def generate_feature_value(query_engine, feature, feature_type, context, existing_values, csv_path):
    """
    Generate a value for a feature based on its type and context
    
    Args:
        query_engine: Query engine for ChromaDB
        feature: Feature name
        feature_type: Type information from get_feature_type
        context: Context from get_feature_context
        existing_values: Dictionary of already generated feature values
        csv_path: Path to the CSV file
        
    Returns:
        Union[float, str, bool, datetime]: Generated value
    """
    # Use OpenAI to generate a value based on context
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Create a prompt for the appropriate type
    python_type = feature_type["python_type"]
    
    # Structure for the response
    if python_type in ["integer", "float"]:
        response_format = {
            "feature_name": feature,
            "feature_type": "numeric",
            "generated_value": {
                "value": 0.0,  # Example value
                "confidence": 0.9,  # Example confidence
                "reasoning": "Reasoning for the value"
            }
        }
        
        model_class = NumericFeatureValue
    elif python_type == "boolean":
        response_format = {
            "feature_name": feature,
            "feature_type": "boolean",
            "generated_value": {
                "value": True,  # Example value
                "confidence": 0.9,  # Example confidence
                "reasoning": "Reasoning for the value"
            }
        }
        
        model_class = BooleanFeatureValue
    elif python_type == "datetime":
        response_format = {
            "feature_name": feature,
            "feature_type": "datetime",
            "generated_value": {
                "value": "2023-01-01T00:00:00",  # Example value
                "confidence": 0.9,  # Example confidence
                "reasoning": "Reasoning for the value"
            }
        }
        
        model_class = DateTimeFeatureValue
    else:  # string/categorical
        response_format = {
            "feature_name": feature,
            "feature_type": "categorical",
            "generated_value": {
                "value": "example_value",  # Example value
                "confidence": 0.9,  # Example confidence
                "reasoning": "Reasoning for the value"
            }
        }
        
        model_class = CategoricalFeatureValue
    
    # Create the prompt
    prompt = f"""
    Generate a realistic value for the feature '{feature}' based on the following context:
    
    Feature Explanation: {context['explanation']}
    
    Relationship with target variable: {context['label_relationship']}
    
    Relationships with other features: {context['other_relationships']}
    
    Technical specifications:
    {json.dumps(feature_type, indent=2)}
    
    Existing values:
    {json.dumps(existing_values, indent=2)}
    
    Return your response in the following JSON format:
    {json.dumps(response_format, indent=2)}
    
    Ensure that:
    1. The value is appropriate for the feature type ({python_type})
    2. The value takes into account the relationships with existing features
    3. The value is consistent with the target variable
    4. The value falls within any known constraints (min/max, categories, etc.)
    5. The confidence reflects how certain you are about this value (0-1)
    6. The reasoning explains why this value makes sense
    """
    
    # Call the API
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data generation expert. Generate realistic feature values based on context and relationships."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        response_text = completion.choices[0].message.content
        response_data = json.loads(response_text)
        
        # Validate with Pydantic
        feature_value_response = FeatureValueResponse(**response_data)
        
        # Extract the actual value
        generated_value = feature_value_response.generated_value.value
        
        # Convert to the appropriate Python type
        if python_type == "integer":
            return int(float(generated_value))
        elif python_type == "float":
            return float(generated_value)
        elif python_type == "boolean":
            return bool(generated_value)
        else:
            return str(generated_value)
        
    except Exception as e:
        logger.error(f"Error generating value for feature {feature}: {str(e)}")
        
        # Fallback to a random value within constraints
        if python_type == "integer":
            min_val = feature_type.get("min", 0)
            max_val = feature_type.get("max", 100)
            return random.randint(int(min_val), int(max_val))
        elif python_type == "float":
            min_val = feature_type.get("min", 0.0)
            max_val = feature_type.get("max", 1.0)
            return random.uniform(float(min_val), float(max_val))
        elif python_type == "boolean":
            return random.choice([True, False])
        elif python_type == "datetime":
            # Default to current date
            return datetime.now().isoformat()
        else:
            # If categorical with known categories
            if "categories" in feature_type and feature_type["categories"]:
                return random.choice(list(feature_type["categories"].keys()))
            return f"fallback_value_for_{feature}"

# 6 & 7. Generate data iteratively
def generate_data(csv_path, n_samples=10, persist_dir="./data/chroma_db", features_dir="./data/features/", 
                  collection_name="dquery", output_path=None):
    """
    Generate synthetic data based on feature relationships and constraints
    
    Args:
        csv_path: Path to the CSV file
        n_samples: Number of samples to generate
        persist_dir: Directory where ChromaDB index is persisted
        features_dir: Directory containing feature documents
        collection_name: Name of the ChromaDB collection
        output_path: Path to save the generated data (optional)
        
    Returns:
        pd.DataFrame: Generated data
    """
    logger.info(f"Generating {n_samples} rows of synthetic data")
    
    # 1. Load query engine
    query_engine = load_query_engine(persist_dir, features_dir, collection_name)
    if query_engine is None:
        logger.error("Failed to load or create query engine")
        return None
    
    # Read the CSV to get column information
    df = pd.read_csv(csv_path)
    all_features = df.columns.tolist()
    
    # Guess the label column (target variable)
    # For simplicity, assuming the last column is the target
    label_column = all_features[-1]
    
    # Get possible values for the label column
    if pd.api.types.is_numeric_dtype(df[label_column]):
        unique_labels = df[label_column].unique()
        if len(unique_labels) > 10:
            # Use the 5 most common values for numeric targets with many values
            label_values = df[label_column].value_counts().head(5).index.tolist()
        else:
            label_values = unique_labels.tolist()
    else:
        # Use all unique values for categorical targets
        label_values = df[label_column].unique().tolist()
    
    # Generate rows
    rows = []
    for i in range(n_samples):
        logger.info(f"Generating row {i+1}/{n_samples}")
        
        # Randomly select a label value
        label_value = random.choice(label_values)
        logger.info(f"Selected {label_column}={label_value}")
        
        # Start with the label value
        row = {label_column: label_value}
        
        # Generate values for each feature one by one
        for feature in all_features:
            # Skip the label column as it's already set
            if feature == label_column:
                continue
            
            logger.info(f"Generating value for feature: {feature}")
            
            # Get feature type
            feature_type = get_feature_type(csv_path, feature)
            
            # Get feature context
            feature_context = get_feature_context(query_engine, feature, label_column, label_value, row)
            
            # Generate feature value
            value = generate_feature_value(query_engine, feature, feature_type, feature_context, row, csv_path)
            
            # Add to row
            row[feature] = value
            
            logger.info(f"Generated {feature}={value}")
        
        # Add completed row
        rows.append(row)
    
    # Create DataFrame
    generated_df = pd.DataFrame(rows)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generated_df.to_csv(output_path, index=False)
        logger.info(f"Saved {n_samples} generated rows to {output_path}")
    
    return generated_df

# Make feature context retrieval asynchronous
async def get_feature_context_async(query_engine, feature, label_column, label_value, existing_values=None):
    """
    Asynchronously get explanation and relationships for a feature
    
    Args:
        query_engine: Query engine for ChromaDB
        feature: Feature name
        label_column: Name of the label column
        label_value: Value of the label
        existing_values: Dictionary of already generated feature values
        
    Returns:
        dict: Context information about the feature
    """
    return await asyncio.to_thread(
        get_feature_context,
        query_engine,
        feature,
        label_column,
        label_value,
        existing_values
    )

# Make feature type retrieval asynchronous
async def get_feature_type_async(csv_path, feature):
    """
    Asynchronously get the data type and statistics for a feature
    
    Args:
        csv_path: Path to the CSV file
        feature: Feature name
        
    Returns:
        dict: Type information and statistics
    """
    return await asyncio.to_thread(get_feature_type, csv_path, feature)

# Make feature value generation asynchronous
async def generate_feature_value_async(query_engine, feature, feature_type, context, existing_values, csv_path):
    """
    Asynchronously generate a value for a feature based on its type and context
    
    Args:
        query_engine: Query engine for ChromaDB
        feature: Feature name
        feature_type: Type information from get_feature_type
        context: Context from get_feature_context
        existing_values: Dictionary of already generated feature values
        csv_path: Path to the CSV file
        
    Returns:
        Union[float, str, bool, datetime]: Generated value
    """
    return await asyncio.to_thread(
        generate_feature_value,
        query_engine,
        feature,
        feature_type,
        context,
        existing_values,
        csv_path
    )

# Build dependency graph for features
def build_feature_dependency_graph(csv_path, query_engine):
    """
    Build a dependency graph for features
    
    Args:
        csv_path: Path to the CSV file
        query_engine: Query engine for ChromaDB
        
    Returns:
        nx.DiGraph: Directed graph of feature dependencies
    """
    # Read the CSV to get column information
    df = pd.read_csv(csv_path)
    all_features = df.columns.tolist()
    
    # Guess the label column
    label_column = all_features[-1]
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add all features as nodes
    for feature in all_features:
        G.add_node(feature)
    
    # Add dependencies based on correlations
    for i, feature1 in enumerate(all_features):
        if feature1 == label_column:
            continue
            
        # Make label column depend on all features
        G.add_edge(feature1, label_column)
        
        # Query for dependencies
        query = f"What other features does '{feature1}' depend on or correlate strongly with?"
        response = query_engine.query(query)
        
        # Use a simple keyword-based approach to detect mentioned features
        response_text = str(response).lower()
        
        for j, feature2 in enumerate(all_features):
            if feature1 != feature2 and feature2.lower() in response_text:
                # Add an edge if feature2 is mentioned in the response
                G.add_edge(feature2, feature1)
    
    return G

# Generate a feature value with caching
async def generate_feature_with_cache(cache, query_engine, feature, label_column, label_value, current_values, csv_path):
    """
    Generate a value for a feature with caching of context and type info
    
    Args:
        cache: Dictionary cache for feature info
        query_engine: Query engine for ChromaDB
        feature: Feature name
        label_column: Name of the label column
        label_value: Value of the label
        current_values: Dictionary of already generated feature values
        csv_path: Path to the CSV file
        
    Returns:
        Any: Generated value for the feature
    """
    # Get feature type (with caching)
    if feature not in cache.get("types", {}):
        if "types" not in cache:
            cache["types"] = {}
        cache["types"][feature] = await get_feature_type_async(csv_path, feature)
    
    feature_type = cache["types"][feature]
    
    # Get feature context
    context = await get_feature_context_async(query_engine, feature, label_column, label_value, current_values)
    
    # Generate value
    value = await generate_feature_value_async(query_engine, feature, feature_type, context, current_values, csv_path)
    
    return value

# Process features in dependency order
async def process_features_in_dependency_order(G, features, label_column, label_value, query_engine, csv_path):
    """
    Process features in order of their dependencies
    
    Args:
        G: Dependency graph
        features: List of features
        label_column: Name of the label column
        label_value: Value of the label
        query_engine: Query engine for ChromaDB
        csv_path: Path to the CSV file
        
    Returns:
        dict: Generated values for all features
    """
    # Start with the label
    values = {label_column: label_value}
    
    # Create a cache for feature info
    cache = {}
    
    # Get topological sort of the graph (respecting dependencies)
    try:
        # Remove label column from sort order (we already have it)
        G_without_label = G.copy()
        if label_column in G_without_label:
            G_without_label.remove_node(label_column)
            
        # Get sort order
        sorted_features = list(nx.topological_sort(G_without_label))
        
        # Some features might not be in the graph due to no dependencies
        missing_features = [f for f in features if f != label_column and f not in sorted_features]
        
        # Put independent features first
        feature_order = missing_features + sorted_features
    except nx.NetworkXUnfeasible:
        # If graph has cycles, fall back to original order
        feature_order = [f for f in features if f != label_column]
    
    # Group features into levels for parallel processing
    feature_levels = []
    current_level = []
    processed_features = set([label_column])
    
    for feature in feature_order:
        # Check if all dependencies are processed
        dependencies = set(G.predecessors(feature)) if feature in G else set()
        if dependencies.issubset(processed_features):
            current_level.append(feature)
        else:
            if current_level:
                feature_levels.append(current_level)
                current_level = [feature]
            else:
                current_level.append(feature)
        
        processed_features.add(feature)
    
    if current_level:
        feature_levels.append(current_level)
    
    # Process features level by level
    for level in feature_levels:
        # Process features in this level concurrently
        tasks = []
        for feature in level:
            if feature != label_column:
                task = generate_feature_with_cache(
                    cache, query_engine, feature, label_column, label_value, values, csv_path
                )
                tasks.append((feature, task))
        
        # Wait for all tasks in this level to complete
        for feature, task in tasks:
            values[feature] = await task
    
    return values

# Main asynchronous data generation function
async def generate_data_async(csv_path, n_samples=10, persist_dir="./data/chroma_db", features_dir="./data/features/", 
                  collection_name="dquery", output_path=None, max_workers=None, batch_size=None):
    """
    Asynchronously generate synthetic data based on feature relationships and constraints
    
    Args:
        csv_path: Path to the CSV file
        n_samples: Number of samples to generate
        persist_dir: Directory where ChromaDB index is persisted
        features_dir: Directory containing feature documents
        collection_name: Name of the ChromaDB collection
        output_path: Path to save the generated data (optional)
        max_workers: Maximum number of workers for parallel processing
        batch_size: Number of rows to process in a single batch (None for all at once)
        
    Returns:
        pd.DataFrame: Generated data
    """
    logger.info(f"Generating {n_samples} rows of synthetic data asynchronously")
    
    # Load query engine
    query_engine = load_query_engine(persist_dir, features_dir, collection_name)
    if query_engine is None:
        logger.error("Failed to load or create query engine")
        return None
    
    # Read the CSV to get column information
    df = pd.read_csv(csv_path)
    all_features = df.columns.tolist()
    
    # Guess the label column
    label_column = all_features[-1]
    
    # Get possible values for the label column
    if pd.api.types.is_numeric_dtype(df[label_column]):
        unique_labels = df[label_column].unique()
        if len(unique_labels) > 10:
            # Use the 5 most common values for numeric targets with many values
            label_values = df[label_column].value_counts().head(5).index.tolist()
        else:
            label_values = unique_labels.tolist()
    else:
        # Use all unique values for categorical targets
        label_values = df[label_column].unique().tolist()
    
    # Build dependency graph for features
    G = build_feature_dependency_graph(csv_path, query_engine)
    
    # Generate rows (in parallel with a limit on concurrency)
    semaphore = asyncio.Semaphore(max_workers or 5)  # Default to 5 concurrent row generations
    
    async def generate_row(i):
        async with semaphore:
            logger.info(f"Generating row {i+1}/{n_samples}")
            
            # Randomly select a label value
            label_value = random.choice(label_values)
            logger.info(f"Selected {label_column}={label_value}")
            
            # Generate all feature values according to dependency order
            row = await process_features_in_dependency_order(
                G, all_features, label_column, label_value, query_engine, csv_path
            )
            
            return row
    
    # Process rows in batches
    batch_size = batch_size or n_samples  # If no batch size provided, process all at once
    all_rows = []
    
    async def process_batch(batch_indices):
        logger.info(f"Processing batch of {len(batch_indices)} rows")
        batch_tasks = [generate_row(i) for i in batch_indices]
        batch_results = await asyncio.gather(*batch_tasks)
        return batch_results
    
    # Create batches
    batch_indices = [list(range(i, min(i + batch_size, n_samples))) for i in range(0, n_samples, batch_size)]
    
    # Process batches asynchronously
    for i, indices in enumerate(batch_indices):
        logger.info(f"Starting batch {i+1}/{len(batch_indices)}")
        batch_rows = await process_batch(indices)
        all_rows.extend(batch_rows)
        logger.info(f"Completed batch {i+1}/{len(batch_indices)}")
    
    # Create DataFrame
    generated_df = pd.DataFrame(all_rows)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generated_df.to_csv(output_path, index=False)
        logger.info(f"Saved {n_samples} generated rows to {output_path}")
    
    return generated_df

# Synchronous wrapper for the asynchronous function
def generate_data(csv_path, n_samples=10, persist_dir="./data/chroma_db", features_dir="./data/features/", 
                  collection_name="dquery", output_path=None, max_workers=None, batch_size=None):
    """
    Generate synthetic data based on feature relationships and constraints
    
    Args:
        csv_path: Path to the CSV file
        n_samples: Number of samples to generate
        persist_dir: Directory where ChromaDB index is persisted
        features_dir: Directory containing feature documents
        collection_name: Name of the ChromaDB collection
        output_path: Path to save the generated data (optional)
        max_workers: Maximum number of workers for parallel processing
        batch_size: Number of rows to process in a single batch (None for all at once)
        
    Returns:
        pd.DataFrame: Generated data
    """
    return asyncio.run(generate_data_async(
        csv_path, n_samples, persist_dir, features_dir, collection_name, output_path, max_workers, batch_size
    ))

if __name__ == "__main__":
    # Example usage
    csv_path = "./datasets/diabetes.csv"
    output_path = "./data/generated/generated_diabetes.csv"
    
    # Generate synthetic data
    generated_data = generate_data(
        csv_path=csv_path,
        n_samples=10,  # Generate 10 samples for testing
        persist_dir="./data/chroma_db",
        features_dir="./data/features/",
        collection_name="dquery",
        output_path=output_path,
        max_workers=5,  # Process 3 rows concurrently
        batch_size=5    # Process in batches of 5 rows
    )
    
    # Display the generated data
    if generated_data is not None:
        print("\nGenerated Data:")
        print(generated_data)
    else:
        print("Error: Could not generate data")
