import llama_index
import numpy as np
import pandas as pd
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

# set up OpenAI
import os
import getpass
import dotenv
import openai

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def build_persisted_index(features_dir="./data/features/", persist_dir="./data/chroma_db", collection_name="dquery"):
    """
    Build and persist a ChromaDB index with feature documents
    
    Args:
        features_dir: Directory containing feature documents
        persist_dir: Directory to persist the ChromaDB index
        collection_name: Name of the ChromaDB collection
        
    Returns:
        query_engine: Query engine for the index
    """
    logging.info(f"Building persisted index from {features_dir}")
    
    # Create persistent client and collection
    os.makedirs(persist_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    # Check if collection exists, if not create it
    try:
        chroma_collection = chroma_client.get_collection(name=collection_name)
        logging.info(f"Using existing collection: {collection_name}")
    except:
        logging.info(f"Creating new collection: {collection_name}")
        chroma_collection = chroma_client.create_collection(name=collection_name)
    
    # define embedding function
    embed_model = OpenAIEmbedding()
    
    # load documents
    documents = SimpleDirectoryReader(features_dir).load_data()
    logging.info(f"Loaded {len(documents)} documents from {features_dir}")
    
    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    
    logging.info(f"Index built and persisted to {persist_dir}")
    return index.as_query_engine()

def load_persisted_index(persist_dir="./data/chroma_db", collection_name="dquery"):
    """
    Load a persisted ChromaDB index
    
    Args:
        persist_dir: Directory where ChromaDB index is persisted
        collection_name: Name of the ChromaDB collection
        
    Returns:
        query_engine: Query engine for the index
    """
    logging.info(f"Loading persisted index from {persist_dir}")
    
    # Check if persist directory exists
    if not os.path.exists(persist_dir):
        logging.error(f"Persist directory {persist_dir} does not exist")
        return None
    
    # Load existing index
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    try:
        chroma_collection = chroma_client.get_collection(name=collection_name)
        logging.info(f"Loaded collection: {collection_name}")
        
        # Set up vector store and index
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Define embedding function
        embed_model = OpenAIEmbedding()
        
        # Create empty index with the existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        
        logging.info("Persisted index loaded successfully")
        return index.as_query_engine()
    except Exception as e:
        logging.error(f"Error loading collection: {str(e)}")
        return None

def query_index(query_text, query_engine=None, persist_dir="./data/chroma_db", collection_name="dquery"):
    """
    Query the index with a question
    
    Args:
        query_text: Question to ask the index
        query_engine: Query engine (if None, load from persisted index)
        persist_dir: Directory where ChromaDB index is persisted
        collection_name: Name of the ChromaDB collection
        
    Returns:
        response: Response from the index
    """
    if query_engine is None:
        query_engine = load_persisted_index(persist_dir, collection_name)
        
    if query_engine is None:
        return "Error: Could not load index"
    
    logging.info(f"Querying index with: {query_text}")
    response = query_engine.query(query_text)
    return response

def get_embedding(text):
    """Generate embedding for a text string using the same embedding model used for indexing"""
    from llama_index.embeddings.openai import OpenAIEmbedding
    
    # Initialize the embedding model - make sure this matches the model used in your index
    embed_model = OpenAIEmbedding()
    
    # Get embedding
    embedding = embed_model.get_text_embedding(text)
    return embedding

def get_stored_embeddings(persist_dir="./data/chroma_db", collection_name="dquery", where_filter=None, limit=None):
    """
    Retrieve embeddings from the ChromaDB collection
    
    Args:
        persist_dir: Directory where ChromaDB index is persisted
        collection_name: Name of the ChromaDB collection
        where_filter: Optional filter condition (dict)
        limit: Optional limit on number of results
        
    Returns:
        Dict containing document IDs, documents, and their embeddings
    """
    logging.info(f"Retrieving embeddings from collection {collection_name}")
    
    # Load existing chroma client and collection
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    try:
        chroma_collection = chroma_client.get_collection(name=collection_name)
        
        # Get documents with embeddings
        results = chroma_collection.get(
            where=where_filter,
            limit=limit,
            include=["embeddings", "documents", "metadatas"]
        )
        
        logging.info(f"Retrieved {len(results.get('ids', []))} embeddings")
        return results
    except Exception as e:
        logging.error(f"Error retrieving embeddings: {str(e)}")
        return None

if __name__ == "__main__":
    # Build and persist index
    persist_dir = "./data/chroma_db"
    features_dir = "./data/features/"
    collection_name = "dquery"
    
    # Build or load the index
    if not os.path.exists(persist_dir):
        print("Building and persisting index...")
        query_engine = build_persisted_index(features_dir, persist_dir, collection_name)
    else:
        print("Loading persisted index...")
        query_engine = load_persisted_index(persist_dir, collection_name)
    
    # Test queries
    if query_engine:
        print("\nAsking question about the dataset:")
        response = query_engine.query("What are the key features in the diabetes dataset and how do they relate to each other?")
        print(f"\nResponse: {response}")
        
        print("\nAsking specific question about a feature:")
        response = query_engine.query("What is the range and distribution of glucose levels in the dataset?")
        print(f"\nResponse: {response}")
        
        print("\nAsking about feature relationships:")
        response = query_engine.query("How might BMI and glucose levels be related to diabetes outcomes?")
        print(f"\nResponse: {response}")
    else:
        print("Error: Could not load or build index.")
