"""
Functions for loading embeddings from parquet files.
"""

import os
import numpy as np
import pandas as pd


def load_embeddings_from_parquet(file_path):
    """
    Load vector embeddings from a parquet file.
    
    Args:
        file_path (str): Path to the parquet file containing embeddings
        
    Returns:
        numpy.ndarray: Array of embeddings
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If embeddings column cannot be found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found at {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # Try to find embedding column - common names include 'embedding', 'vector', 'embeddings'
    embedding_columns = [col for col in df.columns if 'embed' in col.lower() or 'vector' in col.lower()]
    
    if not embedding_columns:
        raise ValueError("Could not find embedding column in parquet file. "
                        "Expected column names containing 'embed' or 'vector'")
    
    embedding_col = embedding_columns[0]
    
    # Check if embeddings are stored as lists or arrays
    sample = df[embedding_col].iloc[0]
    
    if isinstance(sample, list) or isinstance(sample, np.ndarray):
        # Convert to numpy array if needed
        embeddings = np.array(df[embedding_col].tolist())
    else:
        # Handle case where embeddings might be stored in a different format
        raise ValueError(f"Unsupported embedding format: {type(sample)}")
        
    return embeddings


def load_embeddings_from_multiple_parquets(directory_path, pattern="*.parquet"):
    """
    Load embeddings from multiple parquet files in a directory.
    
    Args:
        directory_path (str): Directory containing parquet files
        pattern (str): Glob pattern to match parquet files
        
    Returns:
        numpy.ndarray: Combined array of embeddings from all files
    """
    import glob
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    parquet_files = glob.glob(os.path.join(directory_path, pattern))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {directory_path} matching {pattern}")
    
    all_embeddings = []
    
    for file_path in parquet_files:
        embeddings = load_embeddings_from_parquet(file_path)
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)