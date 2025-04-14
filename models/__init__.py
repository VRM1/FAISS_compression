"""
Search models module for vector similarity search.
"""

from .cosine_search import brute_force_cosine_search
from .faiss_flat import faiss_flat_l2_search, save_flat_index, load_flat_index
from .faiss_ivf import (
    faiss_ivf_search, 
    train_ivf_index, 
    add_vectors_to_index,
    get_searched_regions,
    save_ivf_index, 
    load_ivf_index
)