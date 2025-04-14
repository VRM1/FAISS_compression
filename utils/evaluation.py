"""
Evaluation utilities for comparing search methods.
"""

import numpy as np


def compare_search_results(results1, results2, method1_name="Method 1", method2_name="Method 2"):
    """
    Compare search results between two different methods.
    
    Args:
        results1 (numpy.ndarray): Search results from first method
        results2 (numpy.ndarray): Search results from second method
        method1_name (str): Name of the first method
        method2_name (str): Name of the second method
        
    Returns:
        tuple: (common_matches, percentage_agreement)
    """
    common_matches = set(results1).intersection(set(results2))
    percentage_agreement = len(common_matches) / len(results1) * 100
    
    print(f"Comparing results between {method1_name} and {method2_name}:")
    print(f"Common matches: {len(common_matches)} out of {len(results1)} ({percentage_agreement:.2f}%)")
    print(f"Common match indices: {common_matches}")
    
    return common_matches, percentage_agreement


def evaluate_search_performance(search_func, vectors, queries, k=5, **kwargs):
    """
    Evaluate search performance on multiple queries.
    
    Args:
        search_func (callable): Search function to evaluate
        vectors (numpy.ndarray): Database vectors
        queries (numpy.ndarray): Query vectors
        k (int): Number of nearest neighbors to retrieve
        **kwargs: Additional arguments to pass to the search function
        
    Returns:
        tuple: (avg_time_per_query, all_results)
    """
    import time
    
    n_queries = len(queries)
    all_results = []
    
    start_time = time.time()
    
    for i, query in enumerate(queries):
        results, _ = search_func(vectors, query, k, **kwargs)
        all_results.append(results)
        
    total_time = time.time() - start_time
    avg_time_per_query = total_time / n_queries
    
    return avg_time_per_query, all_results


def run_performance_tests(vector_sizes, dimension, n_queries, search_methods):
    """
    Run performance tests on multiple search methods with different vector sizes.
    
    Args:
        vector_sizes (list): List of vector sizes to test
        dimension (int): Dimensionality of vectors
        n_queries (int): Number of queries for each test
        search_methods (dict): Dictionary mapping method names to search functions
        
    Returns:
        dict: Dictionary with performance results
    """
    from ..dataset.synthetic_data import generate_sample_vectors
    import time
    
    # Dictionary to store results
    results = {
        'sizes': vector_sizes,
    }
    
    # Initialize result lists for each method
    for method_name in search_methods.keys():
        results[method_name] = []
    
    for size in vector_sizes:
        print(f"\nTesting with {size} vectors and {n_queries} queries...")
        vectors = generate_sample_vectors(size, dimension)
        query_vectors = generate_sample_vectors(n_queries, dimension)
        
        for method_name, search_func in search_methods.items():
            # Special handling for IVF
            if 'ivf' in method_name.lower():
                n_regions = min(size // 100, 1000)  # Heuristic for number of regions
                kwargs = {'n_regions': n_regions, 'nprobe': min(n_regions, 5)}
            else:
                kwargs = {}
                
            start_time = time.time()
            
            for query in query_vectors:
                if isinstance(query, np.ndarray) and len(query.shape) == 1:
                    # Make sure query is in the right shape (some methods expect batch)
                    query_formatted = query.reshape(1, -1) if 'flat' in method_name.lower() else query
                else:
                    query_formatted = query
                    
                _, _ = search_func(vectors, query_formatted, k=5, **kwargs)
                
            method_time = (time.time() - start_time) / n_queries
            results[method_name].append(method_time)
            
            print(f"{method_name}: {method_time:.6f} seconds per query")
    
    return results