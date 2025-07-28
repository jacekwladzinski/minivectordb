import time
import numpy as np
from minivectordb.engine import MiniVectorDb

def benchmark_search_methods(n_vectors: int, k: int):
    db = MiniVectorDb()

    print(f"Adding {n_vectors} entries...")
    sample_text = "This is a sample sentence for embedding."

    # Add
    start = time.time()
    for i in range(n_vectors):
        db.add(f"id_{i}", f"{sample_text} #{i}")
    end = time.time()
    add_time = end - start
    print(f"Add time: {add_time:.6f} seconds")

    query_text = "Sample sentence to test similarity"
    query_vector = db.string_to_embedding(query_text)

    # Warm-up
    start = time.time()
    db.search_linear(query_vector, k)
    db.search_kd_tree(query_vector, k)
    end = time.time()
    warm_up_time = end - start
    print(f"Warm-up time: {warm_up_time:.6f} seconds")

    # Benchmark linear search
    start = time.time()
    for i in range(100):
        linear_results = db.search_linear(query_vector, k)
    end = time.time()
    linear_time = end - start
    print(f"Linear Search Time: {linear_time:.6f} seconds")

    # Benchmark KD-Tree search
    start = time.time()
    for i in range(100):
        kd_tree_results = db.search_kd_tree(query_vector, k)
    end = time.time()
    kd_tree_time = end - start
    print(f"KD-Tree Search Time: {kd_tree_time:.6f} seconds")

    if linear_results[0][0] == kd_tree_results[0][0]:
        print("Top result match ✅")
    else:
        print("Top result mismatch ❌")

if __name__ == "__main__":
    benchmark_search_methods(n_vectors=1000, k=5)
