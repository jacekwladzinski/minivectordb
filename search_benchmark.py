import time
import os
import numpy as np
import pandas as pd
from minivectordb.engine import MiniVectorDb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

RESULTS_FILE = "benchmark_results.csv"

def save_results(n_vectors, k, add_time, linear_time, kd_tree_time, top_match):

    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "n_vectors": n_vectors,
        "k": k,
        "add_time_s": add_time,
        "linear_time_s": linear_time,
        "kd_tree_time_s": kd_tree_time,
        "top_match": top_match
    }])

    if not os.path.isfile(RESULTS_FILE):
        row.to_csv(RESULTS_FILE, index=False)
    else:
        row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)


def benchmark_search_methods(n_vectors: int, batch_size: int, k: int):
    db = MiniVectorDb()

    print(f"Adding {n_vectors} entries...")

    # Add
    start = time.time()
    
    n_batches = (int)(n_vectors / batch_size)
    for i in range(n_batches):
        keys = [str(i * batch_size + j) for j in range(batch_size)]
        texts = ["Vector" + str(i * batch_size + j) for j in range(batch_size)]
        db.add_batch(keys, texts, batch_size)

    end = time.time()
    add_time = end - start
    print(f"Add time: {add_time:.6f} seconds")

    query_text = "Sample sentence to test similarity"
    query_vector = db.string_to_embedding(query_text)

    # Warm-up
    start = time.time()
    db.search_linear(query_text, k)
    db.search_kd_tree(query_text, k)
    end = time.time()
    warm_up_time = end - start
    print(f"Warm-up time: {warm_up_time:.6f} seconds")

    # Benchmark linear search
    start = time.time()
    for i in range(100):
        linear_results = db.search_linear(query_text, k)
    end = time.time()
    linear_time = end - start
    print(f"Linear Search Time: {linear_time:.6f} seconds")

    # Benchmark KD-Tree search
    start = time.time()
    for i in range(100):
        kd_tree_results = db.search_kd_tree(query_text, k)
    end = time.time()
    kd_tree_time = end - start
    print(f"KD-Tree Search Time: {kd_tree_time:.6f} seconds")

    top_match = linear_results[0][0] == kd_tree_results[0][0]
    if top_match:
        print("Top result match ✅")
    else:
        print("Top result mismatch ❌")

    save_results(
        n_vectors, k,
        add_time, linear_time, kd_tree_time,
        top_match
    )

if __name__ == "__main__":
    batch_size = 256
    benchmark_search_methods(n_vectors=40*batch_size, batch_size=batch_size, k=5)
