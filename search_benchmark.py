import time
import os
import pandas as pd
from minivectordb.engine import MiniVectorDb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

RESULTS_FILE = "benchmark_results.csv"


def save_results(n_vectors, k, add_time,
                 linear_time, kd_tree_time, ivf_time,
                 top_match_kd_tree, top_match_ivf):

    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "n_vectors": n_vectors,
        "k": k,
        "add_time_s": add_time,
        "linear_time_s": linear_time,
        "kd_tree_time_s": kd_tree_time,
        "ivf_time_s": ivf_time,
        "top_match_kd_tree": top_match_kd_tree,
        "top_match_ivf": top_match_ivf
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

    # Warm-up
    start = time.time()
    db.search_linear(query_text, k)
    db.search_kd_tree(query_text, k)
    db.search_ivf(query_text, k)
    end = time.time()
    warm_up_time = end - start
    print(f"Warm-up time: {warm_up_time:.6f} seconds")

    # Benchmark linear search
    start = time.time()
    for i in range(100):
        linear_results = db.search_linear(query_text, k)
    end = time.time()
    linear_time = end - start
    print(f"Linear search Time: {linear_time:.6f} seconds")

    # Benchmark KD-Tree search
    start = time.time()
    for i in range(100):
        kd_tree_results = db.search_kd_tree(query_text, k)
    end = time.time()
    kd_tree_time = end - start
    print(f"KD-Tree search Time: {kd_tree_time:.6f} seconds")

    # Benchmark IVF
    start = time.time()
    for i in range(100):
        ivf_results = db.search_ivf(query_text, k)
    end = time.time()
    ivf_time = end - start
    print(f"IVF search Time: {ivf_time:.6f} seconds")

    top_match_kd_tree = linear_results[0][0] == kd_tree_results[0][0]
    top_match_ivf = linear_results[0][0] == ivf_results[0][0]

    if top_match_kd_tree:
        print("KD Tree top result match ✅")
    else:
        print("KD Tree top result mismatch ❌")

    if top_match_ivf:
        print("IVF top result match ✅")
    else:
        print("IVF Tree top result mismatch ❌")

    save_results(
        n_vectors, k,
        add_time,
        linear_time, kd_tree_time, ivf_time,
        top_match_kd_tree, top_match_ivf
    )


if __name__ == "__main__":
    batch_size = 256
    benchmark_search_methods(n_vectors=4*batch_size, batch_size=batch_size, k=5)
