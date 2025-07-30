import time
import os
import numpy as np
import pandas as pd
from minivectordb.engine import MiniVectorDb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

RESULTS_FILE = "benchmark_results.csv"


def load_sentences():
    splits = {
        'train': 'data/train-00000-of-00001.parquet',
        'validation': 'data/validation-00000-of-00001.parquet',
        'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/sentence-transformers/stsb/" + splits["train"])

    sentences = np.concatenate(
        [df['sentence1'].to_numpy(), df['sentence2'].to_numpy()],
        axis=0
    )

    sentences = np.unique(sentences)
    sentences = np.random.shuffle(sentences)
    return sentences


def save_results(n_vectors, k, add_time,
                 linear_time, kd_tree_time, ivf_time,
                 mismatch_kd_tree, mismatch_ivf):

    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "n_vectors": n_vectors,
        "k": k,
        "add_time_s": add_time,
        "linear_time_s": linear_time,
        "kd_tree_time_s": kd_tree_time,
        "ivf_time_s": ivf_time,
        "mismatch_kd_tree": mismatch_kd_tree,
        "mismatch_ivf": mismatch_ivf
    }])

    if not os.path.isfile(RESULTS_FILE):
        row.to_csv(RESULTS_FILE, index=False)
    else:
        row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)


def benchmark_search_methods(n_vectors: int, batch_size: int, k: int):
    db = MiniVectorDb()
    sentences = load_sentences()
    n_vectors = min(n_vectors, sentences.size)

    print(f"Sentences: {sentences.size}")
    print(f"Adding {n_vectors} entries...")

    # Add
    start = time.time()

    n_batches = (int)(n_vectors / batch_size)
    for i in range(n_batches):
        keys = [str(i * batch_size + j) for j in range(batch_size)]
        texts = [sentences[i * batch_size + j] for j in range(batch_size)]
        db.add_batch(keys, texts, batch_size)

    end = time.time()
    add_time = end - start
    print(f"Add time: {add_time:.6f} seconds")

    n_repeats = 100
    query_texts = np.random.choice(sentences[:n_batches * batch_size], size=n_repeats)

    # Warm-up
    start = time.time()
    db.search_linear(query_texts[0], k)
    db.search_kd_tree(query_texts[0], k)
    db.search_ivf(query_texts[0], k)
    end = time.time()
    warm_up_time = end - start
    print(f"Warm-up time: {warm_up_time:.6f} seconds")

    # Benchmark linear search
    start = time.time()
    for i in range(n_repeats):
        linear_results = db.search_linear(query_texts[i], k)
    end = time.time()
    linear_time = end - start
    print(f"Linear search Time: {linear_time:.6f} seconds")

    # Benchmark KD-Tree search
    start = time.time()
    for i in range(n_repeats):
        kd_tree_results = db.search_kd_tree(query_texts[i], k)
    end = time.time()
    kd_tree_time = end - start
    print(f"KD-Tree search Time: {kd_tree_time:.6f} seconds")

    # Benchmark IVF
    start = time.time()
    for i in range(n_repeats):
        ivf_results = db.search_ivf(query_texts[i], k)
    end = time.time()
    ivf_time = end - start
    print(f"IVF search Time: {ivf_time:.6f} seconds")

    mismatch_kd_tree = False
    mismatch_ivf = False

    for i in range(k):
        if linear_results[i].key != kd_tree_results[i].key:
            mismatch_kd_tree = True
        if linear_results[i].key != ivf_results[i].key:
            mismatch_ivf = True

    df = pd.DataFrame(columns=[
        'Sentence: linear',
        'Score: linear',
        'Sentence: KD tree',
        'Score: KD tree',
        'Sentence: IVF',
        'Score: IVF']
    )

    for i in range(k):
        row = pd.DataFrame([{
            'Sentence: linear': linear_results[i].text,
            'Score: linear': linear_results[i].score,
            'Sentence: KD tree': kd_tree_results[i].text,
            'Score: KD tree': kd_tree_results[i].score,
            'Sentence: IVF': ivf_results[i].text,
            'Score: IVF': ivf_results[i].score,
        }])

        df = pd.concat([df, row], ignore_index=True)

    print('Query: ', query_texts[n_repeats - 1])
    print(df.head())
    df.to_csv("search_results.csv", index=False)

    if not mismatch_kd_tree:
        print("KD Tree results match ✅")
    else:
        print("KD Tree results mismatch ❌")

    if not mismatch_ivf:
        print("IVF results match ✅")
    else:
        print("IVF Tree results mismatch ❌")

    save_results(
        n_vectors, k,
        add_time,
        linear_time, kd_tree_time, ivf_time,
        mismatch_kd_tree, mismatch_ivf
    )


if __name__ == "__main__":
    batch_size = 256
    benchmark_search_methods(n_vectors=2*batch_size, batch_size=batch_size, k=5)
