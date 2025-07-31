import time
import numpy as np
import pandas as pd
from minivectordb.engine import MiniVectorDb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

RESULTS_FILE = "benchmark_results.csv"
db = MiniVectorDb()


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
    np.random.shuffle(sentences)
    return sentences


def benchmark_method(
        method: str,
        query_texts: np.ndarray,
        k: int,
        n_repeats: int):

    start = time.perf_counter()
    for i in range(n_repeats):
        results = db.search(query_texts[i], k, method=method)
    end = time.perf_counter()

    search_time = end - start
    print(f"{method} search time: {search_time:.6f} seconds")
    return results


def benchmark_search_methods(n_vectors: int, batch_size: int, k: int):
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

    methods = ['linear', 'kdtree', 'ivf', 'lsh', 'hnsw']

    # Warm-up
    start = time.time()
    for method in methods:
        db.search(query_texts[0], k, method)
    end = time.time()
    warm_up_time = end - start
    print(f"Warm-up time: {warm_up_time:.6f} seconds")

    result_dict = {}

    for method in methods:
        result_dict[method] = benchmark_method(method, query_texts, k, n_repeats)

    mismatch_kd_tree = False
    mismatch_ivf = False
    mismatch_lsh = False
    mismatch_hnsw = False

    for i in range(k):
        if result_dict['linear'][i].key != result_dict['kdtree'][i].key:
            mismatch_kd_tree = True
        if result_dict['linear'][i].key != result_dict['ivf'][i].key:
            mismatch_ivf = True
        if result_dict['linear'][i].key != result_dict['lsh'][i].key:
            mismatch_lsh = True
        if result_dict['linear'][i].key != result_dict['hnsw'][i].key:
            mismatch_hnsw = True

    df = pd.DataFrame(columns=[
        'Sentence: linear',
        'Score: linear',
        'Sentence: KD tree',
        'Score: KD tree',
        'Sentence: IVF',
        'Score: IVF',
        'Sentence: LSH',
        'Score: LSH',
        'Sentence: HNSW',
        'Score: HNSW'
        ]
    )

    for i in range(k):
        row = pd.DataFrame([{
            'Sentence: linear': result_dict['linear'][i].text,
            'Score: linear': result_dict['linear'][i].score,
            'Sentence: KD tree': result_dict['kdtree'][i].text,
            'Score: KD tree': result_dict['kdtree'][i].score,
            'Sentence: IVF': result_dict['ivf'][i].text,
            'Score: IVF': result_dict['ivf'][i].score,
            'Sentence: LSH': result_dict['lsh'][i].text,
            'Score: LSH': result_dict['lsh'][i].score,
            'Sentence: HNSW': result_dict['hnsw'][i].text,
            'Score: HNSW': result_dict['hnsw'][i].score,
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
        print("IVF results mismatch ❌")

    if not mismatch_lsh:
        print("LSH results match ✅")
    else:
        print("LSH results mismatch ❌")

    if not mismatch_hnsw:
        print("HNSW results match ✅")
    else:
        print("HNSW results mismatch ❌")


if __name__ == "__main__":
    batch_size = 256
    benchmark_search_methods(n_vectors=2*batch_size, batch_size=batch_size, k=5)
