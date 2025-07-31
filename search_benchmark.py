import time
import numpy as np
import pandas as pd
from minivectordb.engine import MiniVectorDb

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

SEARCH_RESULTS_FILE = "search_results.csv"
db = MiniVectorDb()

DATASET_URL = (
    "hf://datasets/sentence-transformers/stsb/data/train-00000-of-00001.parquet"
)


def load_sentences():
    df = pd.read_parquet(DATASET_URL)

    sentences = np.concatenate(
        [df["sentence1"].to_numpy(), df["sentence2"].to_numpy()], axis=0
    )

    sentences = np.unique(sentences)
    np.random.shuffle(sentences)
    return sentences


def benchmark_method(method: str, query_texts: np.ndarray, k: int, n_repeats: int):
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

    print(f"Adding {n_vectors} entries...")

    start = time.perf_counter()

    n_batches = (int)(n_vectors / batch_size)
    for i in range(n_batches):
        keys = [str(i * batch_size + j) for j in range(batch_size)]
        texts = [sentences[i * batch_size + j] for j in range(batch_size)]
        db.add_batch(keys, texts, batch_size)

    end = time.perf_counter()
    add_time = end - start
    print(f"Add time: {add_time:.6f} seconds")

    n_repeats = 100
    query_texts = np.random.choice(sentences[: n_batches * batch_size], size=n_repeats)

    methods = ["linear", "kdtree", "ivf", "lsh"]

    # Warm-up
    start = time.perf_counter()
    for method in methods:
        db.search(query_texts[0], k, method)
    end = time.perf_counter()
    warm_up_time = end - start
    print(f"Warm-up time: {warm_up_time:.6f} seconds")

    result_dict = {}

    for method in methods:
        result_dict[method] = benchmark_method(method, query_texts, k, n_repeats)

    mismatch = {}

    for i in range(k):
        for method in methods[1:]:
            if result_dict[method] is not None:
                exact = result_dict["linear"][i].key
                approximate = result_dict[method][i].key
                mismatch[method] = exact != approximate
            else:
                mismatch[method] = True

    columns = np.concatenate([["Sentence: " + m, "Score: " + m] for m in methods])
    df = pd.DataFrame(columns=columns)

    rows = []
    for i in range(k):
        row = {}
        for method in methods:
            row[f"Sentence: {method}"] = result_dict[method][i].text
            row[f"Score: {method}"] = result_dict[method][i].score
        rows.append(row)

    df = pd.DataFrame(rows)

    print("Query: ", query_texts[n_repeats - 1])
    print(df.head())
    df.to_csv(SEARCH_RESULTS_FILE, index=False)

    for method, value in mismatch.items():
        if not mismatch[method]:
            print(f"{method} results match ✅")
        else:
            print(f"{method} results mismatch ❌")


if __name__ == "__main__":
    batch_size = 256
    benchmark_search_methods(n_vectors=2 * batch_size, batch_size=batch_size, k=5)
