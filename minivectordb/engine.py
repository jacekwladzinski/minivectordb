from typing import List, Tuple, NamedTuple

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


class SearchResult(NamedTuple):
    key: str
    score: float
    text: str


class MiniVectorDb:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self, n_clusters: int = 100, n_probe: int = 10):
        self.dim = self.model.get_sentence_embedding_dimension()
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        self.keys: List[str] = []
        self.texts: dict = {}
        self.kd_tree = None
        self.needs_rebuild = False

        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.centroids: np.ndarray = None
        self.inverted_index: dict = {}

    @staticmethod
    def string_to_embedding(text: str) -> np.ndarray:
        embedding = MiniVectorDb.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def add(self, key: str, text: str) -> None:
        # stack numpy vector vertically (really slow)
        vector = MiniVectorDb.string_to_embedding(text)
        self.vectors = np.vstack([self.vectors, vector])
        self.keys.append(key)
        self.texts[key] = text
        self.needs_rebuild = True

    def add_batch(self, keys: List[str], texts: List[str], batch_size=256) -> None:
        batch_vectors = self.model.encode(texts,
                                batch_size=batch_size,
                                normalize_embeddings=True)
        batch_vectors = np.array(batch_vectors, dtype=np.float32)

        self.keys.extend(keys)
        self.texts.update({k: t for k, t in zip(keys, texts)})

        if self.vectors.size == 0:
            self.vectors = batch_vectors
        else:
            self.vectors = np.concatenate([self.vectors, batch_vectors], axis=0)

        self.needs_rebuild = True

    def delete(self, key: str) -> None:
        index = -1
        try:
            index = self.keys.index(key)
        except ValueError:
            return
        self.vectors = np.delete(self.vectors, index, axis=0)
        self.keys.pop(index)
        self.texts.pop(key)
        self.needs_rebuild = True

    def cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        return self.vectors.dot(query)

    def rebuild_tree(self) -> None:
        if self.vectors.shape[0] > 0:
            self.kd_tree = KDTree(self.vectors, metric='euclidean')
        else:
            self.kd_tree = None
        self.needs_rebuild = False

    def rebuild_ivf(self) -> None:
        n_vectors = self.vectors.shape[0]
        if n_vectors == 0:
            self.centroids = None
            self.inverted_index = {}
            self.needs_rebuild = False
            return

        n_clusters = min(self.n_clusters, n_vectors)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.vectors)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)

        inverted_index: dict = {cluster_id: [] for cluster_id in range(n_clusters)}
        for index, cluster_id in enumerate(labels):
            inverted_index[cluster_id].append(index)
        self.inverted_index = inverted_index
        self.needs_rebuild = False

    def search_linear(self, query_text: str, k: int = 5) -> List[SearchResult]:
        query = self.string_to_embedding(query_text)
        similarities = self.cosine_similarity(query)
        
        topk_index = np.argsort(-similarities)[:k]
        results = []
        for index in topk_index:
            key = self.keys[index]
            result = SearchResult(key, float(similarities[index]), self.texts.get(key))
            results.append(result)
        return results

    def search_kd_tree(self, query_text: str, k: int = 5) -> List[SearchResult]:

        n = self.vectors.shape[0]
        if n == 0:
            return []
            
        if self.needs_rebuild or self.kd_tree is None:
            self.rebuild_tree()

        query = self.string_to_embedding(query_text)

        distance, index = self.kd_tree.query(query.reshape(1, -1), k=min(k, n))
        distance = distance[0]
        index  = index[0]

        results = []
        for d, i in zip(distance, index):
            # calculate cosine similarity
            similarity = 1.0 - (d ** 2) / 2.0
            result = SearchResult(self.keys[i], float(similarity), self.texts[self.keys[i]])
            results.append(result)
        return results

    def search_ivf(self, query_text: str, k: int = 5) -> List[SearchResult]:
        if self.needs_rebuild or self.centroids is None:
            self.rebuild_ivf()
        if self.centroids is None:
            return []

        query = self.string_to_embedding(query_text)

        dists = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(dists)[:self.n_probe]

        candidates = []
        for cluster_id in nearest_clusters:
            candidates.extend(self.inverted_index.get(cluster_id, []))
        if not candidates:
            return []

        candidate_vectors = self.vectors[candidates]
        similarities = candidate_vectors.dot(query)
        topk_local = np.argsort(-similarities)[:k]
        results = []

        for local_index in topk_local:
            index = candidates[local_index]
            results.append(SearchResult(
                self.keys[index],
                float(similarities[local_index]),
                self.texts[self.keys[index]]
            ))
        return results

    def search(self, query_text: str, k: int = 5, method='kdtree') -> List[SearchResult]:
        if method == 'kdtree':
            return self.search_kd_tree(query_text, k)
        elif method == 'linear':
            return self.search_linear(query_text, k)
        else:
            return self.search_ivf(query_text, k)
