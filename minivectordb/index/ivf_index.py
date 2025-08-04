from typing import List

import numpy as np
from sklearn.cluster import KMeans

from .base_index import BaseIndex
from .base_index import SearchResult


class IVFIndex(BaseIndex):
    def __init__(self,
                 vectors,
                 keys,
                 texts,
                 n_clusters: int = 100,
                 n_probe: int = 10):
        super().__init__(vectors, keys, texts)
        self.needs_rebuild = True

        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.centroids: np.ndarray = None
        self.inverted_index: dict = {}

    def rebuild(self) -> None:
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

    def search_ivf(self, query, k: int = 5) -> List[SearchResult]:
        # https://developer.nvidia.com/blog/accelerated-vector-search-approximating-with-nvidia-cuvs-ivf-flat/
        if self.needs_rebuild or self.centroids is None:
            self.rebuild_ivf()
        if self.centroids is None:
            return []

        distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(distances)[:self.n_probe]

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
