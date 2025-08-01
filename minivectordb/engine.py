from typing import List
import heapq

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from .index.base_index import SearchResult
from .index.base_index import BaseIndex


class MiniVectorDb:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(
        self,
        n_clusters: int = 100,
        n_probe: int = 10,
        n_hash_tables: int = 4,
        hash_size: int = 8,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ):
        self.dim = self.model.get_sentence_embedding_dimension()
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        self.keys: List[str] = []
        self.texts: dict = {}

        # KD tree
        self.kd_tree = None

        # IVF
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.centroids: np.ndarray = None
        self.inverted_index: dict = {}

        # LSH
        self.n_hash_tables = n_hash_tables
        self.hash_size = hash_size
        self.lsh_planes = None
        self.lsh_tables = None

        # HNSW
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.node_levels: List[int] = []
        self.max_level: int = 0
        # adjacency: list of dicts per layer -> node_id -> list of neighbor ids
        self.hnsw_graph: List[dict] = []
        self.entry_point: int = 0

        self.needs_rebuild = False

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

    def rebuild_lsh(self) -> None:
        n_vectors = self.vectors.shape[0]

        if n_vectors == 0:
            self.lsh_planes = None
            self.lsh_tables = []
            self.needs_rebuild = False
            return

        rng = np.random.RandomState(42)

        self.lsh_planes = [
            rng.randn(self.hash_size, self.dim).astype(np.float32)
            for _ in range(self.n_hash_tables)
        ]
        self.lsh_tables = []

        # build hash tables
        for planes in self.lsh_planes:
            projections = np.dot(self.vectors, planes.T)  # (n, hash_size)
            bits = projections >= 0  # boolean mask
            table: dict = {}
            for idx, hash_bits in enumerate(bits):
                key = tuple(int(b) for b in hash_bits)
                table.setdefault(key, []).append(idx)
            self.lsh_tables.append(table)

        self.needs_rebuild = False

    def _get_random_level(self) -> int:
        lvl = 0
        # geometric distribution with p=1/hnsw_m
        while np.random.rand() < 1.0 / self.hnsw_m:
            lvl += 1
        return lvl

    def rebuild_hnsw(self) -> None:
        n = self.vectors.shape[0]
        if n == 0:
            self.node_levels = []
            self.hnsw_graph = []
            self.entry_point = 0
            self.max_level = 0
            self.needs_rebuild = False
            return

        self.node_levels = [self._get_random_level() for _ in range(n)]
        self.max_level = max(self.node_levels)

        self.hnsw_graph = [dict() for _ in range(self.max_level + 1)]

        for level in range(self.max_level + 1):
            nodes = [i for i, l in enumerate(self.node_levels) if l >= level]
            for node in nodes:
                vectors = self.vectors[nodes]
                distances = np.linalg.norm(vectors - self.vectors[node], axis=1)
                if len(distances) <= 1:
                    neighbors = []
                else:
                    idx_sorted = np.argsort(distances)

                    selected = idx_sorted[1:self.hnsw_m + 1]
                    neighbors = [nodes[j] for j in selected]
                self.hnsw_graph[level][node] = neighbors

        self.entry_point = max(range(n), key=lambda i: self.node_levels[i])
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
        index = index[0]

        results = []
        for d, i in zip(distance, index):
            # calculate cosine similarity
            similarity = 1.0 - (d ** 2) / 2.0
            result = SearchResult(self.keys[i], float(similarity), self.texts[self.keys[i]])
            results.append(result)
        return results

    def search_ivf(self, query_text: str, k: int = 5) -> List[SearchResult]:
        # https://developer.nvidia.com/blog/accelerated-vector-search-approximating-with-nvidia-cuvs-ivf-flat/
        if self.needs_rebuild or self.centroids is None:
            self.rebuild_ivf()
        if self.centroids is None:
            return []

        query = self.string_to_embedding(query_text)

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

    def search_lsh(self, query_text: str, k: int = 5) -> List[SearchResult]:
        # rebuild LSH tables if needed
        if self.needs_rebuild or self.lsh_planes is None:
            self.rebuild_lsh()
        if not self.lsh_tables:
            return []

        query = self.string_to_embedding(query_text)
        candidates = set()
        # probe each hash table
        for planes, table in zip(self.lsh_planes, self.lsh_tables):
            proj = np.dot(planes, query)  # (hash_size,)
            bits = proj >= 0
            key = tuple(int(b) for b in bits)
            candidates.update(table.get(key, []))

        if not candidates:
            # fallback to brute-force
            return self.search_linear(query_text, k)

        candidates = list(candidates)
        cand_vecs = self.vectors[candidates]
        sims = cand_vecs.dot(query)
        top_local = np.argsort(-sims)[:k]

        return [
            SearchResult(
                self.keys[candidates[i]],
                float(sims[i]),
                self.texts[self.keys[candidates[i]]]
            ) for i in top_local
        ]

    def _search_layer_greedy(self, query: np.ndarray, entry: int, level: int) -> int:
        curr = entry
        curr_dist = np.linalg.norm(self.vectors[curr] - query)
        improved = True
        while improved:
            improved = False
            for nbr in self.hnsw_graph[level].get(curr, []):
                d = np.linalg.norm(self.vectors[nbr] - query)
                if d < curr_dist:
                    curr_dist = d
                    curr = nbr
                    improved = True
        return curr

    def search_hnsw(self, query_text: str, k: int = 5) -> List[SearchResult]:
        if self.needs_rebuild or not self.hnsw_graph:
            self.rebuild_hnsw()
        n = self.vectors.shape[0]
        if n == 0:
            return []
        query = self.string_to_embedding(query_text)

        entry = self.entry_point
        for level in range(self.max_level, 0, -1):
            entry = self._search_layer_greedy(query, entry, level)

        candidates = [(np.linalg.norm(self.vectors[entry] - query), entry)]
        visited = {entry}

        topk_heap = [(-candidates[0][0], entry)]
        heapq.heapify(topk_heap)
        while candidates:
            dist, cur = heapq.heappop(candidates)
            # early stop
            if len(topk_heap) >= self.hnsw_ef_search and dist > -topk_heap[0][0]:
                break
            for neighbour in self.hnsw_graph[0].get(cur, []):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                d = np.linalg.norm(self.vectors[neighbour] - query)
                heapq.heappush(candidates, (d, neighbour))
                if len(topk_heap) < self.hnsw_ef_search:
                    heapq.heappush(topk_heap, (-d, neighbour))
                else:
                    if d < -topk_heap[0][0]:
                        heapq.heapreplace(topk_heap, (-d, neighbour))

        top_candidates = heapq.nsmallest(k, [(-s, index) for s, index in topk_heap])
        results = []
        for dist, idx in top_candidates:
            results.append(SearchResult(self.keys[idx],
                                        float(1.0 - (dist**2) / 2.0),
                                        self.texts[self.keys[idx]]))

    def search(self, query_text: str, k: int = 5, method='lsh') -> List[SearchResult]:
        if method == 'linear':
            return self.search_linear(query_text, k)
        elif method == 'kdtree':
            return self.search_kd_tree(query_text, k)
        elif method == 'ivf':
            return self.search_ivf(query_text, k)
        elif method == 'hnsw':
            return self.search_hnsw(query_text, k)
        else:
            return self.search_lsh(query_text, k)
