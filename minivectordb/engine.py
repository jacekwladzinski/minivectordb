from typing import List, NamedTuple

import numpy as np
from sklearn.neighbors import KDTree
from sentence_transformers import SentenceTransformer


class SearchResult(NamedTuple):
    key: str
    score: float
    text: str


class MiniVectorDb:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self):
        self.dim = self.model.get_sentence_embedding_dimension()
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        self.keys: List[str] = []
        self.texts: dict = {}
        self.kd_tree = None
        self.needs_rebuild = False

    @staticmethod
    def string_to_embedding(text: str) -> np.ndarray:
        embedding = MiniVectorDb.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def add(self, key: str, text: str) -> None:
        # stack numpy vector vertically
        vector = MiniVectorDb.string_to_embedding(text)
        self.vectors = np.vstack([self.vectors, vector])
        self.keys.append(key)
        self.texts[key] = text
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
        query_normalized = query / np.linalg.norm(query)
        return self.vectors.dot(query_normalized)

    def search_linear(self, query: np.ndarray, k: int = 5) -> List[SearchResult]:
        similarities = self.cosine_similarity(query)
        
        topk_index = np.argsort(-similarities)[:k]
        results = []
        for index in topk_index:
            key = self.keys[index]
            result = SearchResult(key, float(similarities[index]), self.texts.get(key))
            results.append(result)
        return results

    def rebuild_tree(self) -> None:
        if self.vectors.shape[0] > 0:
            self.kd_tree = KDTree(self.vectors, metric='euclidean')
        else:
            self.kd_tree = None
        self.needs_rebuild = False

    def search_kd_tree(self, query: np.ndarray, k: int = 5) -> List[SearchResult]:

        n = self.vectors.shape[0]
        if n == 0:
            return []
            
        if self.needs_rebuild or self.kd_tree is None:
            self.rebuild_tree()

        query_normalized = query / np.linalg.norm(query)

        distance, index = self.kd_tree.query(query_normalized.reshape(1, -1), k=min(k, n))
        distance = distance[0]
        index  = index[0]

        results = []
        for d, i in zip(distance, index):
            # calculate cosine similarity
            similarity = 1.0 - (d ** 2) / 2.0
            result = SearchResult(self.keys[i], float(similarity), self.texts[self.keys[i]])
            results.append(result)
        return results

    def search(self, query: np.ndarray, k: int = 5, method='kdtree') -> List[SearchResult]:
        if method == 'kdtree':
            return self.search_kd_tree(query, k)
        else:
            return self.search_linear(query, k)
