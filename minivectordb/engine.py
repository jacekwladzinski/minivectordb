import numpy as np
from typing import List, Tuple

from sklearn.neighbors import KDTree


class MiniVectorDb:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self):
        self.dim = self.model.get_sentence_embedding_dimension()
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids: List[str] = []
        self.texts: dict = {}

    @staticmethod
    def string_to_embedding(text: str) -> np.ndarray:
        embedding = MiniVectorDb.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def add(self, id: str, vector: np.ndarray, text: str):
        # stack numpy vector vertically
        self.vectors = np.vstack([self.vectors, vector])
        self.ids.append(id)
        self.texts[id] = text

    def delete(self, id: str):
        index = self.ids.index(id)
        self.vectors = np.delete(self.vectors, index, axis=0)
        self.ids.pop(index)
        self.texts.pop(id)

    def cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        query_normalized = query / np.linalg.norm(query)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        vectors_normalized = self.vectors / np.clip(norms, 1e-9, None)
        return vectors_normalized.dot(query_normalized)

    def search_linear(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float, str]]:
        similarities = self.cosine_similarity(query)
        
        topk_index = np.argsort(-similarities)[:k]
        results = []
        for idx in topk_index:
            id = self.ids[idx]
            results.append((id, float(similarities[idx]), self.texts.get(id)))
        return results

    def search_kd_tree(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float, str]]:

        n = self.vectors.shape[0]
        if n == 0:
            return []

        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        vectors_normalized = self.vectors / np.clip(norms, 1e-9, None)

        tree = KDTree(vectors_normalized, metric='euclidean')

        query_normalized = query / np.linalg.norm(query)

        distance, index = tree.query(query_normalized.reshape(1, -1), k=min(k, n))
        distance = distance[0]
        index  = index[0]

        results = []
        for d, i in zip(distance, index):
            vector_id = self.ids[i]
            # calculate cosine similarity
            similarity = 1.0 - (d ** 2) / 2.0
            results.append((vector_id, float(similarity), self.texts[vector_id]))
        return results

    def search(self, query: np.ndarray, k: int = 5, method='kdtree') -> List[Tuple[str, float, str]]:
        if method == 'kdtree':
            return self.search_kd_tree(query, k)
        else:
            return self.search_linear(query, k)
