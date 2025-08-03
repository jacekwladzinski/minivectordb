from typing import List

from sklearn.neighbors import KDTree

from .base_index import BaseIndex
from .base_index import SearchResult


class KDTreeIndex(BaseIndex):
    def __init__(self, vectors, keys, texts):
        super().__init__(vectors, keys, texts)
        self.kd_tree = None
        self.needs_rebuild = True

    def rebuild(self) -> None:
        if self.vectors.shape[0] > 0:
            self.kd_tree = KDTree(self.vectors, metric='euclidean')
        else:
            self.kd_tree = None
        self.needs_rebuild = False

    def search(self, query: str, k: int = 5) -> List[SearchResult]:

        n = self.vectors.shape[0]
        if n == 0:
            return []

        if self.needs_rebuild or self.kd_tree is None:
            self.rebuild()

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
