from typing import List

import numpy as np

from .base_index import BaseIndex
from .base_index import SearchResult


class LinearIndex(BaseIndex):
    def __init__(self, vectors, keys, texts):
        super().__init__(vectors, keys, texts)
        self.needs_rebuild = True

    def rebuild(self) -> None:
        pass

    def search(self, query, k) -> List[SearchResult]:
        similarities = self.vectors.dot(query)

        topk_index = np.argsort(-similarities)[:k]
        results = []
        for index in topk_index:
            key = self.keys[index]
            result = SearchResult(key, float(similarities[index]), self.texts.get(key))
            results.append(result)
        return results
