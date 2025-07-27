import numpy as np
import pytest
from typing import List


class MiniVectorDb:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids: List[str] = []
        self.texts: dict = {}

    @staticmethod
    def string_to_embedding(text: str, dim: int) -> np.ndarray:
        vector = np.zeros(dim, dtype=np.float32)
        tokens = list(text)

        for token in tokens:
            index = abs(hash(token)) % dim
            vector[index] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        return vector


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
