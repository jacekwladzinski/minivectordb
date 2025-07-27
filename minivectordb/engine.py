import numpy as np
from typing import List


class MiniVectorDb:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids: List[str] = []
        self.texts: dict = {}

    def add(self, id: str, vector: np.ndarray, text: str):
        # stack numpy vector vertically
        self.vectors = np.vstack([self.vectors, vector])
        self.ids.append(id)
        self.texts[id] = text
