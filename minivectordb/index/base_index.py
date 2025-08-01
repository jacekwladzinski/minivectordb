from abc import ABC, abstractmethod
from typing import List, NamedTuple

import numpy as np


class SearchResult(NamedTuple):
    key: str
    score: float
    text: str

class BaseIndex(ABC):
    def __init__(self):
        self.needs_rebuild = True

    @abstractmethod
    def rebuild(self, vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def search(self, query_text: np.ndarray, k: int) -> List[SearchResult]:
        pass
