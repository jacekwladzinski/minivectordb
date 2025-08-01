from abc import ABC, abstractmethod
from typing import List, NamedTuple


class SearchResult(NamedTuple):
    key: str
    score: float
    text: str


class BaseIndex(ABC):
    def __init__(self, vectors, keys, texts):
        self.vectors = vectors
        self.keys = keys
        self.texts = texts
        self.needs_rebuild = True

    @abstractmethod
    def rebuild(self) -> None:
        pass

    @abstractmethod
    def search(self, query_text: str, k: int) -> List[SearchResult]:
        pass
