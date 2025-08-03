from typing import List

from .base_index import BaseIndex
from .base_index import SearchResult


class IVFIndex(BaseIndex):
    def __init__(self, vectors, keys, texts):
        super().__init__(vectors, keys, texts)
        self.needs_rebuild = True

    def rebuild(self) -> None:
        pass

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        return None
