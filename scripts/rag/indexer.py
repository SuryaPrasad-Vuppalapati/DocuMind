import faiss
import numpy as np


class FaissIndexer:
    def build(self, vectors: np.ndarray) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def save(self, index: faiss.Index, index_path: str) -> None:
        faiss.write_index(index, index_path)

    def load(self, index_path: str) -> faiss.Index:
        return faiss.read_index(index_path)

