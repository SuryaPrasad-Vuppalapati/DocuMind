from typing import Dict, List

import faiss
import numpy as np
from openai import OpenAI
import concurrent.futures
from .cost import track_cost


class Embedder:
    def __init__(
        self, client: OpenAI, embedding_model: str = "text-embedding-3-small"
    ) -> None:
        self.client = client
        self.embedding_model = embedding_model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model, input=texts
        )
        track_cost(response, is_embedding=True)
        return [item.embedding for item in response.data]

    def embed_chunks(self, chunks: List[Dict], batch_size: int = 50) -> np.ndarray:
        texts = [chunk["text"] for chunk in chunks]
        vectors = []
        
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            batch_results = list(executor.map(self.get_embeddings, batches))
            
        for batch_vec in batch_results:
            vectors.extend(batch_vec)
            
        array = np.array(vectors, dtype="float32")
        faiss.normalize_L2(array)
        return array

    def embed_query(self, query: str) -> np.ndarray:
        query_vec = np.array(self.get_embeddings([query]), dtype="float32")
        faiss.normalize_L2(query_vec)
        return query_vec

