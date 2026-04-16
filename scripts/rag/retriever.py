from typing import Dict, List

import faiss
import numpy as np

from .embedder import Embedder

class Retriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def retrieve(
        self, 
        query: str, 
        index: faiss.Index, 
        chunks: List[Dict], 
        bm25=None, 
        method: str="vector", 
        k: int = 10,
        pre_filters: Dict = None,
        post_filters: Dict = None
    ) -> List[Dict]:
        # Always use the main query
        queries = [query]
        
        # Simple extraction for "compare A and B"
        query_lower = query.lower()
        if "compare" in query_lower or "vs" in query_lower or "difference" in query_lower:
            import re
            match = re.search(r'(?:compare\s+(?:between\s+)?|difference\s+between\s+)?(.+?)\s+(?:and|vs\.?)\s+(.+)', query_lower)
            if match:
                q1 = match.group(1).replace("what is the", "").replace("can you", "").strip()
                q2 = match.group(2).replace("?", "").strip()
                if len(q1) > 2 and len(q2) > 2:
                    queries.append(q1)
                    queries.append(q2)

        # Stage 1: Pre-filter (indexed, selective)
        allowed_ids = None
        if pre_filters:
            _allowed = []
            for i, chunk in enumerate(chunks):
                match = True
                for k_filt, v_filt in pre_filters.items():
                    if chunk.get(k_filt) != v_filt:
                        match = False
                        break
                if match:
                    _allowed.append(i)
            # if _allowed is empty, no chunk matches; we return fast.
            if not _allowed:
                return []
            allowed_ids = np.array(_allowed, dtype=np.int64)

        # Helper for vector search
        def search_vector():
            query_vecs = self.embedder.get_embeddings(queries)
            query_vecs_np = np.array(query_vecs, dtype="float32")
            faiss.normalize_L2(query_vecs_np)
            
            # Stage 2: ANN vector search with semantic ranking on the filtered subset
            search_k = k * 3 if post_filters else k
            
            if allowed_ids is not None:
                sel = faiss.IDSelectorArray(allowed_ids)
                try:
                    params = faiss.SearchParameters(sel=sel)
                    scores, indices = index.search(query_vecs_np, k=search_k, params=params)
                except AttributeError:
                    # fallback for older faiss or IVF variations
                    params = faiss.SearchParametersIVF(sel=sel)
                    scores, indices = index.search(query_vecs_np, k=search_k, params=params)
            else:
                scores, indices = index.search(query_vecs_np, k=search_k)
            
            result_ranks = {}
            for q_idx in range(len(queries)):
                result_ranks[q_idx] = []
                for i, chunk_idx in enumerate(indices[q_idx]):
                    if chunk_idx >= 0: # Ensure valid FAISS id (-1 means no more neighbors)
                        result_ranks[q_idx].append((int(chunk_idx), float(scores[q_idx][i])))
            return result_ranks

        # Helper for bm25 search (Stage 1 / 2)
        def search_bm25():
            result_ranks = {}
            for q_idx, q in enumerate(queries):
                tokenized_query = q.lower().split()
                scores = bm25.get_scores(tokenized_query)
                
                # Apply Stage 1 Pre-filters to BM25 scores manually
                if allowed_ids is not None:
                    allowed_set = set(allowed_ids.tolist())
                    for i in range(len(scores)):
                        if i not in allowed_set:
                            scores[i] = -float('inf')

                # Stage 2 for BM25
                search_k = k * 3 if post_filters else k
                top_n = np.argsort(scores)[::-1][:search_k]
                result_ranks[q_idx] = [(int(idx), float(scores[idx])) for idx in top_n if scores[idx] > -float('inf')]
            return result_ranks

        # Execute selected methods
        vec_results = search_vector() if method in ["vector", "hybrid"] else None
        bm25_results = search_bm25() if method in ["bm25", "hybrid"] and bm25 is not None else None

        # Function to combine ranks for a specific query using RRF
        def get_rrf_combined(q_idx):
            combined_scores = {}
            rrf_k = 60
            if vec_results:
                for rank, (chunk_idx, _) in enumerate(vec_results[q_idx]):
                    combined_scores[chunk_idx] = combined_scores.get(chunk_idx, 0) + 1 / (rrf_k + rank + 1)
            if bm25_results:
                for rank, (chunk_idx, _) in enumerate(bm25_results[q_idx]):
                    combined_scores[chunk_idx] = combined_scores.get(chunk_idx, 0) + 1 / (rrf_k + rank + 1)
            
            sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            search_k = k * 3 if post_filters else k
            return sorted_combined[:search_k]

        # Get final ranked indices per query
        query_top_indices = {}
        for q_idx in range(len(queries)):
            if method == "hybrid":
                query_top_indices[q_idx] = get_rrf_combined(q_idx)
            elif method == "bm25" and bm25_results:
                query_top_indices[q_idx] = bm25_results[q_idx]
            elif vec_results:
                query_top_indices[q_idx] = vec_results[q_idx]
            else:
                query_top_indices[q_idx] = []

        final_results = []
        seen_chunks = set()

        def apply_post_filter(chunk_data):
            if not post_filters: return True
            for pk, pv in post_filters.items():
                if chunk_data.get(pk) != pv: return False
            return True

        if len(queries) == 1:
            # Normal query: resolve up to k items, passing the post-filter
            for chunk_idx, score in query_top_indices[0]:
                if chunk_idx not in seen_chunks:
                    result = dict(chunks[chunk_idx])
                    # Stage 3: Post-filter (Lightweight refinement)
                    if apply_post_filter(result):
                        result["score"] = score
                        final_results.append(result)
                        seen_chunks.add(chunk_idx)
                        if len(final_results) == k:
                            break
        else:
            # Comparison: top k from A (q1=1) and top k from B (q2=2)
            limit = k // 2 if k > 1 else 1 # Adjust dynamically to fill k items
            for q_idx in [1, 2]:
                added = 0
                for chunk_idx, score in query_top_indices[q_idx]:
                    if chunk_idx not in seen_chunks:
                        result = dict(chunks[chunk_idx])
                        if apply_post_filter(result):
                            result["score"] = score
                            final_results.append(result)
                            seen_chunks.add(chunk_idx)
                            added += 1
                            if added == limit:
                                break
                            
        return final_results
