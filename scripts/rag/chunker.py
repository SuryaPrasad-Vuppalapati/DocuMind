import re
from typing import Dict, List

class TextChunker:
    def __init__(self, chunk_size: int = 1500, overlap: int = 200) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        # Split text into sentences using simple regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If current chunk plus new sentence is under limit, keep adding
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + " "
            else:
                # Chunk is full. Save it.
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                # Backtrack to grab the last few sentences roughly equal to 'overlap'
                overlap_text = ""
                words = current_chunk.split()
                if len(words) > 0:
                    overlap_words = words[-max(1, self.overlap // 5):] # roughly 5 chars per word
                    overlap_text = " ".join(overlap_words) + " "
                
                current_chunk = overlap_text + sentence + " "
                
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        # fallback for very long sentences
        final_chunks = []
        for c in chunks:
            if len(c) > self.chunk_size * 1.5:
                # If a single sentence was ridiculously long, hard split it
                start = 0
                while start < len(c):
                    final_chunks.append(c[start : start + self.chunk_size])
                    start += self.chunk_size - self.overlap
            else:
                final_chunks.append(c)
                
        return final_chunks

    def chunk_corpus(self, corpus: List[Dict]) -> List[Dict]:
        chunked: List[Dict] = []
        for entry in corpus:
            text = entry.get("text", "")
            if not text:
                continue
            for chunk in self.chunk_text(text):
                chunked.append(
                    {
                        "text": chunk,
                        "source": entry.get("source", "unknown"),
                        "page": entry.get("page", -1),
                    }
                )
        return chunked

