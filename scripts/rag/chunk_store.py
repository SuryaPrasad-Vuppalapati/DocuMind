import json
from typing import Dict, List


class ChunkStore:
    def save(self, chunks: List[Dict], chunks_path: str) -> None:
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def load(self, chunks_path: str) -> List[Dict]:
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

