import json
from typing import Dict, List


class CorpusLoader:
    def __init__(self, corpus_path: str = "data/corpus.json") -> None:
        self.corpus_path = corpus_path

    def load(self) -> List[Dict]:
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            return json.load(f)

