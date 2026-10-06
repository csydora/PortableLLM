from threading import RLock


class EmbeddingService:
    def __init__(self, config):
        self.model_path = config.embedding_model
        self.model = None
        self.lock = RLock()
        with self.lock:
            self._load()

    def _load(self):
        if self.model is None:
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"Embedding model not found: {self.model_path}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(self.model_path), device="cpu", local_files_only=True)

    def similarity(self, text1: str, text2: str) -> float:
        with self.lock:
            self._load()
            vectors = self.model.encode([text1, text2], normalize_embeddings=True)
            return float(vectors[0] @ vectors[1])

    def score_dictionary(self, sentence: str, dictionary: dict[str, str]) -> dict[str, float]:
        if not dictionary:
            return {}
        with self.lock:
            self._load()
            vectors = self.model.encode([sentence, *dictionary.values()], normalize_embeddings=True)
            return {key: float(vectors[0] @ vectors[i + 1]) for i, key in enumerate(dictionary)}

