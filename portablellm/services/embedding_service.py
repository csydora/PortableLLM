from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from threading import RLock

class EmbeddingService:
    def __init__(self, config):
        self.model_path = config.embedding_model
        self.model = SentenceTransformer(str(self.model_path))
        self.cache = {}
        self.lock = RLock()
        with self.lock:
            self._load()

    def _load(self):
        if self.model is None:
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"Embedding model not found: {self.model_path}")
            self.model = SentenceTransformer(str(self.model_path), device="cpu", local_files_only=True)

    def similarity(self, text1: str, text2: str) -> float:
        with self.lock:
            e1 = self.model.encode(text1)
            e2 = self.model.encode(text2)

            return float(dot(e1, e2) / (norm(e1) * norm(e2)))

    def score_dictionary(self, sentence: str, dictionary: dict[str, str]) -> dict[str, float]:
        sentence_embedding = self.model.encode(sentence)
        results = {}
        with self.lock:
            for word, definition in dictionary.items():
                if word not in self.cache:
                   self.cache[word] = self.model.encode(definition)
                
                embedding = self.cache[word]
                score = float(dot(sentence_embedding, embedding)                / (norm(sentence_embedding) * norm(embedding))            )
                results[word] = score

            return results