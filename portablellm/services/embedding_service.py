from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from pathlib import Path
from portablellm.utils import application_path


class EmbeddingService:

    def __init__(self):
        model_path = ( application_path() / "portablellm" / "models" / "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(str(model_path))
        self.cache = {}

    def similarity(self, text1: str, text2: str) -> float:
        e1 = self.model.encode(text1)
        e2 = self.model.encode(text2)

        return float(dot(e1, e2) / (norm(e1) * norm(e2)))

    def score_dictionary( self, sentence: str, dictionary: dict[str, str] ) -> dict[str, float]:
        sentence_embedding = self.model.encode(sentence)
        results = {}
        for word, definition in dictionary.items():
            if word not in self.cache:
                self.cache[word] = self.model.encode(definition)
                
            embedding = self.cache[word]
            score = float(
                dot(sentence_embedding, embedding)
                / (norm(sentence_embedding) * norm(embedding))
            )
            results[word] = score

        return results