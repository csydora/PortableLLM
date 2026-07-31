from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    prompt_file: str = "default.txt"
    model: int = 1

class GenerateResponse(BaseModel):
    response: str

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    score: float

class DictionarySimilarityRequest(BaseModel):
    sentence: str
    dictionary: dict[str, str]

class DictionarySimilarityResponse(BaseModel):
    scores: dict[str, float]