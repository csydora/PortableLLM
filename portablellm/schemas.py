from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    prompt_file: str = "default.txt"
    model: int = 1
    max_tokens: int | None = Field(default=None, ge=1, le=4096)

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
