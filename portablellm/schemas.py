from typing import Any

from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = ""
    prompt_information: str = ""
    model: int = 1

class GenerateResponse(BaseModel):
    response: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    inference_time_seconds: float

class ParallelCapacityResponse(BaseModel):
    model_id: int | None
    model_file: str | None
    max_parallel_requests: int
    configured_max_parallel_requests: int
    active_requests: int
    queued_requests: int
    max_queued_requests: int
    gpu_layers_per_slot: list[int]
    safety_trimmed_slots: int
    machine: dict[str, Any]

class SimpleCapacityResponse(BaseModel):
    max_parallel_requests: int

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
