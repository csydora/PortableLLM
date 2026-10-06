from fastapi import APIRouter
from portablellm.schemas import GenerateRequest, GenerateResponse
from portablellm.services.service_provider import (
    llm_service,
    embedding_service
)

from portablellm.schemas import (
    SimilarityRequest,
    SimilarityResponse,
    DictionarySimilarityRequest,
    DictionarySimilarityResponse
)

router = APIRouter()

@router.get("/")
def root():
    return {
        "status": "Running"
    }

@router.post("/generate")
def generate(request: GenerateRequest) -> GenerateResponse:
    return GenerateResponse(
        response=llm_service.generate(
            request.prompt,
            request.prompt_information,
            "chat.txt",
            request.model,
        )
    )

@router.post("/sparql")
def sparql(request: GenerateRequest) -> GenerateResponse:
    return GenerateResponse(
        response=llm_service.generate(
            request.prompt,
            request.prompt_information,
            "sparql.txt",
            request.model,
        )
    )

@router.post("/rdf")
def rdf(request: GenerateRequest) -> GenerateResponse:
    return GenerateResponse(
        response=llm_service.generate(
            request.prompt,
            request.prompt_information,
            "rdf.txt",
            request.model,
        )
    )

@router.post("/similarity")
def similarity( request: SimilarityRequest ) -> SimilarityResponse:
    return SimilarityResponse(
        score=embedding_service.similarity(
            request.text1,
            request.text2
        )
    )

@router.post("/dictionary_similarity")
def dictionary_similarity( request: DictionarySimilarityRequest ) -> DictionarySimilarityResponse:
    return DictionarySimilarityResponse(
        scores=embedding_service.score_dictionary(
            request.sentence,
            request.dictionary
        )
    )
