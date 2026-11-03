from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from portablellm.api.routes import router
from portablellm.services.service_provider import config, llm_service
from portablellm.services.llm_service import LLMCapacityError

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading models...")

    for model in config.preload_models:
        llm_service.load_model(model)

    print("Models ready.")

    try:
        yield
    finally:
        llm_service.close()
        print("Shutting down...")


app = FastAPI(
    title="PortableLLM",
    version="0.1",
    lifespan=lifespan,
)

app.include_router(router)


@app.exception_handler(ValueError)
async def bad_request(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def missing_file(request, exc):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(LLMCapacityError)
async def capacity_full(request, exc):
    return JSONResponse(status_code=429, content={"detail": str(exc)})
