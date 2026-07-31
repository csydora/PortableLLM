from contextlib import asynccontextmanager

from fastapi import FastAPI

from portablellm.api.routes import router
from portablellm.services.service_provider import config, llm_service

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading models...")

    for model in config.preload_models:
        llm_service.load_model(model)

    print("Models ready.")

    yield

    print("Shutting down...")


app = FastAPI(
    title="PortableLLM",
    version="0.1",
    lifespan=lifespan,
)

app.include_router(router)