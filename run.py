from portablellm.main import app
from portablellm.services.service_provider import config
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "portablellm.main:app",
        host=config.host,
        port=config.port
    )