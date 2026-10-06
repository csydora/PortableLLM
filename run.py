import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import uvicorn
from portablellm.main import app
from portablellm.services.service_provider import config

if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)

