import json
from pathlib import Path
from portablellm.utils import application_path

class ConfigService:

    def __init__(self):

        config_path = application_path() / "config.json"

        print(f"Loading configuration from: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.models = self.config["models"]
        self.preload_models = self.config["preload_models"]

    def get_model(self, model_number: int) -> str:

        key = str(model_number)

        if key not in self.models:
            raise ValueError(f"Unknown model: {model_number}")

        return self.models[key]

    def get_model_download_url(self, model_number: int) -> str | None:
        return self.config.get("model_download_urls", {}).get(str(model_number))

    @property
    def context_size(self):
        return self.config["context_size"]

    @property
    def temperature(self):
        return self.config["temperature"]

    @property
    def max_tokens(self):
        return self.config["max_tokens"]

    @property
    def host(self):
        return self.config["host"]

    @property
    def port(self):
        return self.config["port"]

    @property
    def n_gpu_layers(self):
        return self.config.get("n_gpu_layers", 0)

    @property
    def max_parallel_requests(self):
        value = int(self.config.get("max_parallel_requests", 1))
        if value < 1:
            raise ValueError("max_parallel_requests must be at least 1")
        return value

    @property
    def max_queued_requests(self):
        value = int(self.config.get("max_queued_requests", 8))
        if value < 0:
            raise ValueError("max_queued_requests cannot be negative")
        return value

    @property
    def parallel_memory_reserve_gb(self):
        value = float(self.config.get("parallel_memory_reserve_gb", 3.0))
        if value < 0:
            raise ValueError("parallel_memory_reserve_gb cannot be negative")
        return value

    @property
    def embedding_model(self):
        return application_path() / self.config.get("embedding_model")
