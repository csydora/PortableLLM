from threading import RLock
from llama_cpp import Llama, llama_supports_gpu_offload
from portablellm.utils import application_path


class LLMService:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.lock = RLock()

    def load_model(self, model_number: int):
        with self.lock:
            if model_number in self.models:
                return
            model_path = application_path() / "portablellm" / "models" / self.config.get_model(model_number)
            if not model_path.is_file():
                raise FileNotFoundError(f"Model not found: {model_path}")
            if self.config.n_gpu_layers != 0 and not llama_supports_gpu_offload():
                raise RuntimeError("GPU offload unavailable. Install the CUDA wheel or set n_gpu_layers to 0.")
            self.close()
            print(f"Loading model {model_number}: {model_path}", flush=True)
            self.models[model_number] = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_size,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.config.get("n_batch", 256),
                n_threads=self.config.config.get("n_threads", 8),
                verbose=self.config.config.get("verbose", False),
            )

    def close(self):
        with self.lock:
            for model in self.models.values():
                model.close()
            self.models.clear()

    def load_prompt(self, filename: str) -> str:
        root = (application_path() / "portablellm" / "prompts").resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or path.suffix != ".txt":
            raise ValueError("prompt_file must be a .txt file inside the prompts directory")
        return path.read_text(encoding="utf-8")

    def load_prompt_template(self, filename: str, values: dict[str, str]) -> str:
        prompt = self.load_prompt(filename)
        for key, value in values.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        return prompt

    def generate(self, prompt: str, prompt_file: str = "default.txt", model: int = 1,
                 max_tokens: int | None = None) -> str:
        system_prompt = self.load_prompt_template(prompt_file, {"question": prompt})
        with self.lock:
            self.load_model(model)
            result = self.models[model].create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
            )
            choice = result["choices"][0]
            content = choice["message"].get("content") or ""
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            if choice.get("finish_reason") == "length":
                raise ValueError("Output limit reached. Increase max_tokens or shorten the prompt.")
            if not content.strip():
                raise ValueError("Model returned no final answer")
            return content.strip()

