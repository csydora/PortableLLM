from ctypes import c_void_p
from threading import RLock
import re
import sys
from llama_cpp import (
    Llama,
    llama_log_callback,
    llama_log_set,
    llama_print_system_info,
    llama_supports_gpu_offload,
)
from portablellm.utils import application_path
import time 

class LLMService:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.lock = RLock()
        self.n_gpu_layers = 0
        self._backend_messages = []
        self._verbose = bool(self.config.config.get("verbose", False))
        self._llama_log_callback = llama_log_callback(self._receive_llama_log)
        llama_log_set(self._llama_log_callback, c_void_p(0))

    def _receive_llama_log(self, level, text, user_data):
        message = (text or b"").decode("utf-8", errors="replace")
        is_backend_message = any(
            term in message.lower()
            for term in ("offload", "gpu", "cuda", "vulkan", "rocm", "hip", "device", "error")
        )
        if is_backend_message:
            self._backend_messages.append(message)
        if self._verbose or level >= 3:
            print(message, end="", file=sys.stderr, flush=True)

    def _create_model(self, model_path, n_gpu_layers):
        self._backend_messages.clear()
        return Llama(
            model_path=str(model_path),
            n_ctx=self.config.context_size,
            n_gpu_layers=n_gpu_layers,
            n_batch=self.config.config.get("n_batch", 256),
            n_threads=self.config.config.get("n_threads", 8),
            verbose=self._verbose,
        )

    def _model_layer_count(self, model_path: str) -> int:
        """Read GGUF layer metadata without loading model weights."""
        probe = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_batch=64,
            n_threads=1,
            n_gpu_layers=0,
            vocab_only=True,
            verbose=False,
        )
        try:
            metadata = probe.metadata
            architecture = metadata.get("general.architecture")
            layer_count = metadata.get(f"{architecture}.block_count") if architecture else None
            if layer_count is None:
                raise ValueError("The model does not report its transformer layer count")
            return int(layer_count)
        finally:
            probe.close()

    def _load_with_max_gpu_layers(self, model_path, upper_bound: int):
        """Find the largest GPU layer count that can initialize successfully."""
        best_model = self._create_model(model_path, 0)
        best_layers = 0
        best_messages = list(self._backend_messages)
        low = 1
        high = upper_bound

        while low <= high:
            candidate_layers = (low + high) // 2
            try:
                candidate = self._create_model(model_path, candidate_layers)
            except Exception as exc:
                print(
                    f"GPU offload with {candidate_layers} layers failed: {exc}",
                    flush=True,
                )
                high = candidate_layers - 1
            else:
                best_model.close()
                best_model = candidate
                best_layers = candidate_layers
                best_messages = list(self._backend_messages)
                low = candidate_layers + 1

        self._backend_messages = best_messages
        return best_model, best_layers

    def _report_gpu_load(self, requested_layers: int):
        if self.n_gpu_layers == 0:
            if requested_layers != 0:
                print("GPU layer offload unavailable; the model is running on CPU.", flush=True)
            return

        match = re.search(
            r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers",
            "".join(self._backend_messages),
            re.IGNORECASE,
        )
        if match:
            print(
                f"GPU layer offload confirmed: {match.group(1)}/{match.group(2)} layers.",
                flush=True,
            )
        else:
            print(
                "The model loaded, but llama.cpp did not report GPU layer offload. "
                "Check the backend wheel and GPU driver.",
                flush=True,
            )

    def load_model(self, model_number: int):
        if model_number in self.models:
            return
        with self.lock:
            if model_number in self.models:
                return
            model_path = application_path() / "portablellm" / "models" / self.config.get_model(model_number)
            if not model_path.is_file():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.close()
            print(f"Loading model {model_number}: {model_path}", flush=True)

            requested_layers = int(self.config.n_gpu_layers)
            gpu_backend_available = requested_layers != 0 and llama_supports_gpu_offload()
            if not gpu_backend_available:
                self.n_gpu_layers = 0
                if requested_layers != 0:
                    print("This llama.cpp build has no GPU backend; loading on CPU.", flush=True)
                model = self._create_model(model_path, 0)
            else:
                self._backend_messages.clear()
                try:
                    model = self._create_model(model_path, requested_layers)
                    self.n_gpu_layers = requested_layers
                except Exception as gpu_error:
                    print(
                        f"GPU model load failed ({gpu_error}); retrying with the largest layer count that fits.",
                        flush=True,
                    )
                    try:
                        layer_count = self._model_layer_count(str(model_path))
                        upper_bound = (
                            layer_count + 1
                            if requested_layers < 0
                            else min(requested_layers - 1, layer_count + 1)
                        )
                        model, self.n_gpu_layers = self._load_with_max_gpu_layers(
                            model_path,
                            max(0, upper_bound),
                        )
                        print(
                            f"Loaded with {self.n_gpu_layers}/{layer_count} layers assigned to GPU; "
                            "remaining layers use CPU.",
                            flush=True,
                        )
                    except Exception as fallback_error:
                        print(
                            f"GPU fallback failed ({fallback_error}); retrying on CPU.",
                            flush=True,
                        )
                        model = self._create_model(model_path, 0)
                        self.n_gpu_layers = 0

            self.models[model_number] = model
            if self._verbose:
                system_info = llama_print_system_info().decode("utf-8", errors="replace").strip()
                print(f"llama.cpp build: {system_info}", flush=True)
            self._report_gpu_load(requested_layers)

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

    def generate(self, prompt: str, prompt_information: str, prompt_file: str = "chat.txt", model: int = 1) -> str:
        if not prompt_information or not prompt_information.strip():
            system_prompt = self.load_prompt_template(prompt_file, {"question": prompt})
        else:
            system_prompt = prompt_information
        
        print(
            f"System Prompt: {system_prompt[:20] + ("..." if len(system_prompt) > 20 else "")}\n"
            f"Content: {prompt[:20] + ("..." if len(prompt) > 20 else "")}"
        )
        start = time.perf_counter()          # start the timer

        with self.lock:
            self.load_model(model)
            result = self.models[model].create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            choice = result["choices"][0]
            content = choice["message"].get("content") or ""
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            if choice.get("finish_reason") == "length":
                raise ValueError("Output limit reached. Increase max_tokens or shorten the prompt.")
            if not content.strip():
                raise ValueError("Model returned no final answer")

            returnResult = content.strip()
            print(f"Result: {returnResult[:20] + ("..." if len(returnResult) > 20 else "")}")
            elapsed = time.perf_counter() - start  # stop the timer
            print(f"Inference time: {elapsed:.2f} seconds")
            return returnResult

