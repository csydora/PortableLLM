import json
import gc
import os
import re
import shutil
import subprocess
import sys
import time
from ctypes import c_void_p
from datetime import datetime, timezone
from threading import Condition

import psutil
from llama_cpp import (
    Llama,
    llama_log_callback,
    llama_log_set,
    llama_print_system_info,
    llama_supports_gpu_offload,
)
from portablellm.utils import application_path


class LLMCapacityError(Exception):
    """Raised when the configured waiting queue is full."""


class LLMService:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self._worker_models = []
        self._idle_models = []
        self._worker_gpu_layers = []
        self._safety_trimmed_slots = 0
        self._current_model = None
        self._active_requests = 0
        self._queued_requests = 0
        self._switching_model = False
        self._closing = False
        self._state = Condition()
        self.n_gpu_layers = 0
        self._backend_messages = []
        self._verbose = bool(self.config.config.get("verbose", False))
        self._llama_log_callback = llama_log_callback(self._receive_llama_log)
        llama_log_set(self._llama_log_callback, c_void_p(0))
        self._machine_info = self._collect_machine_info()

    def _receive_llama_log(self, level, text, user_data):
        message = (text or b"").decode("utf-8", errors="replace")
        is_backend_message = any(
            term in message.lower()
            for term in ("offload", "gpu", "cuda", "vulkan", "rocm", "hip", "device", "error")
        )
        if is_backend_message:
            self._backend_messages.append(message)
        if self._verbose:
            print(message, end="", file=sys.stderr, flush=True)

    @staticmethod
    def _nvidia_memory_info():
        executable = shutil.which("nvidia-smi")
        if not executable:
            return None
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            first_gpu = result.stdout.strip().splitlines()[0].split(",")
            return {
                "name": first_gpu[0].strip(),
                "free_bytes": int(first_gpu[1].strip()) * 1024 * 1024,
                "total_bytes": int(first_gpu[2].strip()) * 1024 * 1024,
            }
        except (OSError, IndexError, ValueError, subprocess.SubprocessError):
            return None

    def _collect_machine_info(self):
        memory = psutil.virtual_memory()
        gpu_memory = self._nvidia_memory_info()
        return {
            "logical_cpu_count": psutil.cpu_count(logical=True) or os.cpu_count() or 1,
            "memory_total_bytes": int(memory.total),
            "memory_available_bytes": int(memory.available),
            "gpu_backend_compiled": bool(llama_supports_gpu_offload()),
            "gpu_memory": gpu_memory,
        }

    def _thread_budget(self, setting_name: str, fallback: int) -> int:
        configured = int(self.config.config.get(setting_name, fallback))
        if configured > 0:
            return configured
        return max(1, int(self._machine_info["logical_cpu_count"]))

    def _create_model(self, model_path, n_gpu_layers, worker_count: int):
        self._backend_messages.clear()
        configured_slots = max(1, min(self.config.max_parallel_requests, worker_count))
        threads = max(1, self._thread_budget("n_threads", 8) // configured_slots)
        batch_threads = max(
            1,
            self._thread_budget("n_threads_batch", self._thread_budget("n_threads", 8))
            // configured_slots,
        )
        return Llama(
            model_path=str(model_path),
            n_ctx=self.config.context_size,
            n_gpu_layers=n_gpu_layers,
            n_batch=self.config.config.get("n_batch", 256),
            n_threads=threads,
            n_threads_batch=batch_threads,
            verbose=self._verbose,
        )

    def _model_metadata(self, model_path: str):
        """Read GGUF metadata without loading model weights."""
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
            return dict(probe.metadata)
        finally:
            probe.close()

    @staticmethod
    def _metadata_int(metadata, key, default=None):
        try:
            return int(metadata[key])
        except (KeyError, TypeError, ValueError):
            return default

    def _estimate_kv_cache_bytes(self, metadata, model_size_bytes: int) -> int:
        architecture = metadata.get("general.architecture")
        if architecture:
            layer_count = self._metadata_int(metadata, f"{architecture}.block_count")
            embedding_length = self._metadata_int(metadata, f"{architecture}.embedding_length")
            attention_heads = self._metadata_int(metadata, f"{architecture}.attention.head_count")
            kv_heads = self._metadata_int(
                metadata,
                f"{architecture}.attention.head_count_kv",
                attention_heads,
            )
            head_length = self._metadata_int(metadata, f"{architecture}.attention.key_length")
            if head_length is None and embedding_length and attention_heads:
                head_length = embedding_length // attention_heads
            if layer_count and kv_heads and head_length:
                # K and V caches, assuming FP16. The estimate is per context slot.
                return (
                    layer_count
                    * kv_heads
                    * head_length
                    * int(self.config.context_size)
                    * 4
                )
        return max(256 * 1024**2, min(2 * 1024**3, model_size_bytes // 4))

    def _estimate_probe_limit(self, metadata, model_size_bytes: int) -> int:
        """Use host memory as a conservative preflight before loading contexts."""
        memory = psutil.virtual_memory()
        available = int(memory.available)
        reserve = max(1024**3, int(memory.total * 0.15))
        kv_cache_bytes = self._estimate_kv_cache_bytes(metadata, model_size_bytes)
        requested_layers = int(self.config.n_gpu_layers)
        has_gpu = requested_layers != 0 and bool(self._machine_info["gpu_backend_compiled"])

        # File-backed CPU weights are often shared by Windows' cache. GPU backends
        # may also allocate model buffers in system memory, especially on UMA GPUs.
        weight_allowance = min(model_size_bytes, 1024**3) if has_gpu else min(model_size_bytes // 4, 512 * 1024**2)
        per_extra_slot = kv_cache_bytes + 384 * 1024**2 + weight_allowance
        base_allowance = min(model_size_bytes, max(0, available // 3))
        slot_budget = max(0, available - reserve - base_allowance)
        memory_slots = 1 + slot_budget // max(1, per_extra_slot)
        return max(1, min(self.config.max_parallel_requests, int(memory_slots)))

    def _load_with_max_gpu_layers(self, model_path, upper_bound: int, worker_count: int):
        """Find the largest GPU layer count that initializes with current allocations."""
        best_model = self._create_model(model_path, 0, worker_count)
        best_layers = 0
        low = 1
        high = upper_bound

        while low <= high:
            candidate_layers = (low + high) // 2
            try:
                candidate = self._create_model(model_path, candidate_layers, worker_count)
            except Exception:
                high = candidate_layers - 1
            else:
                best_model.close()
                best_model = candidate
                best_layers = candidate_layers
                low = candidate_layers + 1

        return best_model, best_layers

    def _load_worker_model(self, model_path, metadata, worker_count: int):
        requested_layers = int(self.config.n_gpu_layers)
        layer_count = self._metadata_int(
            metadata,
            f"{metadata.get('general.architecture', '')}.block_count",
        )
        if requested_layers == 0 or not llama_supports_gpu_offload():
            return self._create_model(model_path, 0, worker_count), 0

        try:
            model = self._create_model(model_path, requested_layers, worker_count)
            reported_layers = self._reported_offloaded_layers()
            if reported_layers is not None:
                actual_layers = reported_layers
            elif requested_layers > 0 and layer_count:
                actual_layers = min(requested_layers, layer_count + 1)
            elif layer_count:
                actual_layers = layer_count + 1
            else:
                actual_layers = max(0, requested_layers)
            return model, actual_layers
        except Exception:
            if layer_count is None:
                return self._create_model(model_path, 0, worker_count), 0
            upper_bound = (
                layer_count + 1
                if requested_layers < 0
                else min(requested_layers - 1, layer_count + 1)
            )
            return self._load_with_max_gpu_layers(model_path, max(0, upper_bound), worker_count)

    def _reported_offloaded_layers(self):
        match = re.search(
            r"offloaded\s+(\d+)\s*/\s*\d+\s+layers",
            "".join(self._backend_messages),
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else None

    def _rebalance_worker_threads(self, worker_models):
        count = max(1, len(worker_models))
        thread_budget = self._thread_budget("n_threads", 8)
        batch_budget = self._thread_budget("n_threads_batch", self._thread_budget("n_threads", 8))
        threads_per_worker = max(1, thread_budget // count)
        batch_threads_per_worker = max(1, batch_budget // count)
        for model in worker_models:
            model._ctx.set_n_threads(threads_per_worker, batch_threads_per_worker)

    def _trim_unsafe_worker_slots(self, worker_models, worker_gpu_layers):
        trimmed = 0
        reserve_bytes = int(self.config.parallel_memory_reserve_gb * 1024**3)
        gpu_can_offload = (
            int(self.config.n_gpu_layers) != 0
            and bool(self._machine_info["gpu_backend_compiled"])
        )

        while len(worker_models) > 1:
            available_bytes = int(psutil.virtual_memory().available)
            memory_is_low = available_bytes < reserve_bytes
            gpu_is_degraded = False
            if gpu_can_offload and len(worker_gpu_layers) > 1:
                best_prior_gpu_layers = max(worker_gpu_layers[:-1], default=0)
                gpu_is_degraded = (
                    best_prior_gpu_layers > 0
                    and worker_gpu_layers[-1] < best_prior_gpu_layers
                )

            if not memory_is_low and not gpu_is_degraded:
                break

            worker = worker_models.pop()
            worker_gpu_layers.pop()
            self._close_models([worker])
            gc.collect()
            trimmed += 1

        return trimmed

    def _build_worker_pool(self, model_number: int):
        model_path = application_path() / "portablellm" / "models" / self.config.get_model(model_number)
        if not model_path.is_file():
            download_url = self.config.get_model_download_url(model_number)
            message = f"Model file is missing: {model_path}"
            if download_url:
                def quote_powershell(value):
                    return "'" + str(value).replace("'", "''") + "'"

                command = (
                    "New-Item -ItemType Directory -Force -Path "
                    f"{quote_powershell(model_path.parent)} | Out-Null; "
                    f"Invoke-WebRequest -Uri {quote_powershell(download_url)} "
                    f"-OutFile {quote_powershell(model_path)}"
                )
                message += f"\nDownload it with PowerShell:\n{command}"
            else:
                message += (
                    f"\nNo download URL is configured for model {model_number}; "
                    "set model_download_urls in config.json."
                )
            print(message, file=sys.stderr, flush=True)
            raise FileNotFoundError(message)

        model_size_bytes = model_path.stat().st_size
        try:
            metadata = self._model_metadata(str(model_path))
        except Exception:
            metadata = {}

        probe_limit = self._estimate_probe_limit(metadata, model_size_bytes)
        print(
            f"Loading model {model_number}; checking up to {probe_limit} parallel request slots.",
            flush=True,
        )

        worker_models = []
        worker_gpu_layers = []
        for _ in range(probe_limit):
            try:
                worker, gpu_layers = self._load_worker_model(
                    model_path,
                    metadata,
                    len(worker_models) + 1,
                )
            except Exception as exc:
                if not worker_models:
                    raise
                print(
                    "Additional model context did not fit; keeping the contexts already loaded "
                    f"({type(exc).__name__}).",
                    flush=True,
                )
                break
            worker_models.append(worker)
            worker_gpu_layers.append(max(0, int(gpu_layers)))

        if not worker_models:
            raise RuntimeError(f"Could not load model {model_number}")

        safety_trimmed_slots = self._trim_unsafe_worker_slots(worker_models, worker_gpu_layers)
        self._rebalance_worker_threads(worker_models)
        if safety_trimmed_slots:
            print(
                f"Safety check removed {safety_trimmed_slots} request slot(s); "
                f"keeping {len(worker_models)}.",
                flush=True,
            )
        if self._verbose:
            system_info = llama_print_system_info().decode("utf-8", errors="replace").strip()
            print(f"llama.cpp build: {system_info}", flush=True)
        return (
            model_path,
            worker_models,
            worker_gpu_layers,
            model_size_bytes,
            probe_limit,
            safety_trimmed_slots,
        )

    @staticmethod
    def _close_models(models):
        for model in models:
            try:
                model.close()
            except Exception as exc:
                print(f"Could not close a model context cleanly: {exc}", flush=True)

    def _install_worker_pool(self, model_number: int, old_models):
        self._close_models(old_models)
        try:
            (
                model_path,
                worker_models,
                gpu_layers,
                model_size,
                probe_limit,
                safety_trimmed_slots,
            ) = self._build_worker_pool(model_number)
        except Exception:
            with self._state:
                self._switching_model = False
                self._state.notify_all()
            raise

        machine_info = self._collect_machine_info()
        machine_info["model_size_bytes"] = model_size
        machine_info["memory_probe_limit"] = probe_limit
        machine_info["context_size"] = int(self.config.context_size)
        machine_info["parallel_memory_reserve_gb"] = self.config.parallel_memory_reserve_gb
        with self._state:
            self._worker_models = worker_models
            self._idle_models = list(worker_models)
            self._worker_gpu_layers = gpu_layers
            self._safety_trimmed_slots = safety_trimmed_slots
            self.models = {model_number: worker_models[0]}
            self._current_model = model_number
            self.n_gpu_layers = max(gpu_layers, default=0)
            self._machine_info = machine_info
            self._switching_model = False
            self._state.notify_all()

        effective = len(worker_models)
        gpu_summary = ", ".join(str(count) for count in gpu_layers)
        print(
            f"Parallel capacity: {effective}/{self.config.max_parallel_requests} request slots "
            f"for model {model_number}; GPU layers per slot: {gpu_summary}.",
            flush=True,
        )
        self._persist_capacity()

    def load_model(self, model_number: int):
        with self._state:
            while self._switching_model:
                self._state.wait()
            if self._current_model == model_number and self._worker_models:
                return
            self._switching_model = True
            while self._active_requests > 0:
                self._state.wait()
            old_models = self._worker_models
            self._worker_models = []
            self._idle_models = []
            self._worker_gpu_layers = []
            self._safety_trimmed_slots = 0
            self.models = {}
            self._current_model = None
        self._install_worker_pool(model_number, old_models)

    def _join_queue_locked(self, already_queued: bool) -> bool:
        if already_queued:
            return True
        if self._queued_requests >= self.config.max_queued_requests:
            raise LLMCapacityError(
                "The server is at its parallel request and queue limits. Retry after a request finishes."
            )
        self._queued_requests += 1
        return True

    def _acquire_model_slot(self, model_number: int):
        queued = False
        while True:
            old_models = None
            with self._state:
                if self._closing:
                    if queued:
                        self._queued_requests -= 1
                    raise LLMCapacityError("The server is shutting down.")
                if self._switching_model:
                    queued = self._join_queue_locked(queued)
                    self._state.wait()
                    continue

                if self._current_model != model_number or not self._worker_models:
                    if self._active_requests > 0:
                        queued = self._join_queue_locked(queued)
                    self._switching_model = True
                    while self._active_requests > 0:
                        self._state.wait()
                    old_models = self._worker_models
                    self._worker_models = []
                    self._idle_models = []
                    self._worker_gpu_layers = []
                    self._safety_trimmed_slots = 0
                    self.models = {}
                    self._current_model = None
                elif self._active_requests < len(self._worker_models):
                    if queued:
                        self._queued_requests -= 1
                        queued = False
                    self._active_requests += 1
                    model = self._idle_models.pop()
                    return model
                else:
                    queued = self._join_queue_locked(queued)
                    self._state.wait()
                    continue

            # A model change waits for all active requests, then replaces the pool.
            try:
                self._install_worker_pool(model_number, old_models)
            except Exception:
                if queued:
                    with self._state:
                        self._queued_requests -= 1
                        self._state.notify_all()
                raise

    def _release_model_slot(self, model):
        with self._state:
            self._idle_models.append(model)
            self._active_requests -= 1
            self._state.notify_all()

    def generate(
        self,
        prompt: str,
        prompt_information: str,
        prompt_file: str = "chat.txt",
        model: int = 1,
    ) -> dict[str, str | int | float]:
        if not prompt_information or not prompt_information.strip():
            system_prompt = self.load_prompt_template(prompt_file, {"question": prompt})
        else:
            system_prompt = prompt_information

        print(
            f"System Prompt: {system_prompt[:20] + ('...' if len(system_prompt) > 20 else '')}\n"
            f"Content: {prompt[:20] + ('...' if len(prompt) > 20 else '')}"
        )
        worker_model = self._acquire_model_slot(model)
        try:
            inference_start = time.perf_counter()
            result = worker_model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            inference_time = time.perf_counter() - inference_start
            choice = result["choices"][0]
            usage = result["usage"]
            content = choice["message"].get("content") or ""
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            if choice.get("finish_reason") == "length":
                raise ValueError("Output limit reached. Increase max_tokens or shorten the prompt.")
            if not content.strip():
                raise ValueError("Model returned no final answer")

            result_text = content.strip()
            print(f"Result: {result_text[:20] + ('...' if len(result_text) > 20 else '')}")
            print(f"Inference time: {inference_time:.2f} seconds")
            return {
                "response": result_text,
                "prompt_tokens": usage["prompt_tokens"],
                "response_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "inference_time_seconds": inference_time,
            }
        finally:
            self._release_model_slot(worker_model)

    def get_capacity(self):
        with self._state:
            model_number = self._current_model
            model_file = None
            if model_number is not None:
                model_file = self.config.get_model(model_number)
            return {
                "model_id": model_number,
                "model_file": model_file,
                "max_parallel_requests": len(self._worker_models),
                "configured_max_parallel_requests": self.config.max_parallel_requests,
                "active_requests": self._active_requests,
                "queued_requests": self._queued_requests,
                "max_queued_requests": self.config.max_queued_requests,
                "gpu_layers_per_slot": list(self._worker_gpu_layers),
                "safety_trimmed_slots": self._safety_trimmed_slots,
                "machine": dict(self._machine_info),
            }

    def _persist_capacity(self):
        status_path = application_path() / "logs" / "parallel_capacity.json"
        snapshot = self.get_capacity()
        snapshot["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
        snapshot["active_requests"] = 0
        snapshot["queued_requests"] = 0
        try:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = status_path.with_suffix(".tmp")
            temporary_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            temporary_path.replace(status_path)
        except OSError as exc:
            print(f"Could not save parallel capacity status: {exc}", flush=True)

    def close(self):
        with self._state:
            while self._switching_model:
                self._state.wait()
            self._closing = True
            self._switching_model = True
            self._state.notify_all()
            while self._active_requests > 0:
                self._state.wait()
            models = self._worker_models
            self._worker_models = []
            self._idle_models = []
            self._worker_gpu_layers = []
            self._safety_trimmed_slots = 0
            self.models = {}
            self._current_model = None
            self.n_gpu_layers = 0
        self._close_models(models)
        with self._state:
            self._switching_model = False
            self._state.notify_all()

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
