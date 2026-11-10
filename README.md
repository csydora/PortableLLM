# PortableLLM

PortableLLM runs local GGUF language models and provides them through a local REST API. It also includes text similarity endpoints.

## Quick start

These steps are for Windows. You need Python 3.12 and an internet connection for the initial dependency and model downloads. A compatible GPU is optional; CPU mode is available.

### 1. Install the dependencies

Install Python 3.12 for Windows, including the `py` launcher, if it is not already installed. In the PortableLLM folder, double-click **`update_dependencies.bat`** and let it finish. This creates the project’s Python environment, installs the app’s dependencies, and selects a llama.cpp backend based on your hardware. It can fall back to CPU.

Run this setup again if you need to reinstall the dependencies or change the GPU backend.

### 2. Download a language model

Generation needs one GGUF model file. Model files are not included in a fresh checkout. The default model is ID **3**: download [Qwen2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf) and save it here, using this exact filename:

```text
portablellm/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

The included embedding model is ready to use, so similarity requests do not need another model download.

### 3. Start PortableLLM

Double-click **`start_server.bat`** and leave its window open while you use the service. Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API page.

To try text generation, expand **POST /generate**, click **Try it out**, and submit:

```json
{
  "prompt": "Explain how a local language model works."
}
```

Model 3 is used by default. The first generation request loads the model and may take a little while. The `/docs` page is an API tool, not a chat application.

### 4. Stop PortableLLM

Press **Ctrl+C** in the server window, or run **`stop_server.bat`**. Closing the browser tab does not stop the server.

## Additional information

### Available language models

Download the model file or files into `portablellm/models/`, preserving the paths shown below. The request's `model` value selects the model. Model 3 is the default.

| ID | Model | Filename | Download |
|---:|---|---|---|
| 1 | Qwen2.5 0.5B Instruct | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | [Download](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf) |
| 2 | Qwen2.5 1.5B Instruct | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | [Download](https://huggingface.co/second-state/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf) |
| 3 | Qwen2.5 3B Instruct | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | [Download](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf) |
| 4 | Qwen3 8B | `Qwen_Qwen3-8B-Q4_K_M.gguf` | [Download](https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf) |
| 5 | Qwen3.5 4B | `Qwen3.5-4B-Q4_K_M.gguf` | [Download](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf) |
| 6 | Qwen3.5 9B | `Qwen3.5-9B-Q4_K_M.gguf` | [Download](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf) |
| 7 | Qwen3.6 35B-A3B | `Qwen3.6-35B-A3B-Q4_K_M.gguf` (about 20.5 GB) | [Download](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf) |
| 8 | Qwen3.8-Flash-Next (UD-Q4_K_XL) | Four GGUF shards under `UD-Q4_K_XL/` (about 111 GB total; one worker, tight on 128 GB system RAM) | [Model files](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q4_K_XL) |

If requested model files are missing, the API returns an error with a PowerShell command to download them. Change the available model filenames and download URLs in `config.json`; change the default model ID in `portablellm/schemas.py`. Model downloads are not automatic: run the command shown by the API.

### API endpoints

The API listens on `127.0.0.1:8000` by default. Open `/docs` while the server is running to see request fields and try endpoints.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Check whether the service is running |
| GET | `/capacity` | View current generation capacity and queue status |
| GET | `/capacity/simple` | Read the effective parallel request limit |
| POST | `/generate` | Generate a response |
| POST | `/rdf` | Generate RDF-oriented text |
| POST | `/sparql` | Generate SPARQL-oriented text |
| POST | `/similarity` | Compare two texts |
| POST | `/dictionary_similarity` | Score definitions against a sentence |

### GPU and configuration

`update_dependencies.bat` reads `gpu_backend` from `config.json`, which defaults to `auto`. Automatic selection looks for NVIDIA or AMD hardware; backend setup can fall back to Vulkan or CPU when appropriate. Set `gpu_backend` to `cuda`, `hip`, `vulkan`, or `cpu` to choose a backend manually, then run `update_dependencies.bat` again. GPU use depends on compatible drivers and runtimes; the service can also run on CPU.

Edit `config.json` to change the local address, model files, context size, generation settings, or request limits. Restart the server after changing configuration. Model entries may name one GGUF file or a list of shards. Larger models and context sizes need more memory. The service adjusts the number of simultaneous generation contexts to available resources.

PortableLLM is built with FastAPI, llama-cpp-python, and Sentence Transformers. The `installation/` folder contains the pinned Python dependencies and backend setup files; `logs/` contains runtime logs.
