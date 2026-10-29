# PortableLLM

A small local REST API for GGUF language models and sentence similarity.
Built with FastAPI, llama-cpp-python, and Sentence Transformers.

## Layout

```text
PortableLLM/
  README.md
  config.json
  run.py
  start_server.bat
  stop_server.bat
  update_dependencies.bat
  installation/
    requirements.txt          # Common dependencies
    requirements-cuda.txt     # CUDA backend
    requirements-hip.txt      # AMD HIP backend
    requirements-vulkan.txt   # Vulkan backend
    requirements-cpu.txt      # CPU fallback
    install_gpu_backend.ps1   # Selects the llama.cpp backend
  portablellm/
    __init__.py          # Windows CUDA/HIP DLL registration
    main.py              # FastAPI application and model lifecycle
    schemas.py           # Request and response definitions
    server_control.py    # Background process management
    utils.py             # Application paths
    api/                 # HTTP routes
    services/            # Configuration, generation, and embeddings
    prompts/             # System prompt templates
    models/              # Local GGUF weights (Git-ignored)
    embedding_models/    # Local embedding weights (Git-ignored)
  logs/                  # Runtime logs (Git-ignored)
  .venv/                 # Python environment (Git-ignored)
```

## Run

The existing local installation is ready to use:

1. Run `start_server.bat`.
2. Open [API documentation](http://127.0.0.1:8000/docs).
3. Expand `POST /generate`, select **Try it out**, and submit:

```json
{
  "prompt": "Explain how a local language model works.",
  "model": 2,
  "max_tokens": 512
}
```

Use model 1 for Qwen3.5-4B or model 2 for Qwen3.5-9B (the default).
Only one generation model stays loaded; switching releases the previous model.
Run `stop_server.bat` to stop the service. Closing the browser does not stop it.

For foreground development, stop the background service first, then run:

```powershell
.\.venv\Scripts\python.exe run.py
```

Background management is also available through the project environment:
`python -m portablellm.server_control start` or `stop`.
The controller verifies process identity before stopping it.

## Install

Tested with Windows 11 and Python 3.12. Backend selection can be automatic or
set explicitly in `config.json`.

```powershell
.\update_dependencies.bat
```

The installer creates `.venv` if needed, installs common dependencies, then
selects the pinned llama.cpp backend wheel. The `installation/` folder contains
the common dependency list and the CUDA, HIP, Vulkan, and CPU backend lists.
Standalone EXE packaging is not configured.

### Models

Place these files in `portablellm/models/`:

| ID | Filename | Pinned download |
|---|---|---|
| 1 | Qwen3.5-4B-Q4_K_M.gguf | [4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf) |
| 2 | Qwen3.5-9B-Q4_K_M.gguf | [9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf) |

SHA256 checksums:

```text
4B  00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4
9B  03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8
```

For similarity endpoints, download the embedding model once:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir='portablellm/embedding_models/all-MiniLM-L6-v2', allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*'])"
```

The embedding model loads on CPU when first requested. Generation works without
it. All models are already present locally. `run.py` enables Hugging Face offline
mode for normal operation.

## Configuration

Edit `config.json`, then restart.

| Setting | Current value | Purpose |
|---|---|---|
| host / port | 127.0.0.1 / 8000 | Local API address |
| models | IDs 1 and 2 | GGUF filenames |
| preload_models | [3] | Load model 3 at startup |
| gpu_backend | auto | Select CUDA, HIP, Vulkan, or CPU during dependency setup |
| n_gpu_layers | -1 | Request all model layers on GPU, with automatic fallback |
| context_size | 8192 | Context window in tokens |
| n_batch | 256 | Prompt processing batch size |
| n_threads | 8 | CPU worker threads |
| temperature | 0.2 | Sampling temperature |
| max_tokens | 2048 | Default generation limit |
| verbose | false | Engine diagnostics |

The API default model is defined in `portablellm/schemas.py`.
Requests may override max_tokens. Context overflow and truncated output return
HTTP 400. Larger contexts require more memory.

## GPU acceleration

FastAPI -> llama-cpp-python -> llama.cpp -> selected GPU backend.

- `gpu_backend: auto` selects CUDA for NVIDIA, HIP for AMD, and CPU when neither
  vendor is detected. Set it to `cuda`, `hip`, `vulkan`, or `cpu` to choose
  manually, then rerun `update_dependencies.bat`.
- The AMD Radeon AI PRO R9700 is supported by AMD's Windows HIP SDK. Install a
  compatible AMD graphics driver and HIP runtime before setup. The HIP wheel
  also needs the Windows LLVM OpenMP runtime (`libomp140.x86_64.dll`); the app
  searches installed AMD SDK and Visual Studio runtime folders for it. Setup
  tries the HIP wheel first for AMD, then Vulkan, then CPU if installation or
  import fails. Vulkan is also available for NVIDIA and AMD GPUs.
- `n_gpu_layers: -1` asks llama.cpp to offload all model layers. Startup prints
  a concise offloaded layer count. Set `verbose: true` to show full llama.cpp
  diagnostics and build details. If full GPU loading fails, the service probes
  lower layer counts and keeps the largest configuration that loads; CPU is the
  final fallback.
- The CUDA wheel and NVIDIA runtime libraries are pinned in
  `installation/requirements-cuda.txt`. AMD uses the pinned HIP Radeon wheel,
  with installed HIP runtime DLL paths registered on Windows. Vulkan uses a
  pinned cross-vendor wheel.

Q4_K_M reduces weight size. GPU acceleration handles model layers while
tokenization and request handling still involve the CPU. Model size, context
length, free VRAM, and other GPU workloads affect how many layers fit.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | / | Service health |
| POST | /generate | Text generation with a prompt template |
| POST | /rdf | RDF generation prompt |
| POST | /sparql | SPARQL generation prompt |
| POST | /similarity | Cosine similarity between two texts |
| POST | /dictionary_similarity | Score definitions against a sentence |

See /docs for request schemas. Generation returns `{"response": "..."}`.
RDF/SPARQL endpoints do not validate or execute generated text. Each request is
independent; automatic conversation history and image input are not configured.

## Maintenance

Use English for project documentation, comments, examples, and maintenance
messages. Runtime inputs and model outputs may use any supported language.

Keep application code in portablellm/, configuration and launch entry points at
the root, and dependency installation files in installation/. Remove one-off
download/verification scripts and deployment reports after use. Keep model
provenance in this README and dependency versions in installation/requirements.txt.

Runtime logs are overwritten at startup. Model weights, logs, the virtual
environment, Python caches, and .server.json are excluded from Git.
Third-party packages and model assets are not project-maintained source.
