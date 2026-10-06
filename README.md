# PortableLLM

> 本机已配置版本：请优先阅读 [LOCAL_SETUP.md](LOCAL_SETUP.md)。
> 双击 `start_server.bat` 启动，访问 http://127.0.0.1:8000/docs；双击 `stop_server.bat` 停止。
> 下方保留上游说明，其中原始模型示例和 EXE 构建步骤不代表本次部署配置。

PortableLLM is a lightweight local REST API for running Large Language Models (LLMs) using `llama-cpp-python`. It provides a simple HTTP interface for text generation and semantic similarity without requiring Ollama or any external services.

## Features

- Local LLM inference using GGUF models
- Multiple model support
- Prompt templates
- Sentence similarity using Sentence Transformers
- Dictionary similarity scoring
- FastAPI REST API
- Swagger documentation (`/docs`)
- Standalone executable build

---

## Requirements

- Python 3.12+
- Git
- Windows (currently tested)
- At least one GGUF model
- An embedding model (Sentence Transformers)

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd PortableLLM
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Models

### LLM Models

Place GGUF models inside:

```
portablellm/models/
```

Example:

```
portablellm/models/
    Qwen_Qwen3-8B-Q4_K_M.gguf
    qwen2.5-3b-instruct-q4_k_m.gguf
```

Configure them in `config.json`.

---

### Embedding Model

Download the Sentence Transformer model:

```
sentence-transformers/all-MiniLM-L6-v2
```

Place it in:

```
portablellm/embedding_models/
    models--sentence-transformers--all-MiniLM-L6-v2/
```

---

## Configuration

All settings are stored in:

```
config.json
```

Example:

```json
{
    "host": "127.0.0.1",
    "port": 8000,

    "models": {
        "1": "Qwen_Qwen3-8B-Q4_K_M.gguf",
        "2": "qwen2.5-3b-instruct-q4_k_m.gguf"
    },

    "preload_models": [1],

    "context_size": 4096,
    "temperature": 0.2,
    "max_tokens": 1024
}
```

---

## Running

Start the server:

```powershell
start_server.bat
```

or

```powershell
uvicorn portablellm.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

## Building

Build the standalone executable:

```powershell
build.bat
```

The output is placed in:

```
build/
```

Run the server with:

```
PortableLLM.exe
```

---

## API Endpoints

### Generate Text

```
POST /generate
```

```json
{
    "prompt": "Tell me something interesting.",
    "prompt_file": "default.txt",
    "model": 1
}
```

Returns:

```json
{
    "response": "Octopuses have three hearts..."
}
```

---

### RDF Generation

```
POST /rdf
```

```json
{
    "prompt": "John is 25 years old.",
    "model": 1
}
```

Returns:

```json
{
    "response": "@prefix ex: <https://example.com/> .\n@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n\nex:John foaf:age 25 ."
}
```

---

### SPARQL Generation

```
POST /sparql
```

```json
{
    "prompt": "Find all walls higher than 3 metres.",
    "model": 1
}
```

Returns:

```json
{
    "response": "SELECT ?wall WHERE {\n  ?wall rdf:type ex:Wall .\n  ?wall ex:height ?height .\n  FILTER(?height > 3)\n}"
}
```

---

### Sentence Similarity

```
POST /similarity
```

```json
{
    "text1": "The wall is concrete.",
    "text2": "The wall is made of concrete."
}
```

Returns:

```json
{
    "score": 0.94
}
```

---

### Dictionary Similarity

```
POST /dictionary_similarity
```

```json
{
    "sentence": "The wall is reinforced concrete.",
    "dictionary": {
        "Concrete": "A composite construction material.",
        "Steel": "An alloy primarily consisting of iron."
    }
}
```

Returns:

```json
{
    "scores": {
        "Concrete": 0.96,
        "Steel": 0.43
    }
}
```

---

## Project Structure

```
PortableLLM/
│
├── config.json
├── build.bat
├── start_server.bat
├── run.py
│
└── portablellm/
    ├── api/
    ├── models/
    ├── embedding_models/
    ├── prompts/
    ├── services/
    ├── schemas.py
    └── main.py
```

---

## Notes

- GGUF models are not included in the repository.
- Embedding models are not included in the repository.
- Large model files should be downloaded separately.
- The API is intended to run locally by default. To expose it to other machines, set `"host": "0.0.0.0"` in `config.json`.

---

## License

MIT License.
