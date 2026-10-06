"""Download the pinned models. Existing GGUF weights are verified and reused."""
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path(__file__).resolve().parent
filename = "Qwen3.5-4B-Q4_K_M.gguf"
model_path = root / "portablellm" / "models" / filename
expected_hash = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
if not model_path.exists():
    hf_hub_download(
        repo_id="unsloth/Qwen3.5-4B-GGUF", filename=filename,
        revision="e87f176479d0855a907a41277aca2f8ee7a09523",
        local_dir=model_path.parent,
    )
with model_path.open("rb") as stream:
    actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
if actual_hash != expected_hash:
    raise RuntimeError("GGUF SHA256 mismatch; inspect the file before retrying.")
print("GGUF SHA256 verified.")
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    local_dir=root / "portablellm" / "embedding_models" / "all-MiniLM-L6-v2",
    allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
)

